#%%

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from datetime import timedelta

import pandas as pd
import numpy as np
import sys 
import lightgbm as lgb
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from datetime import timedelta
import optuna
import lightgbm as lgb
from sklearn.metrics import mean_squared_error


# ダミーデータ作成（30分粒度のデータ）
dates = pd.date_range(start='2024-04-01 00:00:00', end='2025-03-31 23:30:00', freq='30min')
np.random.seed(42)
df = pd.DataFrame({
    'Date': dates,
    'y': np.random.rand(len(dates)) * 100  # 0～100の乱数
})

df = df.sort_values('Date').reset_index(drop=True)

# =============================================================================
# 1) 前提：df に 'Date' 列と 'y' 列がある DataFrame がロード済み
# =============================================================================
# 例:
# df = pd.read_csv('your_data.csv', parse_dates=['Date'])

# =============================================================================
# 2) 特徴量生成関数の定義
# =============================================================================

def add_daily_lags(df, min_lag, max_lag, date_col='Date', value_col='y'):
    """
    min_lag～max_lag 日前までのラグ値を追加。
    """
    df_out = df.copy()
    df_out[date_col] = pd.to_datetime(df_out[date_col])
    ts = df_out.set_index(date_col)[value_col]
    for lag in range(min_lag, max_lag + 1):
        col = f'{value_col}_lag{lag}d'
        shifted = ts.rename(col)
        shifted.index = shifted.index + pd.Timedelta(days=lag)
        df_out = df_out.merge(
            shifted.to_frame(),
            left_on=date_col,
            right_index=True,
            how='left'
        )
    return df_out

def add_time_aggregated_features(df, d, windows=[14,7,3,2],
                                 date_col='Date', value_col='y'):
    """
    生データの rolling 統計を windows 日間で複数作成。
    同じく日次 lag、曜日・時刻別 rolling。
    """
    df_feat = df.copy()
    df_feat[date_col] = pd.to_datetime(df_feat[date_col])
    df_feat = df_feat.sort_values(date_col).reset_index(drop=True)

    # 原系列と d 日先シフト版
    ts = df_feat.set_index(date_col)[value_col]
    ts_shifted = ts.copy()
    ts_shifted.index = ts_shifted.index + pd.Timedelta(days=d)
    ts_shifted = ts_shifted.sort_index()

    # (1) 生データ rolling 統計
    for w in windows:
        stats = ts_shifted.rolling(f'{w}D', closed='both') \
                          .agg(['mean','max','min','std'])
        stats.columns = [
            f'{value_col}_past_{stat}_{d}d_{w}d'
            for stat in ['mean','max','min','std']
        ]
        df_feat = df_feat.merge(
            stats,
            left_on=date_col, right_index=True, how='left'
        )

    # (2) 日次集計 & d 日前 lag
    df_feat['Date_day'] = df_feat[date_col].dt.normalize()
    daily = (
        df_feat
        .groupby('Date_day')[value_col]
        .agg(['mean','max','min','std'])
        .rename(columns={
            'mean': f'{value_col}_day_mean',
            'max':  f'{value_col}_day_max',
            'min':  f'{value_col}_day_min',
            'std':  f'{value_col}_day_std',
        })
        .sort_index()
    )
    for stat in ['mean','max','min','std']:
        daily[f'{value_col}_day_{stat}_lag{d}d'] = \
            daily[f'{value_col}_day_{stat}'].shift(d)
    df_feat = df_feat.merge(
        daily[[f'{value_col}_day_{stat}_lag{d}d'
               for stat in ['mean','max','min','std']]],
        left_on='Date_day', right_index=True, how='left'
    )
    df_feat.drop(columns=['Date_day'], inplace=True)

    # 共通ヘルパー：groupby + rolling
    def _group_rolling(ts_shifted, grouper, prefix):
        df_res = pd.DataFrame(index=ts_shifted.index)
        for w in windows:
            stats = (
                ts_shifted
                .groupby(grouper)
                .rolling(f'{w}D', closed='both')
                .agg(['mean','max','min','std'])
            )
            stats.index = stats.index.droplevel(0)
            stats.columns = [
                f'{prefix}_{stat}_{d}d_{w}d'
                for stat in ['mean','max','min','std']
            ]
            df_res = df_res.join(stats)
        return df_res

    # (3) 曜日ごと
    weekday_stats = _group_rolling(
        ts_shifted,
        ts_shifted.index.weekday,
        f'{value_col}_weekday'
    )
    df_feat = df_feat.merge(
        weekday_stats,
        left_on=date_col, right_index=True, how='left'
    )

    # (4) 時刻（30分刻み）ごと
    time_stats = _group_rolling(
        ts_shifted,
        ts_shifted.index.time,
        f'{value_col}_time'
    )
    df_feat = df_feat.merge(
        time_stats,
        left_on=date_col, right_index=True, how='left'
    )

    return df_feat

def extract_datetime_features(df, datetime_col='Date'):
    """
    時間・分・曜日を追加する簡易関数
    """
    df_out = df.copy()
    dt = pd.to_datetime(df_out[datetime_col])
    df_out['hour']    = dt.dt.hour
    df_out['minute']  = dt.dt.minute
    df_out['weekday'] = dt.dt.weekday
    return df_out



#%% 全てのdについてハイパラチューニング

pred_start = pd.to_datetime('2024-12-01')
pred_end   = pd.to_datetime('2025-02-01')
last_date = df['Date'].max()  # 例: 2025-03-31
prediction_dates = df.loc[(df['Date'] >= pred_start) & (df['Date'] < pred_end), 'Date'] \
                     .dt.normalize().drop_duplicates()

best_results = {}  # d -> {'rmse':..., 'params':...}

for d in range(1, 15):  # 1日先～14日先まで
    print(f"\n=== Horizon: {d} days ahead ===")

    # Optuna 目的関数
    def objective(trial):
        # 3-1) 探索空間
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 20, 200),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.6, 1.0),
            'bagging_fraction':  trial.suggest_uniform('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        }

        rmses = []
        for pred_date in prediction_dates:
            # --- データ準備 ---
            df_data = df.copy()
            df_data['Target'] = df_data['y'].shift(-d*48)

            # 特徴量生成
            df_data = add_daily_lags(df_data, min_lag=1, max_lag=30)
            df_data = add_time_aggregated_features(df_data, d=d)
            df_data = extract_datetime_features(df_data)

            # 学習範囲のフィルタリング
            df_start = pred_date - pd.DateOffset(months=8)
            df_end   = pred_date + timedelta(days=d)
            df_data = df_data[(df_data['Date'] >= df_start) &
                              (df_data['Date'] < df_end)]

            # トレイン/バリデーション分割 & dropna
            train = df_data[df_data['Date'] < pred_date].dropna()
            valid = df_data[(df_data['Date'] >= pred_date) &
                            (df_data['Date'] < pred_date + timedelta(days=d))] \
                            .dropna()
            if train.empty or valid.empty:
                continue

            feature_cols = [c for c in train.columns
                            if c not in ['Date','y','Target']]
            X_train, y_train = train[feature_cols], train['Target']
            X_valid, y_valid = valid[feature_cols], valid['Target']

            # --- モデル学習＆予測 ---
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_valid)

            rmses.append(mean_squared_error(y_valid, preds, squared=False))

        return np.mean(rmses)

    # Study 作成＆最適化
    study = optuna.create_study(direction='minimize', study_name=f"horizon_{d}d")
    study.optimize(objective, n_trials=50)

    # 結果保存
    best_results[d] = {
        'rmse':   study.best_value,
        'params': study.best_params
    }

    # 結果表示
    print(f">>> Best RMSE for d={d}: {study.best_value:.4f}")
    print(f">>> Best params for d={d}: {study.best_params}")

# ====================================
# 4) 全ホライゾン結果のまとめ表示
# ====================================
print("\n=== Summary of hyperparameters per horizon ===")
for d, res in best_results.items():
    print(f"d = {d:2d} → RMSE = {res['rmse']:.4f}")
    for k, v in res['params'].items():
        print(f"    {k}: {v}")
    print()

#%%
'''
# =============================================================================
# 3) 予測日・ホライゾン d の設定
# =============================================================================
d = 3  # 固定で 3 日後予測

pred_start = pd.to_datetime('2024-12-01')
pred_end   = pd.to_datetime('2025-02-01')
last_date = df['Date'].max()  # 例: 2025-03-31
prediction_dates = df.loc[(df['Date'] >= pred_start) & (df['Date'] < pred_end), 'Date'] \
                     .dt.normalize().drop_duplicates()

# =============================================================================
# 4) Optuna 目的関数の定義
# =============================================================================
def objective(trial):
    # 4-1) 探索するパラメータ空間
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.6, 1.0),
        'bagging_fraction':  trial.suggest_uniform('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
    }

    rmses = []  # 各予測日の RMSE を蓄積
    for pred_date in prediction_dates:
        # --- データ準備 ---
        df_data = df.copy()
        # d 日後の目標 y を Target 列に
        df_data['Target'] = df_data['y'].shift(-d*48)

        # 特徴量生成
        df_data = add_daily_lags(df_data, min_lag=1, max_lag=30,
                                 date_col='Date', value_col='y')
        df_data = add_time_aggregated_features(df_data, d=d,
                                               date_col='Date', value_col='y')
        df_data = extract_datetime_features(df_data, datetime_col='Date')

        # 学習＋検証対象の期間に切り出し
        df_start = pred_date - pd.DateOffset(months=8)
        df_end   = pred_date + timedelta(days=d)
        df_data = df_data[(df_data['Date'] >= df_start) &
                          (df_data['Date'] < df_end)]

        # トレイン/バリデーション分割
        train = df_data[df_data['Date'] < pred_date]
        valid = df_data[(df_data['Date'] >= pred_date) &
                        (df_data['Date'] < pred_date + timedelta(days=d))] \
                        
        if train.empty or valid.empty:
            continue

        # 説明変数・目的変数
        feature_cols = [c for c in train.columns
                        if c not in ['Date','y','Target']]
        X_train, y_train = train[feature_cols], train['Target']
        X_valid, y_valid = valid[feature_cols], valid['Target']

        # --- モデル学習 & 予測 ---
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)

        # RMSE を計算
        rmse = mean_squared_error(y_valid, preds)
        rmses.append(rmse)

    # 予測日ごとの平均 RMSE を返す
    return np.mean(rmses)

# =============================================================================
# 5) 最適化の実行
# =============================================================================
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print("Best RMSE:", study.best_value)
print("Best params:", study.best_params)
'''
#%%

'''
# =============================================================================
# 6) 最良パラメータで最終モデルを学習
# =============================================================================
best_params = study.best_params
final_model = lgb.LGBMRegressor(**best_params)
# 全 12/1～3/31 のデータを使う例
full_train = df[df['Date'] < last_date].dropna()
X_full = add_time_aggregated_features(
             add_daily_lags(full_train,1,30),
             d=d
         ).pipe(extract_datetime_features, datetime_col='Date') \
          .drop(columns=['Date','y'])
y_full = full_train['y'].shift(-d*48).dropna()
final_model.fit(X_full, y_full)

# → final_model で未来予測などに進めます
'''