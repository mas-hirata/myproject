#%% # データの読込み

from datetime import timedelta

import japanize_matplotlib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prophet import Prophet
from scipy.stats import t
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm

# df = pd.read_csv(filepath_or_buffer=r'.\エリアプライス九州.csv')
# df.rename(columns={'受渡日時刻': 'Date', 'エリアプライス九州': 'y'}, inplace=True)
# df['Date'] = pd.to_datetime(df['Date'])

df = pd.read_csv(filepath_or_buffer=r'.\spot_summary_2024.csv', encoding='shift-jis', usecols=['受渡日', '時刻コード', 'エリアプライス九州(円/kWh)'], header=0)
df.rename(columns={'受渡日':'Date', 'エリアプライス九州(円/kWh)':'y'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
# 2) 時刻コード × 30 分 でタイムデルタを作成
df['Time'] = pd.to_timedelta((df['時刻コード'] - 1) * 30, unit='minutes')

# 3) 日付 + 時刻デルタ → 完全な日時
df['Date'] = df['Date'] + df['Time']
df.drop(columns=['Time', '時刻コード'], inplace=True)


#%% TODOリスト＆特徴量エンジニアリング
''' # TODOリスト
# TODOリスト
# 2. 予測ホライゾンに応じたラグ特徴量作成
# max_lagは説明変数として利用する「過去何日分」まで用意するかの上限です。

# TODO:prediction_daysの値に応じて使用できるlagの値は変わる？
#      特徴量の与え方に注意が必要
# NOTE:過去何日以降のレコードを使用したか、カラム名にわかるように記載する必要あり


# TODO:予測日より過去の自己情報を使う
    #    目的変数、Precited_dateの日付をshiftでずらして得る
'''


'''
def add_daily_lag_features(df, d,
                           date_col='Date'):

    """
    df に対して、日次集計した y の平均・最大・最小を
    そのまま d 日前の値として追加します。

    Parameters
    ----------
    df : pandas.DataFrame
        元の時系列データ（30分粒度など）
    d : int
        何日分前の値を使うか (shift の値)
    date_col : str
        df の日時列名
    value_col : str
        集計対象の値列名

    Returns
    -------
    out : pandas.DataFrame
        元の df に以下のカラムを追加して返します：
          - y_mean_lag{d}d：d 日前の日次平均
          - y_max_lag{d}d ：d 日前の日次最大
          - y_min_lag{d}d ：d 日前の日次最小
    """

    df['Date_day'] = df[date_col].dt.normalize()

    # 2) 日次集計
    daily = (
        df
        .groupby('Date_day')['y']
        .agg(
            y_mean='mean',
            y_max='max',
            y_min='min'
        )
        .sort_index()
    )

    # 3) d 日前の lag 特徴量
    daily[f'y_mean_lag{d}d'] = daily['y_mean'].shift(d)
    daily[f'y_max_lag{d}d']  = daily['y_max'].shift(d)
    daily[f'y_min_lag{d}d']  = daily['y_min'].shift(d)

    # 4) 元Df にマージ
    out = df.merge(
        daily[[f'y_mean_lag{d}d', f'y_max_lag{d}d', f'y_min_lag{d}d']],
        how='left',
        left_on='Date_day',
        right_index=True
    )

    # 5) 後処理：キー列を削除
    out.drop(columns=['Date_day'], inplace=True)
    return out
'''

#  TODO: 意味のある特徴量が生成されているか確認する
def add_daily_lags(
    df: pd.DataFrame,
    min_lag: int = 1,
    max_lag: int = 30,
    date_col: str = 'Date',
    value_col: str = 'y'
) -> pd.DataFrame:
    """
    df に対して、min_lag～max_lag 日前までの
    日次ラグを説明変数として追加します。

    Parameters
    ----------
    df : pd.DataFrame
        元データ。date_col と value_col を含むこと。
    min_lag : int
        何日前からラグを取るか（最小日数）。
    max_lag : int
        最大で何日前までラグを取るか（最大日数）。
    date_col : str
        時刻が入った日時列の名前。
    value_col : str
        集計対象の値列の名前。

    Returns
    -------
    pd.DataFrame
        元の df に、日次ラグ列 y_lag{n}d を追加したもの。
    """
    # コピー＆日時変換
    df_out = df.copy()
    df_out[date_col] = pd.to_datetime(df_out[date_col])

    # 元データの Series として準備
    ts = df_out.set_index(date_col)[value_col]

    # ラグの作成ループ
    for lag in range(min_lag, max_lag + 1):
        col_name = f'{value_col}_lag{lag}d'
        # インデックスを lag 日だけ先にずらす
        shifted = ts.rename(col_name)
        shifted.index = shifted.index + pd.Timedelta(days=lag)
        # マージしてラグ列を追加
        df_out = df_out.merge(
            shifted.to_frame(),
            left_on=date_col,
            right_index=True,
            how='left'
        )

    return df_out


def add_time_aggregated_features(
    df,
    d=1,
    date_col='Date',
    value_col='y'
):
    """
    df に対して以下の特徴量を追加します：
      (1) 過去 d 日より前かつ直近 w 日間の生データ統計
          → w = 14, 7, 3, 2 日
      (2) 日次集計（mean,max,min,std）からの d 日前 lag
      (3) 曜日ごと・時刻ごと（30分刻み）に、
          過去 d 日より前かつ直近 w 日間の統計
    """

    df_feat = df.copy()
    df_feat[date_col] = pd.to_datetime(df_feat[date_col])
    df_feat = df_feat.sort_values(date_col).reset_index(drop=True)

    # 元系列と、d 日シフト版を用意
    ts = df_feat.set_index(date_col)[value_col]
    ts_shifted = ts.copy()
    ts_shifted.index = ts_shifted.index + pd.Timedelta(days=d)
    ts_shifted = ts_shifted.sort_index()

    # 集計ウィンドウ長リスト TODO: stdだとデータが1や3の時エラーが発生する
    windows = [21, 14, 7]

    # ── (1) 生データ統計 ─────────────────────────
    for w in windows:
        stats = ts_shifted.rolling(f'{w}D', closed='both') \
                          .agg(['mean', 'max', 'min', 'std'])
        stats.columns = [
            f'{value_col}_past_{stat}_{d}d_{w}d'
            for stat in ['mean', 'max', 'min', 'std']
        ]
        df_feat = df_feat.merge(
            stats,
            left_on=date_col, right_index=True, how='left'
        )

    # ── (2) 日次集計 & d 日前 lag ────────────────────
    df_feat['Date_day'] = df_feat[date_col].dt.normalize()
    daily = (
        df_feat
        .groupby('Date_day')[value_col]
        .agg(['mean', 'max', 'min', 'std'])
        .rename(columns={
            'mean': f'{value_col}_day_mean',
            'max':  f'{value_col}_day_max',
            'min':  f'{value_col}_day_min',
            'std':  f'{value_col}_day_std',
        })
        .sort_index()
    )
    for stat in ['mean', 'max', 'min', 'std']:
        daily[f'{value_col}_day_{stat}_lag{d}d'] = \
            daily[f'{value_col}_day_{stat}'].shift(d)
    df_feat = df_feat.merge(
        daily[[f'{value_col}_day_{stat}_lag{d}d'
               for stat in ['mean', 'max', 'min', 'std']]],
        left_on='Date_day', right_index=True, how='left'
    )
    df_feat.drop(columns=['Date_day'], inplace=True)

    # ── ヘルパー：groupby + rolling(window) ───────────
    def _group_rolling(ts_shifted, grouper, prefix):
        result = pd.DataFrame(index=ts_shifted.index)
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
            result = result.join(stats)
        return result

    # ── (3) 曜日ごとの統計 ────────────────────────────
    weekday_stats = _group_rolling(
        ts_shifted,
        ts_shifted.index.weekday,
        f'{value_col}_weekday'
    )
    df_feat = df_feat.merge(
        weekday_stats,
        left_on=date_col, right_index=True, how='left'
    )

    # ── (4) 時刻ごとの統計（30分刻み） ─────────────────
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
    指定した日時列から、年・月・日・曜日・時間・分・秒・週末フラグ・季節などの特徴量を抽出します。

    Parameters:
    - df : pandas.DataFrame
    - datetime_col : str, 日時が格納されたカラム名（デフォルト: 'Date'）

    Returns:
    - 元のDataFrameに新たな特徴量を追加したもの
    """
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    df['year'] = df[datetime_col].dt.year
    df['month'] = df[datetime_col].dt.month
    df['day'] = df[datetime_col].dt.day
    df['weekday'] = df[datetime_col].dt.weekday  # 月=0, 日=6
    df['hour'] = df[datetime_col].dt.hour
    df['minute'] = df[datetime_col].dt.minute
    # df['second'] = df[datetime_col].dt.second

    # 日内インデックス（30分粒度）
    df['minute_index'] = df['hour'] * 2 + (df['minute'] // 30)

    # 週末かどうか（0=False, 1=True）
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)

    # 季節（spring=1, summer=2, autumn=3, winter=4）
    def get_season(month):
        if month in [3, 4, 5]:
            return 1  # 春
        elif month in [6, 7, 8]:
            return 2  # 夏
        elif month in [9, 10, 11]:
            return 3  # 秋
        else:
            return 4  # 冬

    df['season'] = df['month'].apply(get_season)

    return df


#%% model ver4

# ─────────────── 特徴量生成関数は省略 ────────────────
# add_daily_lags(...)
# add_time_aggregated_features(...)
# extract_datetime_features(...)
max_horizon = 14

# get_prediction_interval(res, X_train_with_const, y_train, X_new, ci=0.95)

# def get_prediction_interval(model, X_train, y_train, X_new, ci=0.95):

#     X_train_mat = np.asarray(X_train)
#     X_new_mat = np.asarray(X_new)
#     y_train_arr = np.asarray(y_train)

#     # Extract parameters
#     params = np.asarray(model.params)

#     # Number of observations and parameters
#     n, p = X_train_mat.shape

#     # Predicted on training to compute residual variance
#     y_fit = X_train_mat @ params
#     RSS = np.sum((y_train_arr - y_fit) ** 2)
#     sigma2 = RSS / (n - p)

#     # Inverse of (X'X)
#     # inv_XTX = np.linalg.inv(X_train_mat.T @ X_train_mat)

#     XTX = X_train_mat.T @ X_train_mat
#     try:
#         inv_XTX = np.linalg.inv(XTX)
#     except np.linalg.LinAlgError:
#         # 特異行列の場合は擬似逆行列を使う
#         inv_XTX = np.linalg.pinv(XTX)

#     # Predictions on new data
#     y_pred_new = X_new_mat @ params

#     # Compute variance for each new point
#     var_term = np.array([x @ inv_XTX @ x for x in X_new_mat])
#     var_term = np.clip(var_term, 0, None)

#     df = n - p
#     t_val = t.ppf((1 + ci) / 2, df)
#     se_pred = np.sqrt(sigma2 * (1 + var_term))

#     # Interval bounds
#     lower = y_pred_new - t_val * se_pred
#     upper = y_pred_new + t_val * se_pred

#     return y_pred_new, lower, upper

def forecast_with_model(
    df,
    model_type: str,
    pred_start: pd.Timestamp,
    pred_end: pd.Timestamp,
):

    # 予測日リスト
    prediction_dates = (
        df.loc[(df['Date'] >= pred_start) & (df['Date'] < pred_end), 'Date']
        .dt.normalize().drop_duplicates()
        )
    # global fi_list
    results = []
    fi_list = []

    # ─── 2) 予測ループ ─────────────────────────────────
    for pred_date in prediction_dates:
        last_date = df['Date'].max()
        horizon = min(max_horizon, (last_date - pred_date).days)
        if horizon < 1:
            continue

        # for d in range(1, horizon+1): NOTE
        for d in [1, 14]:

            # ■ Target 列だけシフト
            # ─── 1) 特徴量はループ外で一度だけ作成 ───────────
            df_feat = df.copy()
            df_feat = add_daily_lags(df_feat, min_lag=1, max_lag=30)
            # max_horizon 日分の rolling 統計を一度に生成
            df_feat = add_time_aggregated_features(df_feat, d=1)
            df_feat = extract_datetime_features(df_feat, datetime_col='Date')

            df_loop = df_feat.copy()
            df_loop['Target'] = df_loop['y'].shift(-d*48)

            # ■ 学習／検証期間フィルタ
            df_start = pred_date - pd.DateOffset(months=8)

            # df_end = pred_date + timedelta(days=d)
            # df_loop = df_loop[(df_loop['Date'] >= df_start) & (df_loop['Date'] < df_end)]
            df_loop = (df_loop[(df_loop['Date'] >= df_start) &
                               (df_loop['Date'] <= (pred_date + timedelta(hours=23, minutes=30)))])

            # global train
            # global valid
            # global model
            global X_train, y_train, X_valid
            global model, feature_cols
            global X_train_with_const, X_new

            train = df_loop[df_loop['Date'] < pred_date].dropna()
            valid = df_loop[(df_loop['Date'].dt.date == pred_date.date())]
            # print(valid)
            # valid = df_loop[(df_loop['Date'] >= pred_date) &
            #                 (df_loop['Date'] < pred_date + timedelta(days=d))].dropna()

            if train.empty or valid.empty:
                continue

            feature_cols = [c for c in train.columns if c not in ['Date', 'y', 'Target']]
            X_train, y_train = train[feature_cols], train['Target']
            X_valid, _ = valid[feature_cols], valid['Target']

            # ■ モデル切り替え
            if model_type == 'lgbm':
                model = lgb.LGBMRegressor()
                model_lower = lgb.LGBMRegressor(objective='quantile', alpha=0.05)
                model_upper = lgb.LGBMRegressor(objective='quantile', alpha=0.95)
                model.fit(X_train, y_train); model_lower.fit(X_train, y_train); model_upper.fit(X_train, y_train)
                preds, preds_lower, preds_upper = model.predict(X_valid), model_lower.predict(X_valid), model_upper.predict(X_valid) # 重要度
                for feat, imp in zip(feature_cols, model.feature_importances_):
                    fi_list.append({'model_type': model_type, 'horizon': d,
                                    'feature': feat, 'imp_coef': imp,
                                    'Date': valid['Date'].iloc[0].date(),
                                    'forecast_date': valid['Date'].iloc[0] + pd.to_timedelta(d, unit='D'),})

            elif model_type == 'lr':
                # X_train_with_const = sm.add_constant(X_train, has_constant='add')
                # # L1_wt = 0.0：Ridge回帰（L2正則化）、L1_wt = 1.0：Lasso回帰（L1正則化）、L1_wt = 0.0~1.0：Elastic_Net回帰（L1とL2の混合）
                # # model = sm.OLS(y_train, X_train_with_const).fit_regularized(method='elastic_net', alpha=1.0, L1_wt=0.5)
                # # TODO：予測区間の実装
                # model = sm.OLS(y_train, X_train_with_const).fit_regularized(method='elastic_net', alpha=1.0, L1_wt=0.5)
                # X_new = sm.add_constant(X_valid, has_constant='add')

                # lower_q, upper_q = 0.05, 0.95
                # qr_low = sm.QuantReg(y_train, X_train_with_const).fit(q=lower_q)
                # qr_up  = sm.QuantReg(y_train, X_train_with_const).fit(q=upper_q)
                # model = sm.OLS(y_train, X_train_with_const).fit()
                # pred_df = model.get_prediction(sm.add_constant(X_valid, has_constant='add')).summary_frame(alpha=0.05)

                # preds, preds_lower, preds_upper = pred_df['mean'].to_numpy(), qr_low.predict(X_new), qr_up.predict(X_new)

                X_train_with_const = sm.add_constant(X_train, has_constant='add')
                model = sm.OLS(y_train, X_train_with_const).fit()
                pred_df = model.get_prediction(sm.add_constant(X_valid, has_constant='add')).summary_frame(alpha=0.05)
                preds, preds_lower, preds_upper = pred_df['mean'].to_numpy(), pred_df['obs_ci_lower'].to_numpy(), pred_df['obs_ci_upper'].to_numpy()

                # 係数
                for feat, coef in zip(feature_cols, model.params.to_numpy()):
                    fi_list.append({'model_type': model_type, 'horizon': d,
                                    'feature': feat, 'imp_coef': coef,
                                    'Date': valid['Date'].iloc[0].date(),
                                    'forecast_date': valid['Date'].iloc[0] + pd.to_timedelta(d, unit='D'),})

            elif model_type == 'prophet':
                # Prophet に説明変数を全投入して回帰係数も取得
                df_prop = train[['Date', 'Target']].rename(columns={'Date': 'ds', 'Target': 'y'})
                m = Prophet(daily_seasonality=True, weekly_seasonality=True)
                # for reg in feature_cols:
                # m.add_regressor(reg)
                # df_prop[reg] = train[reg]
                m.fit(df_prop)

                future = m.make_future_dataframe(periods=d*48, freq='30min')
                future = future.merge(
                    valid[['Date']+feature_cols]
                    .rename(columns={'Date': 'ds'}),
                    on='ds', how='left'
                )
                fcst = m.predict(future)
                preds = fcst.set_index('ds').loc[valid['Date'], 'yhat'].values

                # 回帰係数 β を取得
                betas = m.params['beta'][:, -1]
                for feat, b in zip(feature_cols, betas):
                    fi_list.append({'model': model_type, 'horizon': d,
                                    'feature': feat, 'coefficient': b})

                # # 周期性のグラフはここで毎回表示
                # fig = m.plot_components(fcst)
                # fig.suptitle(f'Prophet Components (horizon={d}d)')
                # plt.show()

            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            # ■ 予測結果格納

            out = valid[['Date', 'Target']].copy().assign(
                predicted=preds,
                predicted_lower=preds_lower,
                predicted_upper=preds_upper,
                model_type=model_type,
                horizon=d,
                error=valid['Target'] - preds,
                forecast_date=valid['Date'] + pd.to_timedelta(d, unit='D')
                )

            results.append(out)

    fi_df = pd.DataFrame(fi_list)
    fi_df['Date'] = pd.to_datetime(fi_df['Date'])
    fi_df['forecast_date'] = pd.to_datetime(fi_df['forecast_date'])

    return pd.concat(results, ignore_index=True), pd.DataFrame(fi_list)

# 予測期間の設定
pred_start = pd.to_datetime('2024-12-01')
pred_end = pd.to_datetime('2025-02-01')

# モデルの種類リスト
# model_types = ['lgbm', 'lr', 'prophet']

model_types = ['lr', 'lgbm']
# model_types = ['lgbm']

# 結果をためておく辞書を用意
all_results = {}
all_fi = {}
all_metrics_h = {}
all_metrics_m = {}
all_overall = {}
results_list = []
results_list_fi = []

for model_type in model_types:
    print(f"\n===== Running model: {model_type} =====")

    res_df, fi_df = forecast_with_model(
        df,
        model_type=model_type,
        pred_start=pred_start,
        pred_end=pred_end
    )


    res_df = res_df.copy()
    # res_df['model_type'] = model_type
    results_list.append(res_df)

    # fi_df['model_type'] = model_type
    results_list_fi.append(fi_df)

df_all = pd.concat(results_list, ignore_index=True)

metrics_horizon = (
    df_all
    .groupby(['model_type', 'horizon'])
    .apply(lambda g: pd.Series({
        'RMSE': mean_squared_error(g['Target'], g['predicted']),
        'MAE':  mean_absolute_error(g['Target'], g['predicted'])
    }))
    .reset_index()
)

df_all['forecast_month'] = df_all['forecast_date'].dt.to_period('M')
metrics_monthly = (
    df_all
    .groupby(['model_type', 'horizon', 'forecast_month'])
    .apply(lambda g: pd.Series({
        'RMSE': mean_squared_error(g['Target'], g['predicted']),
        'MAE':  mean_absolute_error(g['Target'], g['predicted'])
    }))
    .reset_index()
)

metrics_overall = (
    df_all
    .groupby('model_type')
    .apply(lambda g: pd.Series({
        'RMSE': mean_squared_error(g['Target'], g['predicted']),
        'MAE':  mean_absolute_error(g['Target'], g['predicted'])
    }))
    .reset_index())

m_overall = metrics_overall.copy()
m_overall['horizon'] = 'all'
m_overall['forecast_month'] = 'all'

m_horizon = metrics_horizon.copy()
m_horizon['forecast_month'] = 'all'

# 3) metrics_monthly はそのまま（forecast_month がある）
m_monthly = metrics_monthly.copy()

# 4) 上記３つを縦に結合
metrics_combined = pd.concat(
    [m_overall, m_horizon, m_monthly],
    ignore_index=True,
    sort=False
)

metrics_combined = (metrics_combined[['model_type', 'horizon', 'forecast_month', 'RMSE', 'MAE']]
                    .sort_values(by=['model_type', 'horizon', 'forecast_month'])
                    .reset_index(drop=True))


# *****************************************************************************************
# horizonごとの集計

df_all_fi = pd.concat(results_list_fi, ignore_index=True)
df_all_fi['forecast_month'] = df_all_fi['forecast_date'].dt.to_period('M')

# horizonごとの集計
imp_coef_horizon = (
    df_all_fi.groupby(['feature', 'model_type', 'horizon'])
    .apply(lambda g: pd.Series({
        'mean_imp_coef': g['imp_coef'].mean(),
        # 'mean_coefficient': g['coefficient'].mean() if 'coefficient' in g else None
    }))
    .reset_index()
)

# 月ごとの集計
imp_coef_monthly = (
    df_all_fi.groupby(['feature', 'model_type', 'horizon', 'forecast_month'])
    .apply(lambda g: pd.Series({
        'mean_imp_coef': g['imp_coef'].mean(),
        # 'mean_coefficient': g['coefficient'].mean() if 'coefficient' in g else None
    }))
    .reset_index()
)

# モデル全体での集計
imp_coef_overall = (
    df_all_fi.groupby(['feature', 'model_type'])
    .apply(lambda g: pd.Series({
        'mean_imp_coef': g['imp_coef'].mean(),
        # 'mean_coefficient': g['coefficient'].mean() if 'coefficient' in g else None
    }))
    .reset_index()
)

# モデル全体の集計
# m_overall = metrics_overall.copy()
imp_coef_overall['horizon'] = 'all'
imp_coef_overall['forecast_month'] = 'all'

# horizonごとの集計
# m_horizon = metrics_horizon.copy()
imp_coef_horizon['forecast_month'] = 'all'

# 月ごとの集計
# m_monthly = metrics_monthly.copy()

# 3つの集計結果を縦に結合
metrics_combined = pd.concat([imp_coef_overall, imp_coef_horizon, imp_coef_monthly], ignore_index=True, sort=False)

# 最終結果を整形
# metrics_combined = (metrics_combined[['feature', 'model_type', 'horizon', 'forecast_month', 'mean_importance', 'mean_coefficient']]
#                     .sort_values(by=['feature', 'model_type', 'horizon', 'forecast_month'])
#                     .reset_index(drop=True))

# #%%
# metrics_combined['abs_mean_coefficient'] = metrics_combined['mean_coefficient'].abs()
# lgbm_fi_top10 = metrics_combined[(metrics_combined['horizon']=='all') & (metrics_combined['model_type']=='lgbm')].sort_values(by='mean_importance', ascending=False)['feature'].head(10).to_list()
# lr_fi_top10 = metrics_combined[(metrics_combined['horizon']=='all') & (metrics_combined['model_type']=='lr')].sort_values(by='abs_mean_coefficient', ascending=False)['feature'].head(10).to_list()
# df_fi_10 = metrics_combined[(metrics_combined['feature'].isin(lgbm_fi_top10 + lr_fi_top10)) & (metrics_combined['forecast_month']=='all')]


# TODO:複数のモデルが作成されるが、それらのモデルの特徴量重要度、偏回帰係数をまとめる

#%%

# metrics_combined['abs_coefficient']

metrics_combined['abs_mean_coefficient'] = metrics_combined['mean_coefficient'].abs()

#%%


def plot_actual_vs_predicted(all_results, model_type, horizon=1, month=1):
    """
    all_results: dict of DataFrame, all_results['lgbm'], all_results['lr'], ...
    model_type:  'lgbm' or 'lr' or 'prophet'
    horizon:     何日先か (デフォルト 1)
    month:       何月分を出すか (デフォルト 1=Jan)
    """
    df = all_results[all_results['model_type']==model_type]
    # 指定月分だけ
    df_month = df[df['forecast_date'].dt.month == month]
    # 指定ホライゾンだけ
    df_plot = df_month[df_month['horizon'] == horizon]
    if df_plot.empty:
        print(f"No data for {model_type}, horizon={horizon}, month={month}")
        return

    plt.figure(figsize=(12, 5))
    # plt.plot(df_plot['forecast_date'], df_plot['Target'],    label='Actual')
    plt.fill_between(x=df_plot['forecast_date'], y1=df_plot['predicted_lower'], y2=df_plot['predicted_upper']
                     , alpha=0.2, label='95% 予測区間', linestyle='--')
    plt.plot(df_plot['forecast_date'], df_plot['predicted'], label='Predicted', linestyle='--')
    plt.plot(df_plot['forecast_date'], df_plot['Target'], label='Actual', linestyle='--')
    # plt.xlabel('DateTime')
    # plt.ylabel('y value')
    plt.title(f"{model_type.upper()}: Actual vs Predicted (horizon={horizon}, month={month})")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# 使い方例
# January, horizon=1 で LGBM
plot_actual_vs_predicted(df_all, model_type='lgbm', horizon=1, month=1)
# plot_actual_vs_predicted(df_all, model_type='lgbm', horizon=14, month=1)

# 同じく LinearRegression
plot_actual_vs_predicted(df_all, model_type='lr',   horizon=1, month=1)
# plot_actual_vs_predicted(df_all, model_type='lr',   horizon=14, month=1)

# plot_actual_vs_predicted(all_results, model_type='prophet',   horizon=1, month=1)
# plot_actual_vs_predicted(all_results, model_type='prophet',   horizon=14, month=1)

#%%

import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み
# metrics_combined = pd.read_csv('/mnt/data/Cell output 43 [DW].csv')

# フィルタリング: horizonがall, 1, 14 かつ forecast_monthがall
metrics_filtered = metrics_combined[
    (metrics_combined['horizon'].isin(['all', '1', '14'])) &
    (metrics_combined['forecast_month'] == 'all')
]

# LightGBM の上位10特徴量
df_lgbm = metrics_filtered[metrics_filtered['model_type'] == 'lgbm']
top10_lgbm = df_lgbm.nlargest(15, 'mean_imp_coef')
plt.figure()
plt.bar(top10_lgbm['feature'], top10_lgbm['mean_imp_coef'])
plt.xticks(rotation=45, ha='right')
plt.ylabel('Feature Importance (mean_imp_coef)')
plt.title('LightGBM 上位10特徴量')
plt.tight_layout()
plt.show()

# 線形回帰 の上位10特徴量（係数の絶対値でソート）
df_lr = metrics_filtered[metrics_filtered['model_type'] == 'lr'].copy()
df_lr['abs_coef'] = df_lr['mean_imp_coef'].abs()
top10_lr = df_lr.nlargest(15, 'abs_coef')
plt.figure()
plt.bar(top10_lr['feature'], top10_lr['mean_imp_coef'])
plt.xticks(rotation=45, ha='right')
plt.ylabel('Coefficient Magnitude')
plt.title('線形回帰 上位10特徴量')

plt.tight_layout()
plt.show()

