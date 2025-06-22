import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# 例: テストデータに対して過去7日間の特徴量を生成
test_date_idx = pd.date_range(start='2024-04-01', periods=365*48, freq='30min')
test_minutes = test_date_idx.hour * 60 + test_date_idx.minute
test_seasonal = np.sin(2 * np.pi * test_minutes / (24*60))
test_linear_trend = np.linspace(0, 1, len(test_date_idx))
test_noise = 0.1 * np.random.randn(len(test_date_idx))

# テストデータ作成
test_df = pd.DataFrame({'Date': test_date_idx,
                        'value': test_seasonal + test_linear_trend + test_noise})


add_time_aggregated_features(
    test_df,
    d=1,
    date_col='Date',
    value_col='value'
)