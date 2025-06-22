#%%

import pandas as pd
import math
import numpy as np
import pandas as pd

# ダミーデータ作成（30分粒度のデータ）
dates = pd.date_range(start='2024-04-01 00:00:00', end='2025-03-31 23:30:00', freq='30min')
np.random.seed(42)
df = pd.DataFrame({
    'Date': dates,
    'y': np.random.rand(len(dates)) * 100  # 0～100の乱数
})

df = df.sort_values('Date').reset_index(drop=True)

#%%

import pandas as pd

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

    # 集計ウィンドウ長リスト
    windows = [14, 7, 3, 2]

    # ── (1) 生データ統計 ─────────────────────────
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

    # ── (2) 日次集計 & d 日前 lag ────────────────────
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


#%%

# def add_time_aggregated_features(df, d, date_col='Date', value_col='y'): old
    """
    df に対して以下の特徴量を追加します：
      1) 過去 d 日より前の全履歴を用いた統計量 (mean, max, min, std)
      2) 日次集計（mean, max, min, std）からの d 日前 lag
      3) 各曜日・各月・各時間ごとに、過去 d 日より前の履歴を用いた統計量 (mean, max, min, std)
    
    Parameters
    ----------
    df : pandas.DataFrame
    d : int
        何日前より前のデータを「過去」とみなすか
    date_col : str
        日時列名
    value_col : str
        集計対象の値列名
    """
    # ■ 前処理
    df_feat = df.copy()
    df_feat[date_col] = pd.to_datetime(df_feat[date_col])
    # タイムソート＆リセットインデックス
    df_feat = df_feat.sort_values(date_col).reset_index(drop=True)

    # ■ 原系列を取得
    ts = df_feat.set_index(date_col)[value_col]

    # ■ (1) 過去 d 日より前の全履歴統計量
    # TODO: 一定の期間を設ける、全ての期間とするとあまり意味がない気がする
    ts_shifted = ts.copy()
    ts_shifted.index = ts_shifted.index + pd.Timedelta(days=d)
    ts_shifted = ts_shifted.sort_index()

    past_stats = ts_shifted.expanding().agg(['mean','max','min','std'])
    past_stats.columns = [
        f'{value_col}_past_{stat}_{d}d' for stat in ['mean','max','min','std']
    ]
    df_feat = df_feat.merge(
        past_stats,
        left_on=date_col, right_index=True, how='left'
    )

    # ■ (2) 日次集計 & d 日前 lag
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
        daily[f'{value_col}_day_{stat}_lag{d}d'] = daily[f'{value_col}_day_{stat}'].shift(d)
    df_feat = df_feat.merge(
        daily[[f'{value_col}_day_{stat}_lag{d}d' for stat in ['mean','max','min','std']]],
        left_on='Date_day', right_index=True, how='left'
    )
    df_feat.drop(columns=['Date_day'], inplace=True)

    # Helper for dynamic groupby + expanding merge
    def _expand_group(ts_shifted, grouper, prefix):
        stats = (
            ts_shifted
            .groupby(grouper)
            .expanding()
            .agg(['mean','max','min','std'])
        )
        # MultiIndex (group, time) → drop group level
        stats.index = stats.index.droplevel(0)
        stats.columns = [
            f'{prefix}_{stat}_past{d}d' for stat in ['mean','max','min','std']
        ]
        return stats

    # ■ (3) 曜日ごとの過去 d 日より前統計量
    weekday_stats = _expand_group(ts_shifted, ts_shifted.index.weekday, f'{value_col}_weekday')
    df_feat = df_feat.merge(
        weekday_stats,
        left_on=date_col, right_index=True, how='left'
    )

    # ■ (4) 月ごとの過去 d 日より前統計量
    month_stats = _expand_group(ts_shifted, ts_shifted.index.month, f'{value_col}_month')
    df_feat = df_feat.merge(
        month_stats,
        left_on=date_col, right_index=True, how='left'
    )

    # ■ (5) 時間ごとの過去 d 日より前統計量
    hour_stats = _expand_group(ts_shifted, ts_shifted.index.hour, f'{value_col}_hour')
    df_feat = df_feat.merge(
        hour_stats,
        left_on=date_col, right_index=True, how='left'
    )

    return df_feat

# add_time_aggregated_features(df, d=3)

#%%

import pandas as pd

def add_time_aggregated_features(
    df,
    d,
    window_days=30,
    date_col='Date',
    value_col='y'
):
    """
    df に対して以下の特徴量を追加します：
      1) 過去 d 日より前かつ過去 window_days 日間の全履歴統計量 (mean, max, min, std)
      2) 日次集計（mean, max, min, std）からの d 日前 lag
      3) 各曜日・各月・各時間ごとに、過去 d 日より前かつ過去 window_days 日間の統計量 (mean, max, min, std)

    Parameters
    ----------
    df : pandas.DataFrame
    d : int
        何日前より前のデータを「過去」とみなすか（shift の値）
    window_days : int
        「過去何日間分のデータを使うか」を指定
    date_col : str
        日時列名
    value_col : str
        集計対象の値列名
    """
    # ■ 前処理
    df_feat = df.copy()
    df_feat[date_col] = pd.to_datetime(df_feat[date_col])
    df_feat = df_feat.sort_values(date_col).reset_index(drop=True)

    # ■ 原系列（時系列 Series）を取得
    ts = df_feat.set_index(date_col)[value_col]

    # ■ index を d 日だけ未来にシフトして過去統計用の Series を作る
    ts_shifted = ts.copy()
    ts_shifted.index = ts_shifted.index + pd.Timedelta(days=d)
    ts_shifted = ts_shifted.sort_index()

    # ■ (1) 過去 d 日より前かつ過去 window_days 日間の統計量
    window = f'{window_days}D'
    past_stats = (
        ts_shifted
        .rolling(window, closed='both')
        .agg(['mean','max','min','std'])
    )
    past_stats.columns = [
        f'{value_col}_past_{stat}_{d}d_{window_days}d'
        for stat in ['mean','max','min','std']
    ]
    df_feat = df_feat.merge(
        past_stats,
        left_on=date_col, right_index=True, how='left'
    )

    # ■ (2) 日次集計 & d 日前 lag
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
        daily[f'{value_col}_day_{stat}_lag{d}d'] = daily[f'{value_col}_day_{stat}'].shift(d)
    df_feat = df_feat.merge(
        daily[[f'{value_col}_day_{stat}_lag{d}d' for stat in ['mean','max','min','std']]],
        left_on='Date_day', right_index=True, how='left'
    )
    df_feat.drop(columns=['Date_day'], inplace=True)

    # ■ Helper：groupby + rolling(window_days) で “過去 d 日より前かつ window_days 日間” の統計量
    def _expand_group(ts_shifted, grouper, prefix):
        stats = (
            ts_shifted
            .groupby(grouper)
            .rolling(window, closed='both')
            .agg(['mean','max','min','std'])
        )
        stats.index = stats.index.droplevel(0)
        stats.columns = [
            f'{prefix}_{stat}_past{d}d_{window_days}d'
            for stat in ['mean','max','min','std']
        ]
        return stats

    # ■ (3) 曜日ごとの過去統計
    weekday_stats = _expand_group(
        ts_shifted,
        ts_shifted.index.weekday,
        f'{value_col}_weekday'
    )
    df_feat = df_feat.merge(
        weekday_stats,
        left_on=date_col, right_index=True, how='left'
    )

    # ■ (4) 月ごとの過去統計
    month_stats = _expand_group(
        ts_shifted,
        ts_shifted.index.month,
        f'{value_col}_month'
    )
    df_feat = df_feat.merge(
        month_stats,
        left_on=date_col, right_index=True, how='left'
    )

    # ■ (5) 時間（hour）ごとの過去統計
    hour_stats = _expand_group(
        ts_shifted,
        ts_shifted.index.hour,
        f'{value_col}_hour'
    )
    df_feat = df_feat.merge(
        hour_stats,
        left_on=date_col, right_index=True, how='left'
    )

    return df_feat

add_time_aggregated_features(df, d=3)

#%%

def add_multi_scale_lags(
    df,
    d: int,
    max_day: int = 30,
    max_week: int = 4,
    max_month: int = 2,
    max_hour: int = 3,
    date_col: str = 'Date',
    value_col: str = 'y'
):
    """
    df に対して、以下のラグを追加します（ただしオフセット >= d 日のみ）：
      ・日次ラグ：d～max_day 日前
      ・週次ラグ：ceil(d/7)～max_week 週前（7日刻み）
      ・月次ラグ：ceil(d/30)～max_month 月前
      ・時間ラグ：ceil(d*24)～max_hour 時間前

    Parameters
    ----------
    df : pandas.DataFrame
    d : int
        最小ラグ日数
    max_day : int
        日次ラグの最大日数
    max_week : int
        週次ラグの最大週数
    max_month : int
        月次ラグの最大月数
    max_hour : int
        時間ラグの最大時間数
    date_col : str
        タイムスタンプ列名
    value_col : str
        対象変数の列名
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    ts = df.set_index(date_col)[value_col]

    # --- 日次ラグ d～max_day 日前 ---
    for lag in range(d, max_day + 1):
        col = f'{value_col}_lag{lag}d'
        shifted = ts.rename(col)
        shifted.index = shifted.index + pd.Timedelta(days=lag)
        df = df.merge(
            shifted, left_on=date_col, right_index=True, how='left'
        )

    # --- 週次ラグ ceil(d/7)～max_week 週前 (7日刻み) ---
    wk_start = max(1, math.ceil(d / 7))
    for wk in range(wk_start, max_week + 1):
        col = f'{value_col}_lag{wk}w'
        shifted = ts.rename(col)
        shifted.index = shifted.index + pd.Timedelta(days=7 * wk)
        df = df.merge(
            shifted, left_on=date_col, right_index=True, how='left'
        )

    # --- 月次ラグ ceil(d/30)～max_month 月前 ---
    mo_start = max(1, math.ceil(d / 30))
    for mo in range(mo_start, max_month + 1):
        col = f'{value_col}_lag{mo}M'
        shifted = ts.rename(col)
        shifted.index = shifted.index + pd.DateOffset(months=mo)
        df = df.merge(
            shifted, left_on=date_col, right_index=True, how='left'
        )

    # --- 時間ラグ ceil(d*24)～max_hour 時間前 ---
    hr_start = max(1, math.ceil(d * 24))
    for hr in range(hr_start, max_hour + 1):
        col = f'{value_col}_lag{hr}h'
        shifted = ts.rename(col)
        shifted.index = shifted.index + pd.Timedelta(hours=hr)
        df = df.merge(
            shifted, left_on=date_col, right_index=True, how='left'
        )

    return df

#%%

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
    d,
    window_days=30,
    date_col='Date',
    value_col='y'
):
    """
    df に対して以下の特徴量を追加します：
      1) 過去 d 日より前かつ過去 window_days 日間の全履歴統計量 (mean, max, min, std)
      2) 日次集計（mean, max, min, std）からの d 日前 lag
      3) 各曜日・各月・各時間ごとに、過去 d 日より前かつ過去 window_days 日間の統計量 (mean, max, min, std)

    Parameters
    ----------
    df : pandas.DataFrame
    d : int
        何日前より前のデータを「過去」とみなすか（shift の値）
    window_days : int
        「過去何日間分のデータを使うか」を指定
    date_col : str
        日時列名
    value_col : str
        集計対象の値列名
    """
    # ■ 前処理
    df_feat = df.copy()
    df_feat[date_col] = pd.to_datetime(df_feat[date_col])
    df_feat = df_feat.sort_values(date_col).reset_index(drop=True)

    # ■ 原系列（時系列 Series）を取得
    ts = df_feat.set_index(date_col)[value_col]

    # ■ index を d 日だけ未来にシフトして過去統計用の Series を作る
    ts_shifted = ts.copy()
    ts_shifted.index = ts_shifted.index + pd.Timedelta(days=d)
    ts_shifted = ts_shifted.sort_index()

    # ■ (1) 過去 d 日より前かつ過去 window_days 日間の統計量
    window = f'{window_days}D'
    past_stats = (
        ts_shifted
        .rolling(window, closed='both')
        .agg(['mean','max','min','std'])
    )
    past_stats.columns = [
        f'{value_col}_past_{stat}_{d}d_{window_days}d'
        for stat in ['mean','max','min','std']
    ]
    df_feat = df_feat.merge(
        past_stats,
        left_on=date_col, right_index=True, how='left'
    )

    # ■ (2) 日次集計 & d 日前 lag
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
        daily[f'{value_col}_day_{stat}_lag{d}d'] = daily[f'{value_col}_day_{stat}'].shift(d)
    df_feat = df_feat.merge(
        daily[[f'{value_col}_day_{stat}_lag{d}d' for stat in ['mean','max','min','std']]],
        left_on='Date_day', right_index=True, how='left'
    )
    df_feat.drop(columns=['Date_day'], inplace=True)

    # ■ Helper：groupby + rolling(window_days) で “過去 d 日より前かつ window_days 日間” の統計量
    def _expand_group(ts_shifted, grouper, prefix):
        stats = (
            ts_shifted
            .groupby(grouper)
            .rolling(window, closed='both')
            .agg(['mean','max','min','std'])
        )
        stats.index = stats.index.droplevel(0)
        stats.columns = [
            f'{prefix}_{stat}_past{d}d_{window_days}d'
            for stat in ['mean','max','min','std']
        ]
        return stats

    # ■ (3) 曜日ごとの過去統計
    weekday_stats = _expand_group(
        ts_shifted,
        ts_shifted.index.weekday,
        f'{value_col}_weekday'
    )
    df_feat = df_feat.merge(
        weekday_stats,
        left_on=date_col, right_index=True, how='left'
    )

    # ■ (4) 月ごとの過去統計
    month_stats = _expand_group(
        ts_shifted,
        ts_shifted.index.month,
        f'{value_col}_month'
    )
    df_feat = df_feat.merge(
        month_stats,
        left_on=date_col, right_index=True, how='left'
    )

    # ■ (5) 時間（hour）ごとの過去統計
    hour_stats = _expand_group(
        ts_shifted,
        ts_shifted.index.hour,
        f'{value_col}_hour'
    )
    df_feat = df_feat.merge(
        hour_stats,
        left_on=date_col, right_index=True, how='left'
    )

    return df_feat

