#%% 全停止情報と全ユニット情報の読込
import pandas as pd
from IPython.display import display

file_path = r'./全停止情報.csv'
df = pd.read_csv(file_path, encoding='cp932')

# add code
file_path = r'./全ユニット情報.csv'
df_cap = pd.read_csv(file_path, encoding='cp932')

df['停止日時'] = pd.to_datetime(df['停止日時'], format='%Y/%m/%d %H:%M', errors='coerce')
df['復旧予定日'] = pd.to_datetime(df['復旧予定日'], format='%Y/%m/%d', errors='coerce')

df['認可出力_numeric'] = (
    df['認可出力']
    .astype(str)
    .str.replace(',', '', regex=False)
    .str.strip()
    .pipe(pd.to_numeric, errors='coerce')
)

df['unit_id'] = df['発電所名'] + '_' + df['発電形式'] + '_' + df['ユニット名'].astype(str)

df['outage_value'] = df.apply(
    lambda row: row['低下量']
        if (row['停止区分'] == '出力低下' and pd.notna(row['低下量']))
        else row['認可出力_numeric'],
    axis=1
)

# 日付列を datetime 型に変換
df_cap['稼働開始日'] = pd.to_datetime(df_cap['稼働開始日'], errors='coerce').dt.floor('D')
# 稼働終了日の9999/12/31はNaTに変換される。
# NaTに変換されなかった場合は廃止が確定したユニットとなる。
df_cap['稼働終了日'] = pd.to_datetime(df_cap['稼働終了日'], errors='coerce').dt.floor('D')

df_cap['unit_id'] = df_cap['発電所名'] + '_' + df_cap['発電形式'] + '_' + df_cap['ユニット名'].astype(str)


#%% ver5

halfhour_index = pd.date_range(
    start='2023-04-01 00:00:00',
    end='2025-03-31 23:30:00',
    freq='30min'
)

records = []

for unit, group in df.groupby('unit_id'):
    slot_max = pd.Series(0.0, index=halfhour_index)

    # ② 各イベント区間を順に処理
    group = group.sort_values('停止日時').reset_index(drop=True)
    for idx, row in group.iterrows():
        start = row['停止日時'].floor('30min')
        # 復旧予定日のあるなしで end を決定（元ロジックと同じ）
        if pd.notna(row['復旧予定日']):
            # 停止期間の末端
                # ①復旧予定日の「23:30」
                # ②復旧予定日の前日の「23:30」
            end = pd.to_datetime(row['復旧予定日'].strftime('%Y-%m-%d') + ' 23:30:00') # ①
            # end = pd.to_datetime((row['復旧予定日'] - pd.Timedelta(days=1)).strftime('%Y-%m-%d') + ' 23:30:00') # ②
        else:
            if idx < len(group) - 1:
                next_start = group.loc[idx+1, '停止日時'].floor('30min')
                end = next_start - pd.Timedelta(minutes=30)
            else:
                end = halfhour_index[-1]

        if start > end:
            continue

        idx_mask = (slot_max.index >= start) & (slot_max.index <= end)
        slot_max.iloc[idx_mask] = np.maximum(slot_max.iloc[idx_mask], row['outage_value'])
        # print(f"Unit: {unit}, slot_max: {slot_max}")

    # ④ 最後に日単位で合計 → /48 して日次低下量を算出
    #    slot_max は「各コマの低下量」が入っている Series
    daily_sum = slot_max.groupby(slot_max.index.floor('D')).sum()
    daily_outage = daily_sum / 48.0

    # ⑤ レコード化
    for day, loss in daily_outage.items():
        if loss > 0:
            records.append({
                'Date': day,
                'unit_id': unit,
                'daily_outage': loss
            })

df_events_hourly = pd.DataFrame(records)

daily_index = pd.date_range(start='2023-04-01', end='2025-03-31', freq='D')

df_wide = (
    df_events_hourly
    .pivot_table(
        index='Date',
        columns='unit_id',
        values='daily_outage',
        aggfunc='sum'
    )
    .reindex(daily_index, fill_value=0)
)

#%%

# 廃止ユニット以降を出力0にする処理
# ユニットは稼働終了日の23:00:00まで存在していると考える

# 廃止となったユニットのみを集める
# 廃止でないユニットの稼働終了日はすべて9999/12/31となっており、NaTに変換されている

df_retire = df_cap[(pd.notna(df_cap['稼働終了日']))].copy()
df_retire['retire_date'] = df_retire['稼働終了日'].dt.floor('D')

retire_map = (
    df_retire
    .sort_values('retire_date')
    .drop_duplicates('unit_id')[['unit_id', 'retire_date']]
)
retire_dict = dict(zip(retire_map['unit_id'], retire_map['retire_date']))

# df_wide 上で稼働終了日より後は 0 に置き換え
for unit_id, retire_date in retire_dict.items():
    if unit_id not in df_wide.columns:
        continue
    print('*' * 20)
    df_wide.loc[df_wide.index > retire_date, unit_id] = 0

# 発電形式ごとに廃止台数を集計する列を追加
df_retire['発電形式'] = df_retire['unit_id'].str.split('_').str[1]

# 本来の retire_date で「当日廃止台数」を集計 → tmp
tmp = (
    df_retire
    .groupby(['retire_date', '発電形式'])
    .size()
    .unstack(fill_value=0)
)

full_cum = tmp.cumsum()

# 4) 2023-04-01 ～ 2025-03-31 の daily_index に合わせて『前方補完』する
retire_cumulative = full_cum.reindex(daily_index, method='ffill', fill_value=0)




unit_columns = df_wide.columns.tolist()

gen_types = [col.split('_')[1] for col in unit_columns]

unique_gen_types = sorted(set(gen_types))

# 発電形式ごとに合計列を作成
for gen in unique_gen_types:
    cols_of_type = [col for col in unit_columns if col.split('_')[1] == gen]
    df_wide[f"{gen}_停止状況"] = df_wide[cols_of_type].sum(axis=1)

for gen_type in retire_cumulative.columns:
    col_name = f"{gen_type}_廃止累計"
    df_wide[col_name] = retire_cumulative[gen_type]

df_wide = df_wide.reset_index().rename(columns={'index': 'Date'}).fillna(0)

df_wide = df_wide[['Date', 'その他_停止状況', '原子力_停止状況', '水力_停止状況', '火力（ガス）_停止状況', '火力（石油）_停止状況', '火力（石炭）_停止状況', '原子力_廃止累計', '火力（ガス）_廃止累計', '火力（石油）_廃止累計']]

#%% 全ユニット情報の処理

# 停止状況は最終的に日粒度となるため、全ユニット情報の粒度は日のままでよい

# 処理対象の期間を決める
period_start = pd.Timestamp('2023-04-01')
period_end   = pd.Timestamp('2025-03-31')
all_dates = pd.date_range(start=period_start, end=period_end, freq='D')

# 発電所名×発電形式×ユニット名をユニークに取得
df_cap['キー'] = df_cap.apply(
    lambda r: f"{r['発電所名']}_{r['発電形式']}_{r['ユニット名']}", axis=1
)
unique_keys = df_cap['キー'].unique().tolist()

# 出力用を作成（インデックス＝日付、カラム＝ユニットキー、値を 0 で初期化）
df_out = pd.DataFrame(
    data=0,
    index=all_dates,
    columns=unique_keys,
)

# 稼働開始日を0:00:00、稼働終了日を23:30:00として考える
# 各ユニットごとに、稼働期間内の日付範囲を埋めていく
for _, row in df_cap.iterrows():
    key = row['キー']
    cap = row['認可出力']

    start = max(row['稼働開始日'], period_start)
    end = row['稼働終了日']

    if pd.isna(end):
        end = period_end
        
    # 開始日～終了日 が対象期間に一切重ならない場合はスキップ
    if end < period_start or start > period_end:
        continue

    # 範囲内のインデックス部分に対して「認可出力」を代入
    df_out.loc[start:end, key] = cap

mapping = df_cap.set_index('キー')['発電形式'].to_dict()
all_types = df_cap['発電形式'].unique().tolist()

# 各発電形式ごとに該当キーを抽出し、日次合計を新しい列に格納
for gen_type in all_types:

    keys_of_type = [ key for key, t in mapping.items() if t == gen_type ]
    keys_of_type = [key for key in keys_of_type if key in df_out.columns]

    col_name = f"{gen_type}_認可出力"
    df_out[col_name] = df_out[keys_of_type].sum(axis=1)

df_out = df_out.reset_index(drop=False).rename(columns={'index': 'Date'})
df_out = df_out[['Date', '火力（石炭）_認可出力', 'その他_認可出力',
                 '火力（ガス）_認可出力', '原子力_認可出力', '火力（石油）_認可出力', '水力_認可出力']]

display(df_out)

#%%　全停止情報と全ユニット情報のマージ

df = df_out.merge(df_wide, how='inner', on='Date')

# 発電形式のリスト
gen_types = [
    "火力（石炭）",
    "火力（ガス）",
    "火力（石油）",
    "水力",
    "原子力",
    "その他"
]

for t in gen_types:
    cap_col   = f"{t}_認可出力"
    stop_col  = f"{t}_停止状況"

    # 「稼働出力 = 認可出力 − 停止状況」を計算
    oper_col  = f"{t}_稼働状況"
    df[oper_col] = df[cap_col] - df[stop_col]

    # 「稼働率 = 稼働出力 ÷ 認可出力」を計算
    util_col = f"{t}_稼働率"
    # ゼロ除算を防ぐために、認可出力がゼロの行は NaN にするなど工夫できます
    df[util_col] = df[oper_col] / df[cap_col]

#%%　修正項目

{col: df.index[df[col] < 0].tolist() for col 
 in df.select_dtypes(include='number').columns if (df[col] < 0).any()}

#%%　plot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import japanize_matplotlib

# --- 前提:
# df は日粒度で、少なくとも以下の列を持っているものとします。
#   "{発電形式}_稼働出力" と "{発電形式}_停止状況"
#   例: "原子力_稼働出力",   "原子力_停止状況"
#       "水力_稼働出力",     "水力_停止状況"
#       "火力（石炭）_稼働出力", "火力（石炭）_停止状況"
#       …など
#
# また "Date" 列はすでに datetime 型（日粒度）である想定です。
#   もし異なる場合は、以下のように変換してください:
# df['Date'] = pd.to_datetime(df['Date']).dt.normalize()

# コピーして作業用 DataFrame を用意
df_daily = df.copy()
df_daily['Date'] = pd.to_datetime(df_daily['Date']).dt.normalize()

# ── 0) 発電形式リストを定義 ──
gen_types = [
    "原子力",
    "水力",
    "火力（石炭）",
    "火力（ガス）",
    "火力（石油）",
    "その他"
]

# ── 1) 「稼働出力」列を「稼働状況」へリネーム ──
#     各形式について、既存の "<形式>_稼働出力" を "<形式>_稼働状況" に変更します。
rename_map = {}
for t in gen_types:
    old_col = f"{t}_稼働出力"
    new_col = f"{t}_稼働状況"
    if old_col in df_daily.columns:
        rename_map[old_col] = new_col

df_daily = df_daily.rename(columns=rename_map)

# ── 2) 2025年3月分だけ抽出 ──
df_mar2025 = df_daily[df_daily['Date'].dt.to_period('M') == "2025-03"].copy()

# （万一、該当月に日付がない行がある場合に備え、ソートしておきます）
df_mar2025 = df_mar2025.sort_values(by='Date').reset_index(drop=True)

# X 軸のための範囲（03/01 〜 04/01）
period = pd.Period("2025-03", freq="M")
start = period.to_timestamp()            # 2025-03-01 00:00:00
end   = (period + 1).to_timestamp()   # 2025-04-01 00:00:00

# ── 3) 「稼働状況」グラフを描画 ──
#    カラム名はすでに "<形式>_稼働状況" にリネーム済み
oper_categories = [f"{t}_稼働状況" for t in gen_types if f"{t}_稼働状況" in df_mar2025.columns]
oper_colors     = ['#4C4C4C', '#1F77B4', '#FFC0CB', '#FF7F0E', '#D62728', '#F1C40F']

fig, ax = plt.subplots(figsize=(12, 4))
cum = np.zeros(len(df_mar2025))

for cat, col in zip(oper_categories, oper_colors):
    y = df_mar2025[cat].values
    cum2 = cum + y
    ax.fill_between(
        df_mar2025['Date'], cum, cum2,
        step='post', color=col, label=cat.replace("_稼働状況", "（稼働）")
    )
    cum = cum2

ax.set_xlim(start, end)
ax.set_title("2025年3月 の稼働状況", fontsize=14)
ax.set_xlabel("日")
ax.set_ylabel("稼働出力（MW）")

ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))

ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False)
plt.tight_layout()
plt.show()


# ── 4) 「停止状況」グラフを描画 ──
stop_categories = [f"{t}_停止状況" for t in gen_types if f"{t}_停止状況" in df_mar2025.columns]

fig, ax2 = plt.subplots(figsize=(12, 4))
cum_stop = np.zeros(len(df_mar2025))

for cat, col in zip(stop_categories, oper_colors):
    y = df_mar2025[cat].fillna(0).values  # NaN があれば 0 に置き換え
    cum2 = cum_stop + y
    ax2.fill_between(
        df_mar2025['Date'], cum_stop, cum2,
        step='post', color=col, label=cat.replace("_停止状況", "（停止）")
    )
    cum_stop = cum2

ax2.set_xlim(start, end)
ax2.set_title("2025年3月 の停止状況", fontsize=14)
ax2.set_xlabel("日")
ax2.set_ylabel("停止出力（MW）")

ax2.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))

ax2.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False)
plt.tight_layout()
plt.show()

#%%

# 1) Date を datetime に変換してインデックスに設定
# df['Date'] = pd.to_datetime(df['Date'])
# df = df.set_index('Date').sort_index()

# 2) プロットを分ける列のリスト
thermal_cols = [
    '火力（石炭）_稼働率',
    '火力（ガス）_稼働率',
    '火力（石油）_稼働率'
]
other_cols = [
    '水力_稼働率',
    '原子力_稼働率',
    'その他_稼働率'
]

# 3) 2つのサブプロットを作成（縦に並べる）
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# (A) 火力系のプロット
ax1 = axs[0]
for col in thermal_cols:
    if col in df.columns:
        ax1.plot(
            df.index,
            df[col],
            label=col.replace('_稼働率', ''),
            linewidth=1
        )
ax1.set_title('火力系 稼働率推移', fontsize=14)
ax1.set_ylabel('稼働率')
ax1.set_ylim(0, 1.05)
ax1.legend(loc='upper right')

# (B) その他の発電形式のプロット
ax2 = axs[1]
for col in other_cols:
    if col in df.columns:
        ax2.plot(
            df.index,
            df[col],
            label=col.replace('_稼働率', ''),
            linewidth=1
        )
ax2.set_title('水力・原子力・その他 稼働率推移', fontsize=14)
ax2.set_xlabel('日付')
ax2.set_ylabel('稼働率')
ax2.set_ylim(0, 1.05)
ax2.legend(loc='upper right')

# 4) x軸を月ごとに区切り、'YYYY-%m' 形式で表示
axs[-1].xaxis.set_major_locator(mdates.MonthLocator())
axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# 5) x軸ラベルを 45 度傾ける
plt.setp(axs[-1].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()


#%% dailyパターン


# dailyパターン

'''
7. 各イベント（停止 ～ 復旧）を「日ごと」に展開する

   ● イベントの定義：
     停止開始   = row['停止日時'] の「当日午前0時」から
     復旧予定日 = row['復旧予定日'] の「前日まで」を停止中とみなす

   具体的には：
     1) 停止開始日は  row['停止日時'].floor('D') で「当日午前0時」にする
     2) 復旧予定日 があるなら (復旧予定日 - 1日).floor('D') で「前日までの午前0時」にする
     3) 復旧予定日 が NaT なら、任意で (2025-03-31) まで継続とみなす

   → 期間 start_date ～ end_date の間に含まれる日付すべてを、レコード化する
'''

# 粒度を日バージョン
daily_index = pd.date_range(start='2023-04-01', end='2025-03-31', freq='D')

records = []

# 「各ユニットごとに、停止イベントを発生順にソートして処理する」
for unit, group in df.sort_values('停止日時').groupby('unit_id'):
    # 1) 停止日時でソートし直し、インデックスを振り直す
    group = group.sort_values('停止日時').reset_index(drop=True)

    # 2) 同じユニットの各イベント（行）についてループ
    for idx, row in group.iterrows():
        # 2-1) 「停止開始日」を「当日の午前0時」に丸める
        # TODO: 30分粒度として扱い、床関数を適用する
        start_date = row['停止日時'].floor('D')

        # 2-2) “復旧予定日” がある場合はそれをそのまま end_date とし、
        #      ない場合は次の停止イベントの前日まで延長する
        if pd.notna(row['復旧予定日']):
             # ①復旧予定日はある場合、復旧予定日-1日を終了日にする
             # TODO: ②復旧予定日はある場合、復旧予定日の23:30:00を終了日にする
            end_date = (row['復旧予定日'] - pd.Timedelta(days=1)).floor('D')
        else:
            # 復旧予定日 が NaT の場合 → 「次の停止日の前日」まで延長
            #   もし次のイベントがなければ「2025-03-31」まで
            if idx < len(group) - 1:
                next_start = group.loc[idx + 1, '停止日時'].floor('D')
                end_date = (next_start - pd.Timedelta(days=1)).floor('D')
            else:
                end_date = pd.to_datetime('2025-03-31')

        # 2-3) start_date～end_date の間すべての日付を列挙し、daily_index の範囲内だけ追加
        if start_date <= end_date:
            # TODO: 30粒度に変換する、start_dateとend_dateを30分粒度で取り扱う
            for single_date in pd.date_range(start=start_date, end=end_date, freq='D'):
                if single_date in daily_index:
                    records.append({
                        'Date': single_date,
                        'unit_id': unit,
                        'outage_value': row['outage_value']
                    })

df_events_daily = pd.DataFrame(records)
# display(df_events_daily)
# df_events.to_csv('daily.csv', index=False, encoding='cp932')

#%% ver1

halfhour_index = pd.date_range(
    start='2023-04-01 00:00:00',
    end='2025-03-31 23:30:00',
    freq='30min'
)

records = []

# 各ユニットごとに、停止イベントを発生順にソートして処理する
for unit, group in df.sort_values('停止日時').groupby('unit_id'):
    # 同一ユニット内のイベントを停止日時順に並び替え
    group = group.sort_values('停止日時').reset_index(drop=True)

    for idx, row in group.iterrows():

        # 1) 停止開始時刻を直近の30分単位に丸める

        start_ts = row['停止日時'].floor('30min')
        # print(f"before_start_ts: {row['停止日時']}, after_start_ts: {start_ts}")

        # 2) 復旧予定日があれば、その日は「23:30」を終了時刻とみなす
        #    なければ次イベントの前の30分まで次のイベントも無ければ対象期間の最終日時までとする

        if pd.notna(row['復旧予定日']):
            # 停止期間の末端
                # ①復旧予定日の「23:30」
                # ②復旧予定日の前日の「23:30」
            end_ts = pd.to_datetime(row['復旧予定日'].strftime('%Y-%m-%d') + ' 23:30:00') # ①
            # end_ts = pd.to_datetime((row['復旧予定日'] - pd.Timedelta(days=1)).strftime('%Y-%m-%d') + ' 23:30:00') # ②
        else:
            # 次のイベントがある場合は、その停止開始時刻の直前30分を終了時刻とする
            if idx < len(group) - 1:
                next_start = group.loc[idx + 1, '停止日時'].floor('30min')
                end_ts = next_start - pd.Timedelta(minutes=30)
            else:
                # 最後のイベントなら、データ範囲の最終時刻まで
                end_ts = pd.to_datetime('2025-03-31 23:30:00')

        # 3) start_ts 〜 end_ts の半時コマ列を取得しリスト化
        if start_ts <= end_ts:
            span = halfhour_index[(halfhour_index >= start_ts) & (halfhour_index <= end_ts)]
            if len(span) > 0:
                # 4) 「停止していた半時コマ数」を日ごとに集計
                # span の各タイムスタンプを「日」部分に変換
                days = span.floor('D')
                # print(f"span: {span}, days: {days}")

                down_counts = days.value_counts().to_dict()

                # 5) 日ごとに「日次低下量」を計算しレコード化
                #    （停止コマ数 / 全コマ数(48) × outage_value）
                for day, down_intervals in down_counts.items():
                    # 1 日あたりの全コマ数は 48
                    daily_loss = (down_intervals / 48) * row['outage_value']
                    records.append({
                        'Date': day,
                        'unit_id': unit,
                        'daily_outage': daily_loss
                    })

#%% ver2

# 1) 全体の30分刻みインデックス
halfhour_index = pd.date_range(
    start='2023-04-01 00:00:00',
    end='2025-03-31 23:30:00',
    freq='30min'
)

records = []

# 2) ユニットごとに処理
for unit, group in df.sort_values('停止日時').groupby('unit_id'):
    # 2-1) イベントごとの (start, end, value) を集める
    intervals = []

    # 同一ユニット内のイベントを停止日時順に並び替え
    group = group.sort_values('停止日時').reset_index(drop=True)
    for idx, row in group.iterrows():
        # 開始を30分刻みに丸め
        start = row['停止日時'].floor('30min')

        # 復旧予定日がある場合
        if pd.notna(row['復旧予定日']):
            # 停止期間の末端
                # ①復旧予定日の「23:30」
                # ②復旧予定日の前日の「23:30」
            end = pd.to_datetime(row['復旧予定日'].strftime('%Y-%m-%d') + ' 23:30:00') # ①
            # end = pd.to_datetime((row['復旧予定日'] - pd.Timedelta(days=1)).strftime('%Y-%m-%d') + ' 23:30:00') # ②
        else:
            # 次イベント or 最終日時…従来ロジックと同じ
            # idx = row.name
            if idx < len(group) - 1:
                next_start = group.loc[idx+1, '停止日時'].floor('30min')
                end = next_start - pd.Timedelta(minutes=30)
            else:
                end = pd.to_datetime('2025-03-31 23:30:00')

        if start <= end:
            intervals.append((start, end, row['outage_value']))

    # 2-2) 開始時刻でソートしてマージ
    intervals.sort(key=lambda x: x[0])
    merged = []
    for s, e, v in intervals:
        if not merged:
            merged.append([s, e, v])
        else:
            ps, pe, pv = merged[-1]
            # 重複 or 連続（30分以内）ならマージ
            if s <= pe + pd.Timedelta(minutes=30):
                merged[-1][1] = max(pe, e)
                # （同じpv=outage_value前提ならpvの更新不要）
            else:
                merged.append([s, e, v])

    # 3) マージ後の区間を日次で分解→集計
    for s, e, v in merged:
        span = halfhour_index[(halfhour_index >= s) & (halfhour_index <= e)]
        if len(span) == 0:
            continue
        days = span.floor('D')
        down_counts = days.value_counts().to_dict()
        for day, cnt in down_counts.items():
            daily_loss = (cnt / 48) * v
            records.append({
                'Date': day,
                'unit_id': unit,
                'daily_outage': daily_loss
            })

# 最後に DataFrame化
df_events_hourly = pd.DataFrame(records)


#%% ver4
# 例）解析対象の30分刻みインデックス
halfhour_index = pd.date_range(
    start='2023-04-01 00:00:00',
    end='2025-03-31 23:30:00',
    freq='30min'
)

records = []

for unit, group in df.groupby('unit_id'):
    # 1) ユニットあたりの容量（outage_value）はユニークである前提で max() を取る
    capacity = group['outage_value'].max()
    
    # 2) 半時刻ごとのダウンフラグ用 Series を用意
    mask = pd.Series(False, index=halfhour_index)
    
    # 3) 各イベントを半時刻区間に落とし込み
    group = group.sort_values('停止日時').reset_index(drop=True)
    for idx, row in group.iterrows():
        start = row['停止日時'].floor('30min')
        
        if pd.notna(row['復旧予定日']):
            # 復旧日の23:30までダウン
            end = pd.to_datetime(f"{row['復旧予定日'].date()} 23:30:00")
        else:
            # 次イベントがあればその直前30分、なければ全体最終時刻
            if idx < len(group) - 1:
                next_start = group.loc[idx+1, '停止日時'].floor('30min')
                end = next_start - pd.Timedelta(minutes=30)
            else:
                end = halfhour_index[-1]

        if start <= end:
            # ブールマスクを True にセット（重複箇所は上書きでOK）
            mask.loc[start:end] = True

    # 4) 最終マスクから True のタイムスタンプだけ抽出し、日単位でカウント
    down_times = mask[mask].index
    days = down_times.floor('D')
    daily_counts = days.value_counts().sort_index()

    # 5) 日次低下量を計算して記録
    for day, count in daily_counts.items():
        records.append({
            'Date': day,
            'unit_id': unit,
            'daily_outage': (count / 48) * capacity
        })

# 6) 結果の DataFrame 化
df_events_hourly = pd.DataFrame(records)

