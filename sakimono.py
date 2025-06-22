#%%
import pandas as pd
df = pd.read_csv(r'JPP Weekly Trade Tape(Data).csv', encoding='Shift-JIS')

import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns

# 前提: df は既に読み込み・前処理済み (Date, DateTime_CET, Price, Volume (MWh) が適切に型変換済み)
# Kansai にフィルター
df_kansai = df[df['Product'] == 'Kansai Area Base Future'].copy()

df_kansai['Date'] = pd.to_datetime(df_kansai['Date'], format="%Y年%m月%d日")

# 年月カラム
df_kansai['YearMonth'] = df_kansai['Date'].dt.to_period('M').astype(str)
df_kansai['Volume (MWh)'] = (
    df_kansai['Volume (MWh)']
    .astype(str)                     # 念のため文字列化
    .str.replace(',', '', regex=False)  # カンマをすべて除去
    .str.strip()                     # 前後の空白も除去
    .astype(float)
)

df_kansai['DateTime_CET'] = pd.to_datetime(
    df_kansai['Date'].astype(str) + ' ' + df_kansai['Time (in CET)'],
    errors='coerce'
)

#%%
'''
値が大きすぎるせいか表示されない
# 1. Kansai 月次取引量推移
monthly_vol = df_kansai.groupby('YearMonth')['Volume (MWh)'].sum().reset_index()
plt.figure()
plt.plot(monthly_vol['YearMonth'], monthly_vol['Volume (MWh)'])
plt.boxplot(monthly_vol['YearMonth'], monthly_vol['Volume (MWh)'])
plt.xticks(rotation=45)
plt.title('Kansai Area Base Future 月次取引量推移')
plt.xlabel('Year-Month')
plt.ylabel('Total Volume (MWh)')
plt.tight_layout()
plt.show()
'''


# 2. Kansai 月次平均価格推移
monthly_price = df_kansai.groupby('YearMonth')['Price'].sum().reset_index()
plt.figure()
plt.plot(monthly_price['YearMonth'], monthly_price['Price'])
# 3. ２つおきの目盛りとラベルを指定
ticks = range(0, len(monthly_price), 2)                        # インデックスを 0,2,4,...
labels = monthly_price['YearMonth'].iloc[::2].tolist()         # 対応する YearMonth
plt.xticks(ticks, labels, rotation=45, ha='right', fontsize=8)
plt.title('Kansai Area Base Future 月次合計価格推移')
plt.xlabel('Year-Month')
plt.ylabel('Total Price')
plt.tight_layout()
plt.show()

'''
# 3. Contract Period 別総取引量
monthly_price = df_kansai.groupby('YearMonth')['Price'].sum().reset_index()
volume_by_period = df_kansai.groupby('Contract Period')['Volume (MWh)'].sum().sort_index()
plt.figure()
plt.bar(volume_by_period.index, volume_by_period.values)
plt.xticks(rotation=90)
plt.title('Contract Period 別総取引量')
plt.xlabel('Contract Period')
plt.ylabel('Total Volume (MWh)')
plt.tight_layout()
plt.show()
'''

# 4. Contract Type 別価格分布
sns.violinplot(df_kansai, y='Price', x='Contract Type')
sns.violinplot(df_kansai, y='Volume (MWh)', x='Contract Type')

'''
# 5. 時間帯別取引量 (Hour単位)
df_kansai['Hour'] = df_kansai['DateTime_CET'].dt.hour
hourly_vol = df_kansai.groupby('Hour')['Volume (MWh)'].sum().reset_index()
plt.figure()
plt.bar(hourly_vol['Hour'], hourly_vol['Volume (MWh)'])
plt.title('時間帯別取引量 (Hour)')
plt.xlabel('Hour (CET)')
plt.ylabel('Total Volume (MWh)')
plt.tight_layout()
plt.show()
'''

# 6. 価格 vs 取引量 散布図
sns.scatterplot(data=df_kansai, x='Price', y='Volume (MWh)', hue='Contract Type', alpha=0.5)
df_kansai.groupby('Contract Type')[['Price', 'Volume (MWh)']].mean()
# seasonとquarterのVolumeは高い、Priceはほとんど変わらない

#%%

# --- 1: 月次取引量推移 ---
monthly_vol = df_kansai.groupby('YearMonth')['Volume (MWh)'].sum().reset_index()
plt.figure()
plt.plot(monthly_vol['YearMonth'], monthly_vol['Volume (MWh)'])
ticks = range(0, len(monthly_price), 2)                        # インデックスを 0,2,4,...
labels = monthly_price['YearMonth'].iloc[::2].tolist()         # 対応する YearMonth
plt.xticks(ticks, labels, rotation=45, ha='right', fontsize=8)
plt.title('Kansai Area Base Future 月次合計取引量推移')
plt.xlabel('Year-Month')
plt.ylabel('Total Volume (MWh)')
plt.tight_layout()
plt.show()

#%%

# --- 5: 時間帯別取引量 (Hour単位) ---
df_kansai['Hour'] = df_kansai['DateTime_CET'].dt.hour
hourly_vol = df_kansai.groupby('Hour')['Volume (MWh)'].sum().reset_index()
plt.figure()
plt.bar(hourly_vol['Hour'], hourly_vol['Volume (MWh)'])
plt.title('時間帯別取引量 (Hour)')
plt.xlabel('Hour (CET)')
plt.ylabel('Total Volume (MWh)')
plt.tight_layout()
plt.show()

# %%

df_kansai['DayOfWeek'] = df_kansai['DateTime_CET'].dt.day_name()

# 2. 曜日別取引量
dow_vol = df_kansai.groupby('DayOfWeek')['Volume (MWh)'] \
    .sum() \
    .reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']) \
    .reset_index()
plt.figure()
plt.bar(dow_vol['DayOfWeek'], dow_vol['Volume (MWh)'])
plt.title('曜日別 取引量')
plt.xlabel('Day of Week')
plt.ylabel('Total Volume (MWh)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. 時間帯×曜日 取引量ヒートマップ
pivot = df_kansai.pivot_table(
    index='DayOfWeek',
    columns='Hour',
    values='Volume (MWh)',
    aggfunc='sum'
).reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
plt.figure()
plt.imshow(pivot, aspect='auto')
plt.yticks(range(len(pivot.index)), pivot.index)
# plt.xticks(range(8,12), range(8, 12))
plt.title('時間帯×曜日 取引量ヒートマップ')
plt.xlabel('Hour')
plt.ylabel('Day of Week')
plt.colorbar(label='Volume (MWh)')
plt.tight_layout()
plt.show()

# 4. 日次価格の7日移動平均
daily_price = df_kansai.groupby('Date')['Price'] \
    .mean() \
    .sort_index() \
    .reset_index()
daily_price['MA7'] = daily_price['Price'].rolling(window=7).mean()
plt.figure()
plt.plot(daily_price['Date'], daily_price['MA7'])
plt.title('日次価格の7日移動平均')
plt.xlabel('Date')
plt.ylabel('7-day MA of Price')
plt.tight_layout()
plt.show()

# 5. 日次価格の7日ボラティリティ
daily_price['Vol7'] = daily_price['Price'].rolling(window=7).std()
plt.figure()
plt.plot(daily_price['Date'], daily_price['Vol7'], linestyle='-')
plt.title('日次価格の7日ボラティリティ')
plt.xlabel('Date')
plt.ylabel('7-day rolling Std of Price')
plt.tight_layout()
plt.show()

daily_vol = df_kansai.groupby('Date')['Volume (MWh)'].sum().sort_index().reset_index()
# 7日ローリング標準偏差を計算
daily_vol['Vol7_std'] = daily_vol['Volume (MWh)'].rolling(window=7).std()

# プロット
import matplotlib.pyplot as plt

plt.figure()
plt.plot(daily_vol['Date'], daily_vol['Vol7_std'], linestyle='-')
plt.title('日次出来高の7日ボラティリティ')
plt.xlabel('Date')
plt.ylabel('7-day rolling Std of Volume (MWh)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

'''
# 6. 累積取引量推移
daily_vol = df_kansai.groupby('Date')['Volume (MWh)'] \
    .sum() \
    .sort_index() \
    .reset_index()
daily_vol['CumVol'] = daily_vol['Volume (MWh)'].cumsum()
plt.figure()
plt.plot(daily_vol['Date'], daily_vol['CumVol'])
plt.title('累積取引量推移')
plt.xlabel('Date')
plt.ylabel('Cumulative Volume (MWh)')
plt.tight_layout()
plt.show()
'''

# 7. 日次取引回数
daily_cnt = df_kansai.groupby('Date').size().reset_index(name='TradeCount')
plt.figure()
plt.plot(daily_cnt['Date'], daily_cnt['TradeCount'])
plt.title('日次取引回数')
plt.xlabel('Date')
plt.ylabel('Number of Trades')
plt.tight_layout()
plt.show()

# 8. 価格分布ヒストグラム
plt.figure()
plt.hist(df_kansai['Price'].dropna(), bins=30)
plt.title('価格分布ヒストグラム')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

#%%

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# --- 1. 曜日別 Price 分布 (Violin) ---
days = ['Monday','Tuesday','Wednesday','Thursday','Friday']
price_data = [df_kansai[df_kansai['DayOfWeek']==d]['Price'].dropna().values for d in days]
plt.figure()
plt.violinplot(price_data, positions=range(len(days)))
plt.xticks(range(len(days)), days, rotation=45)
# plt.title('Price Distribution by Day of Week')
# plt.ylabel('Price')
plt.tight_layout()
plt.show()

# --- 2. 3D Scatter: Hour vs Price vs Volume ---
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_kansai['Hour'], df_kansai['Price'], df_kansai['Volume (MWh)'])
ax.set_xlabel('Hour')
ax.set_ylabel('Price')
ax.set_zlabel('Volume (MWh)')
# plt.title('3D Scatter: Hour vs Price vs Volume')
plt.tight_layout()
plt.show()

# --- 3. 30日ローリング相関 (Price vs Volume) ---
daily = df_kansai.groupby('Date').agg({'Price':'mean', 'Volume (MWh)':'sum'}).sort_index()
# daily['Corr30'] = daily['Price'].rolling(window=30).corr(daily['Volume (MWh)'])
daily['Corr30'] = daily['Price'].rolling(window=30).corr(daily['Volume (MWh)'])
plt.figure()
plt.plot(daily.index, daily['Corr30'])
# plt.title('30-day Rolling Correlation between Price and Volume')
# plt.xlabel('Date')
# plt.ylabel('Correlation')
plt.tight_layout()
plt.show()

# --- 4. カレンダーヒートマップ: 日次取引量 ---①
daily = df_kansai.groupby('Date').agg({'Price':'mean', 'Volume (MWh)':'mean'}).sort_index()
daily_vol = daily['Volume (MWh)'].reset_index()
daily_vol['Day'] = daily_vol['Date'].dt.day
daily_vol['Month'] = daily_vol['Date'].dt.to_period('M')
months = sorted(daily_vol['Month'].unique(), key=lambda x: x.start_time)
pivot = daily_vol.pivot(index='Day', columns='Month', values='Volume (MWh)').reindex(index=range(1,32))
plt.figure()
plt.imshow(pivot, aspect='auto')
plt.yticks(range(0,31), range(1,32))
plt.xticks(range(len(months)), [str(m) for m in months], rotation=90)
# plt.title('Calendar Heatmap of Daily Volume')
# plt.xlabel('Month')
# plt.ylabel('Day of Month')
plt.colorbar(label='Volume (MWh)')
plt.tight_layout()
plt.show()

# --- 5. 価格帯別取引量 (価格ビンごと) ---
bins = pd.cut(df_kansai['Price'].dropna(), bins=10)
vol_by_price = df_kansai.groupby(bins)['Volume (MWh)'].sum().reset_index()
labels = vol_by_price['Price'].astype(str)
plt.figure()
plt.bar(labels, vol_by_price['Volume (MWh)'])
plt.xticks(rotation=90)
# plt.title('Volume by Price Bins')
# plt.xlabel('Price Bin')
# plt.ylabel('Total Volume (MWh)')
plt.tight_layout()
plt.show()


# %%
