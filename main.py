#%%
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore') #警告を非表示にする


#%%
url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv" #データセットのあるURL
df=pd.read_csv(url,                      #読み込むデータのURL
               index_col='Month',        #変数「Month」をインデックスに設定
               parse_dates=True)         #インデックスを日付型に設定
df.head() #確認

#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split

#%%

# データ読み込み
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(url, parse_dates=['Month'])
df.columns = ['Month', 'Passengers']
df.set_index('Month', inplace=True)

# ラグ特徴量作成(例：過去12ヶ月分)
for i in range(1, 13):
    df[f'lag_{i}'] = df['Passengers'].shift(i)

df.dropna(inplace=True)
# 説明変数と目的変数
X = df.drop(columns=['Passengers'])
y = df['Passengers']

# 時系列的にトレイン/テスト分割(最後12ヶ月をテストとする)
X_train, X_test = X[:-12], X[-12:]
y_train, y_test = y[:-12], y[-12:]

shap.initjs()
# モデル学習(XGBoost回帰)
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# SHAP計算
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
'''
shap_values.values       # SHAP値本体(NumPy array)
shap_values.data         # モデルに渡した入力データ(X_testに対応)
shap_values.base_values  # モデルの基準値(平均予測など)
shap_values.feature_names # 特徴量の名前
'''

# Summary plot
shap.summary_plot(shap_values, X_test, plot_type="dot")

# Feature importance plot
shap.summary_plot(shap_values, X_test, plot_type="bar")

# 特徴量の値が大きくなると予測にどう影響するかがわかる
shap.plots.scatter(shap_values[:, ["lag_1", "lag_2"]], color=shap_values)
# 個別の予測で、どの特徴がどれだけ寄与したかが一目瞭然！
shap.plots.waterfall(shap_values[-1])
            

shap.initjs()
force_plot = shap.force_plot(shap_values[0], matplotlib=False, show=False)
with open("force_plot.html", "w") as f:
    f.write(force_plot.html())

# shap.plots.scatter(shap_values[:, X_test.columns.get_loc("lag_1")], color=shap_values)
shap.plots.scatter(shap_values[:, "lag_1"], color=shap_values[:, "month"])
shap.initjs()



shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(X_test)
shap.summary_plot(shap_interaction_values, X_test)

mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
shap_df = pd.DataFrame({
    'Feature': X_test.columns,
    'MeanAbsSHAP': mean_abs_shap
}).sort_values("MeanAbsSHAP", ascending=False)
print(shap_df)

shap_values = explainer(X_test)  # 推奨(v0.40以降)
# もしくは
shap_values = shap.TreeExplainer(model).shap_values(X_test)

# %%

# 必要なライブラリ
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# データ読み込み
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(url)

# データの前処理
df.columns = ["ds", "y"]
df["ds"] = pd.to_datetime(df["ds"])

# MinMaxScalerでデータをスケーリング
scaler = MinMaxScaler(feature_range=(-1, 1))
df['y'] = scaler.fit_transform(df[['y']])

# 時系列データをLSTM用に変換
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        labels.append(data[i+seq_length])
    return np.array(sequences), np.array(labels)

seq_length = 12  # 過去12ヶ月分を使って予測
data = df['y'].values
sequences, labels = create_sequences(data, seq_length)

# データを訓練用とテスト用に分ける
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, shuffle=False)

# データの形をPyTorchのテンソルに変換
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # 最後に1次元を追加
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)  # 最後に1次元を追加
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# LSTMモデルの定義
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])  # 最後のタイムステップを使って予測
        return predictions

# モデルのインスタンス化
input_size = 1  # 特徴量数（yのみを使用）
hidden_layer_size = 64
output_size = 1
model = LSTM(input_size, hidden_layer_size, output_size)

# 損失関数と最適化手法の定義
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# モデルの学習
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_function(y_pred.squeeze(), y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

# 予測を実行
model.eval()
with torch.no_grad():
    y_pred = model(X_test)

# 予測結果を元のスケールに戻す
y_pred = scaler.inverse_transform(y_pred.detach().numpy())
y_test = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))

# 実測値と予測値を重ね書き
plt.figure(figsize=(10, 6))
plt.plot(df['ds'][-len(y_test):], y_test, label='Actual', color='blue')
plt.plot(df['ds'][-len(y_test):], y_pred, label='Forecast', color='red', linestyle='--')
plt.title("Actual vs Forecasted (LSTM)")
plt.xlabel("Date")
plt.ylabel("Number of Passengers")
plt.legend()
plt.show()
