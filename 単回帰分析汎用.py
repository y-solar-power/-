import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# === 日本語フォント設定 ===
matplotlib.rcParams['font.family'] = 'Meiryo'
matplotlib.rcParams['axes.unicode_minus'] = False

def solar_power_simple_regression():
    # === CSV読み込み ===
    # 列順：温度, 発電量, 雲量（※雲量は今回は使わない）
    df = pd.read_csv(
        "測定データExcel.csv", #データの入ったExcelを入力
        usecols=[1, 6],
        names=["power", "panel_temp"],
        header=0
    )

    print(f"🔍 データ件数: {len(df)}")
    print("NaN 含有数:\n", df.isna().sum())

    # === NaN除去（単回帰では削除が自然） ===
    df = df.dropna()

    # === 特徴量・目的変数 ===
    X = df[["panel_temp"]]   # 単回帰
    y = df["power"]

    # === 標準化 ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # === 学習・テスト分割 ===
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # === モデル学習 ===
    model = LinearRegression()
    model.fit(X_train, y_train)

    # === 予測・評価 ===
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print("\n＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝")
    print("　単回帰分析結果（パネル温度 → 発電量 [W]）")
    print("温度 平均:", df["panel_temp"].mean())
    print("温度 標準偏差:", df["panel_temp"].std())
    print("＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝")
    print(f"決定係数 R²: {r2:.4f}")
    print(f"MSE: {mse:.4f} [W²]")
    print(f"RMSE: {rmse:.4f} [W]")
    print("＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝")

    # === 回帰式 ===
    coef = model.coef_[0]
    intercept = model.intercept_
    print("\n📘 回帰式（標準化温度）:")
    print(f"発電量 [W] = {coef:.4f} × 温度 + {intercept:.4f}")

    # === 可視化（散布図＋回帰直線） ===
    temp_range = np.linspace(df["panel_temp"].min(),
                             df["panel_temp"].max(), 100).reshape(-1, 1)
    temp_range_scaled = scaler.transform(temp_range)
    power_pred_line = model.predict(temp_range_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(df["panel_temp"], df["power"],
                alpha=0.6, label="Actual measurement data")
    plt.plot(temp_range, power_pred_line,
             color="red",linewidth=2, label="regression line")
    plt.xlabel("ECO [%]", fontsize=16) #x軸の名前
    plt.ylabel("Power [W]", fontsize=16) #y軸の名前
    #plt.title("湿度と電力", fontsize=18) #グラフタイトル必要であれば
    plt.legend(fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    solar_power_simple_regression()
