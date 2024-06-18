import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
import google.generativeai as genai

# APIキー入力部分
api_key = st.text_input("APIキーを入力してください:", value="", type="password")
genai.configure(api_key=api_key)

# Generative AIモデルの設定（仮のモデル名）
model = genai.GenerativeModel('gemini-pro')

# セッションステートを初期化
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Streamlitアプリの設定
st.title("HealthTracker(仮)")

# CSVファイルのアップロード
uploaded_file = st.file_uploader("CSVファイルをアップロード", type=["csv"])

if uploaded_file is not None:
    # CSVファイルの読み込み
    data = pd.read_csv(uploaded_file)

    # データの前処理
    data['睡眠時間'] = data['睡眠時間'].replace('4時間以下', '4').str.replace('時間', '').astype(float)
    data['訓練開始時間'] = pd.to_datetime(data['訓練開始時間']).dt.hour + pd.to_datetime(data['訓練開始時間']).dt.minute / 60
    data['訓練終了時間'] = pd.to_datetime(data['訓練終了時間']).dt.hour + pd.to_datetime(data['訓練終了時間']).dt.minute / 60
    data['起床時間'] = pd.to_datetime(data['起床時間']).dt.hour + pd.to_datetime(data['起床時間']).dt.minute / 60
    data['ウキウキ'] = data['ウキウキ'].astype(int)
    data['ストレス'] = data['ストレス'].astype(int)
    data['不安'] = data['不安'].astype(int)
    data['イライラ'] = data['イライラ'].fillna(data['イライラ'].mode()[0]).astype(int)
    data['疲れ'] = data['疲れ'].astype(int)

    # 必要な変数の選択
    selected_columns = ['訓練開始時間', '訓練終了時間', '起床時間', '睡眠時間', 'ウキウキ', 'ストレス', '不安', 'イライラ', '疲れ']
    analysis_data = data[selected_columns]

    # 不要な変数の削除
    analysis_data = analysis_data.drop(columns=['訓練開始時間', '訓練終了時間', '起床時間'])

    # 相関分析の実行
    correlation_matrix_reduced = analysis_data.corr()

    # 回帰分析の実行
    X_reduced = analysis_data.drop(columns=['ウキウキ'])
    y_reduced = analysis_data['ウキウキ']
    model_reduced = sm.OLS(y_reduced, X_reduced).fit()
    regression_results_reduced = model_reduced.summary()

    # 相関行列のヒートマップの表示
    st.subheader("相関行列のヒートマップ")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix_reduced, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    st.pyplot(fig)

    # 回帰係数のプロットの表示
    st.subheader("回帰係数のプロット")
    coefficients_reduced = model_reduced.params
    errors_reduced = model_reduced.bse
    fig, ax = plt.subplots(figsize=(10, 6))
    coefficients_reduced.plot(kind='bar', yerr=errors_reduced, ax=ax)
    ax.set_ylabel('係数値')
    ax.set_title('回帰係数と誤差範囲')
    st.pyplot(fig)

    # 回帰分析の結果の表示
    st.subheader("回帰分析の結果")
    st.text(regression_results_reduced)

    # 会話履歴を表示
    st.subheader("会話の履歴")
    for entry in st.session_state.conversation_history:
        if entry["type"] == "user":
            st.write(f"質問: {entry['content']}")
        elif entry["type"] == "response":
            st.write(f"回答: {entry['content']}")

    # ユーザーからの入力を取得
    prompt = st.text_input("質問を入力してください:", key="input_text")

    if st.button("送信"):
        if prompt:
            st.session_state.conversation_history.append({"type": "user", "content": prompt})
            combined_text = f"{prompt}\n以下の情報を前提にお願いします:\n相関行列の結果:\n{correlation_matrix_reduced.to_string()}\n回帰分析の結果:\n{str(regression_results_reduced)}"
            response = model.generate_content(combined_text)
            st.session_state.conversation_history.append({"type": "response", "content": response.text})
            st.experimental_rerun()  # 送信後にページをリロードして入力欄をクリア

else:
    st.info("CSVファイルをアップロードしてください。")
