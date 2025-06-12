# 프롬프트 수정버전
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
import os
import random
import numpy as np
import tensorflow as tf
from openai import OpenAI
from PyPDF2 import PdfReader
import matplotlib

# 한글 깨짐 방지 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 페이지 설정
st.set_page_config(page_title="XAI 기반 환자 예측 해석", layout="wide")
st.title("🩺 중환자 패혈증 맞춤 진단 도움")

# 시드 고정
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# OpenAI API 설정
client = OpenAI(api_key="")

# 지침서 PDF 로딩 및 텍스트 추출
@st.cache_data
def load_guideline_text():
    try:
        reader = PdfReader("2024 질병관리청 성인 패혈증 초기치료지침서.pdf")
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return ""

guideline_text = load_guideline_text()

# 좌측 사이드바 구성
with st.sidebar:
    st.header("🛠 설정")
    file = st.file_uploader("📂 전처리된 엑셀 파일을 업로드하세요", type="xlsx")
    patient_id_input = st.text_input("환자 ID 입력")
    selected_id = None
    id_list = []

if 'show_judgement' not in st.session_state:
    st.session_state.show_judgement = False
if 'show_strategy' not in st.session_state:
    st.session_state.show_strategy = False

if file:
    df = pd.read_excel(file)
    st.success(f"✅ 파일 '{file.name}' 업로드 완료")

    if 'ID' in df.columns:
        id_list = df['ID'].tolist()

        with st.sidebar:
            st.markdown("#### 🔽 환자 ID 선택")
            selected_id = st.selectbox(
                "환자 ID를 선택하세요",
                options=id_list,
                format_func=lambda x: f"환자 ID: {x}"
            )

        if patient_id_input.isdigit():
            patient_id = int(patient_id_input)
        else:
            patient_id = selected_id

        drop_cols = ['ID', 'Adm', 'Death_time', 'Tr_time', 'endtime',
                    'FiO2', 'PCO2', 'PaO2', 'VASO_dose', 'NS',
                    'APACHE II score', 'SOFA', 'Mortality']
        df_clean = df.drop(columns=[col for col in drop_cols if col in df.columns])

        if 'sum' in df_clean.columns and 'Op' in df_clean.columns:
            df_clean.loc[(df_clean['sum'] == 0) & (df_clean['Op'].isna()), 'Op'] = 0
        if 'VASO' in df_clean.columns and 'cc/hr' in df_clean.columns:
            df_clean.loc[(df_clean['VASO'] == 0) & (df_clean['cc/hr'].isna()), 'cc/hr'] = 0

        df_clean = df_clean.dropna()

        if 'result' not in df_clean.columns:
            st.error("'result' 컬럼이 존재하지 않습니다. 파일을 확인해주세요.")
            st.stop()

        y = df_clean['result'].apply(lambda x: 0 if x in [0, 1] else 1)
        X = df_clean.drop(columns=['result'])

        @st.cache_data
        def train_model(X, y):
            cat = CatBoostClassifier(verbose=0, random_seed=42)
            lgbm = LGBMClassifier(random_state=42)
            xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            voting_model = VotingClassifier(estimators=[('cat', cat), ('lgbm', lgbm), ('xgb', xgb)], voting='soft')
            voting_model.fit(X, y)
            return voting_model

        model = train_model(X, y)
        explainer = shap.TreeExplainer(model.named_estimators_['cat'])
        shap_values_all = explainer.shap_values(X)
        shap_values = shap_values_all[1] if isinstance(shap_values_all, list) else shap_values_all

        if patient_id in df['ID'].values:
            row_idx = df[df['ID'] == patient_id].index[0]
            patient = X.iloc[[row_idx]]
            pred = model.predict(patient)[0]
            prob = model.predict_proba(patient)[0][1]

            with st.container():
                st.markdown("### 👤 환자 정보")
                with st.container(border = True):
                    name = df.loc[row_idx, 'Name'] if 'Name' in df.columns else '홍길동'
                    sex_value = df.loc[row_idx, 'Sex'] if 'Sex' in df.columns else None
                    gender = '남성' if sex_value == 0 else ('여성' if sex_value == 1 else '-')
                    age = df.loc[row_idx, 'Age'] if 'Age' in df.columns else '-'
                    adm = df.loc[row_idx, 'Adm'] if 'Adm' in df.columns else '-'

                    st.write(f"이름: {name}")
                    st.write(f"성별: {gender}")
                    st.write(f"나이: {age}")
                    st.write(f"입실일: {adm}")
                    st.write(f"환자 ID: {patient_id}")

            col1, col2 = st.columns(2)

            with col1:
                with st.container(border=True):
                    st.markdown("#### 📈 SHAP Summary Plot")
                    shap_val_row = shap_values[row_idx]
                    fig_summary = plt.figure(figsize=(8, 3))
                    shap.bar_plot(shap_val_row, feature_names=X.columns, max_display=10, show=False)
                    plt.xlabel("SHAP 변수 (사망률 영향)")  # x-axis 제목 변경
                    st.pyplot(fig_summary)

                    st.markdown("**📋 사망 확률 증가에 기여한 Top 5 위험 변수**")
                    positive_shap_indices = np.where(shap_val_row > 0)[0]
                    top5_risk_idx = positive_shap_indices[np.argsort(shap_val_row[positive_shap_indices])[::-1][:5]]
                    top5_risk_names = X.columns[top5_risk_idx].tolist()
                    top5_shap_values = shap_val_row[top5_risk_idx]

                    gpt_prompt = f"""
                    다음은 환자의 사망률에 영향을 미친 Top 5 변수입니다: {', '.join(top5_risk_names)}.
                    각 변수에 대해 다음과 같이 요약해 주세요:
                    - 변수명과 SHAP 값 기반으로 사망률 영향 수치
                    - 그 이유: 해당 기능 저하 or 변화로 인한 가능성
                    - 전체 길이는 2줄 이내로 요약
                    - 예시: '혈소판 수 감소로 인해 사망률이 24.1% 증가했음. 원인은 출혈 위험 또는 감염 조절 실패 가능성 있음.'
                    지침서와 모델 해석 결과를 참고하세요.
                    """

                    gpt_response = client.chat.completions.create(
                        model="gpt-4",
                        temperature=0.2,
                        messages=[
                            {"role": "system", "content": "당신은 중환자의학 전문의이며 의료데이터 해석 전문가입니다."},
                            {"role": "user", "content": gpt_prompt}
                        ]
                    )

                    explanation_lines = gpt_response.choices[0].message.content.strip().split("\n")
                    explanations_clean = [line.lstrip("0123456789. ") for line in explanation_lines if line.strip()]

                    df_risk = pd.DataFrame({
                        "변수명": top5_risk_names,
                        "SHAP 값": np.round(top5_shap_values, 4),
                        "해석": explanations_clean
                    })
                    st.table(df_risk)

            with col2:
                with st.container(border=True):
                    st.subheader("🧠 판단")
                    if st.button("판단 결과 보기"):
                        with st.expander("🔎 GPT 판단 해석 결과", expanded=True):
                            user_prompt = f"""
                                            환자의 사망률에 영향을 미친 주요 변수는 다음과 같습니다: {', '.join(top5_risk_names)}.

                                            중환자의학 전문의가 이해하기 쉽도록 다음 형식으로 의료적 판단을 요약해 주세요:

                                            ### 출력 형식 예시
                                            🩺 2. 의료적 판단 요약
                                            호흡 기능 악화가 가장 큰 리스크이므로 중환자실에서 집중적인 호흡 치료가 필요해 보여.

                                            나이는 조절 불가능하지만, 나이로 인한 리스크가 꽤 크기 때문에 전체적인 생리기능 보전이 더 중요해.

                                            의식 상태(GCS)가 유지된 건 좋은 신호야. 의식이 좋으면 회복 가능성이 조금 더 있어.

                                            위 예시를 참고하여 SHAP Top 5를 중심으로 작성해 주세요.
                                            """
                            response = client.chat.completions.create(
                                model="gpt-4",
                                temperature=0.2,
                                messages=[
                                    {"role": "system", "content": "당신은 중환자의학+의료데이터 해석 전문의입니다."},
                                    {"role": "user", "content": user_prompt}
                                ]
                            )
                            st.markdown(response.choices[0].message.content)

                with st.container(border=True):
                    st.subheader("💊 맞춤형 치료 전략")
                    if st.button("치료 전략 보기"):
                        st.session_state.show_strategy = True

                    if st.session_state.show_strategy:
                        user_prompt = f"""
                                            아래는 환자의 SHAP Top 5 변수입니다: {', '.join(top5_risk_names)}.

                                            이 변수들이 의미하는 임상적 정보를 토대로 환자 맞춤형 초기 치료 전략을 제시해 주세요.
                                            PDF 가이드는 참고용으로만 사용하세요. 모델 해석 결과에 기반한 실제적이고 구체적인 치료 제안이어야 합니다.

                                            ## 지침서 발췌문
                                            ```
                                            {guideline_text[:1500]}
                                            ```
                                            """
                        response = client.chat.completions.create(
                            model="gpt-4",
                            temperature=0.2,
                            messages=[
                                {"role": "system", "content": "당신은 중환자의학 전문의입니다."},
                                {"role": "user", "content": user_prompt}
                            ]
                        )
                        with st.expander("🩺 GPT 치료 전략 제안", expanded=True):
                            st.markdown(response.choices[0].message.content)

                with st.container(border=True):
                    st.subheader("📝 담당의 코멘트")
                    comment = st.text_area("담당의 코멘트를 입력하세요")
                    signer = st.text_input("서명")
                    if st.button("코멘트 저장"):
                        st.success(f"💾 코멘트 저장 완료: {comment} - 서명: {signer}")

        else:
            st.warning("해당 ID에 해당하는 환자를 찾을 수 없습니다.")
else:
    st.info("왼쪽 사이드바에서 환자 ID를 입력하거나 선택하세요.")
