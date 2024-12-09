import streamlit as st
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.discrete.discrete_model import MNLogit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from tqdm import tqdm  # 진행 상태를 시각적으로 확인하기 위해 추가

np.random.seed(42)
# Streamlit 앱 제목
st.title("Apple Sugar Grade Prediction and Analysis")

# 데이터 불러오기 옵션
st.subheader("Load Dataset")
data_source = st.radio("Select data source:", options=["Upload CSV", "GitHub Link"], index=1)

if data_source == "Upload CSV":
    # 파일 업로드 섹션
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.write(data.head())
else:
    # GitHub에서 데이터 가져오기
    github_url = st.text_input(
        "Enter the GitHub raw URL to your dataset:",
        "https://raw.githubusercontent.com/selffish234/AI_Basic/refs/heads/main/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/apple_sugar_data4.csv"
    )
     
    st.button("Load Data")
    data = pd.read_csv(github_url)
    st.write("Dataset Preview:")
    st.write(data.head())
        
# # 파일 업로드 섹션
# uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")
# if uploaded_file is not None:
#     data = pd.read_csv(uploaded_file)
#     st.write("Dataset Preview:")
#     st.write(data.head())

    # 종속변수 변환
    data['sugar_grade'] = data['sugar_grade'].astype('category')

    # 독립변수 및 종속변수 정의
    X = data[['soil_ec', 'soil_temper', 'soil_humidity', 'soil_potential',
            'temperature', 'humidity', 'sunshine', 'daylight_hours']]
    y_linear = data['sugar_content_nir']

    # 선형 회귀 분석
    if st.checkbox("Run Linear Regression Analysis"):
        X_linear = sm.add_constant(X)
        linear_model = sm.OLS(y_linear, X_linear).fit()
        st.subheader("Linear Regression Summary")
        st.text(linear_model.summary())

        # VIF 계산
        st.subheader("Variance Inflation Factor (VIF)")
        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(X_linear.values, i) for i in range(X_linear.shape[1])]
        vif["features"] = X_linear.columns
        st.write(vif)

    # 다항 로지스틱 회귀
    if st.checkbox("Run Multinomial Logistic Regression"):

        data['sugar_grade'] = data['sugar_grade'].cat.reorder_categories(['A', 'B', 'C'], ordered=True)
        y_logistic = data['sugar_grade'].cat.codes 

        X2 = data[['soil_temper', 'soil_humidity', 'soil_potential',
        'temperature', 'humidity', 'sunshine', 'daylight_hours']]
        X2 = sm.add_constant(X2)

        # OLS 검정
        multi_model = sm.OLS(y_logistic, X2)
        fitted_multi_model = multi_model.fit()
        st.subheader("Multinomial Logistic Regression Summary")
        st.text(fitted_multi_model.summary())

        data['sugar_grade'] = data['sugar_grade'].cat.reorder_categories(['A', 'B', 'C'], ordered=True)
        y_logistic = data['sugar_grade'].cat.codes 
        X_train, X_test, y_train, y_test = train_test_split(X, y_logistic, test_size=0.2, random_state=42)
        multi_logistic_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1)
        multi_logistic_model.fit(X_train, y_train)
        y_pred = multi_logistic_model.predict(X_test)

    

        st.subheader("Confusion Matrix (Logistic Regression)")
        st.write(confusion_matrix(y_test, y_pred))
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

    #XGBoost 모델 학습 및 평가
    if st.checkbox("Run XGBoost Model"):
        data['sugar_grade'] = data['sugar_grade'].cat.reorder_categories(['A', 'B', 'C'], ordered=True)
        y_logistic = data['sugar_grade'].cat.codes 

        X2 = data[['soil_temper', 'soil_humidity', 'soil_potential',
        'temperature', 'humidity', 'sunshine', 'daylight_hours']]
        
        X_train, X_test, y_train, y_test = train_test_split(X2, y_logistic, test_size=0.2, random_state=42)
        xgb_model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=3,
            max_depth=3,
            learning_rate=0.01,
            n_estimators=100,
            subsample=0.7,
            colsample_bytree=0.6,
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)


        st.subheader("Confusion Matrix (XGBoost)")
        st.write(confusion_matrix(y_test, y_pred))
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        # SHAP 시각화
    if st.checkbox("Explain XGBoost Predictions with SHAP"):
        explainer = shap.Explainer(xgb_model, X_train)
        shap_values = explainer(X_test)

        st.subheader("SHAP Summary Plot (Bar)")
        shap.summary_plot(shap_values[:, :, 0], X_test, plot_type="bar", show=False)
        st.pyplot(plt.gcf())
        plt.clf()

        # st.subheader("SHAP Force Plot")
        # shap.force_plot(explainer.expected_value[0], shap_values[0, :, 0].values, X_test_sc.iloc[0], matplotlib=True)
        # st.pyplot(plt.gcf())
        # plt.clf()

        # SHAP Bee Swarm Plot for Grade A
        st.subheader("SHAP Bee Swarm Plot for Grade A")
        class_index = 0  # A등급 클래스 인덱스

        # A등급에 대한 SHAP 값 선택
        shap_values_for_class = shap_values[:, :, class_index]  # A등급에 대한 SHAP 값 선택

        # Bee Swarm Plot 생성
        shap.summary_plot(
            shap_values_for_class,
            X_test.values,  # NumPy 배열로 변환
            show=False,  # Streamlit에서 pyplot 사용
            plot_size=(10, 8)  # 그래프 크기 설정
        )
        st.pyplot(plt.gcf())
        plt.clf()


        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test, columns=X_train.columns)


        # 클래스 0 (A등급)에 대한 Decision Plot
        class_idx = 0  # A등급 클래스 인덱스
        st.subheader("SHAP Decision Plot for Grade A")

        # 클래스 0 (A등급)에 대한 SHAP 값 선택
        shap_values_class = shap_values[:, :, class_idx]

        # Decision Plot 생성
        fig, ax = plt.subplots(figsize=(12, 6))
        shap.decision_plot(
            base_value=explainer.expected_value[class_idx],  # 클래스 A 기준값
            shap_values=shap_values_class.values,  # 클래스 A의 SHAP 값
            features=X_test,  # 독립 변수 데이터
            feature_names=X_test.columns.tolist(),  # 독립 변수 이름
            show=False  # Streamlit에서 pyplot 사용
        )

        # Streamlit에서 플롯 출력
        st.pyplot(fig)


    if st.checkbox("Explain Predictions with LIME1"):

        # LIME Explainer 생성
        lime_explainer = LimeTabularExplainer(
            training_data=X_train.values,
            mode='classification',
            training_labels=y_train,
            feature_names=X_train.columns,
            class_names=["A", "B", "C"],
            discretize_continuous=True
        )

        # A등급 데이터 필터링
        st.subheader("Filtering Grade A Data")
        A_indices = np.where(y_test == 0)[0]  # y_test에서 A등급(0으로 인코딩)인 인덱스 찾기
        A_data = X_test.iloc[A_indices]       # A등급에 해당하는 데이터 추출
        st.write(f"Number of Grade A samples: {len(A_data)}")

        # A등급 데이터의 LIME 기여도 계산
        st.subheader("Calculating LIME Contributions for Grade A")
        all_lime_contributions = []

        progress_bar = st.progress(0)  # 진행 상태 표시

        for i in range(len(A_data)):  # A등급 데이터 포인트를 반복
            data_point = A_data.iloc[i].values
            lime_exp = lime_explainer.explain_instance(
                data_point,
                xgb_model.predict_proba,
                num_features=len(X_train.columns),
                labels=[0]  # "A" 클래스에 대한 설명
            )
            # "A" 클래스의 LIME 기여도 저장
            lime_weights = lime_exp.as_list(label=0)
            feature_contributions = {w[0]: w[1] for w in lime_weights}
            all_lime_contributions.append(feature_contributions)
            
            # 진행 상태 업데이트
            progress_bar.progress((i + 1) / len(A_data))

        # 모든 A등급 데이터의 기여도 평균 계산
        average_contributions = pd.DataFrame(all_lime_contributions).mean()

        # 평균 기여도를 Bar Plot으로 시각화
        st.subheader("Average LIME Feature Contributions for Grade A")
        plt.figure(figsize=(12, 6))
        average_contributions.sort_values().plot(kind="barh", color="skyblue")
        plt.xlabel("Average Feature Contribution")
        plt.title("Average LIME Feature Contribution for Grade A")
        plt.gca().invert_yaxis()
        st.pyplot(plt.gcf())




    if st.checkbox("Explain Predictions with LIME2"):
        # LIME Explainer 생성
        lime_explainer = LimeTabularExplainer(
            training_data=X_train.values,
            mode='classification',
            training_labels=y_train,
            feature_names=X_train.columns,
            class_names=["A", "B", "C"],
            discretize_continuous=True
        )
        
        # 데이터 포인트 선택
        data_point = X_test.iloc[132].values
        
        # 클래스 선택 옵션 추가
        selected_class = st.selectbox(
            "Select the grade to explain",
            options=["A", "B", "C"],
            index=0  # 기본값은 grade A
        )
        
        # 클래스 인덱스 매핑
        class_mapping = {"A": 0, "B": 1, "C": 2}
        class_index = class_mapping[selected_class]
        
        # LIME 설명 생성 (특정 클래스에 대해)
        lime_exp = lime_explainer.explain_instance(
            data_point,
            xgb_model.predict_proba,
            num_features=len(X_train.columns),
            labels=[class_index]  # 선택된 클래스에 대한 설명
        )
        
        # 그래프 출력: LIME 바 차트
        plt.figure(figsize=(12, 6))
        st.subheader(f"LIME Explanation for Grade {selected_class}")
        st.subheader("Feature Contributions (Bar Plot)")
        lime_fig = lime_exp.as_pyplot_figure(label=class_index)  # 선택된 클래스에 대한 설명
        st.pyplot(lime_fig)

        # # Bar Plot으로 Feature Contributions 시각화
        # st.subheader("Feature Contributions (Bar Plot)")
        # lime_weights = lime_exp.as_list(label=class_index)  # 선택된 클래스에 대한 기여도
        # feature_names = [w[0] for w in lime_weights]
        # feature_contributions = [w[1] for w in lime_weights]

        # plt.figure(figsize=(12, 6))
        # plt.barh(feature_names, feature_contributions, color="skyblue")
        # plt.xlabel("Feature Contribution")
        # plt.title(f"LIME Feature Contribution for Grade {selected_class} (Bar Plot)")
        # plt.gca().invert_yaxis()
        # st.pyplot(plt.gcf())





    

