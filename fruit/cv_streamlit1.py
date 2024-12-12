import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown

# 클래스 매핑 딕셔너리
rev_dict = {
    0: "freshoranges",
    1: "rottenbanana",
    2: "신선하지 않은 사과입니다.",
    3: "rottenoranges",
    4: "신선한 사과입니다.",
    5: "freshbanana"
}

# 모델 로드 함수
@st.cache_resource
def load_model():
    #url = "https://drive.google.com/file/d/1hTyhx5cZ6kUHyVjeIC3eF7EJeSAAzbDY/view?usp=drive_link"  # Google Drive의 파일 ID로 대체
    url = "https://drive.google.com/uc?id=1hTyhx5cZ6kUHyVjeIC3eF7EJeSAAzbDY"
    output = "fruit_classifier3.h5"
    gdown.download(url, output, quiet=False)
    model = tf.keras.models.load_model(output)
    return model

model = load_model()

# Streamlit 페이지 제목
st.title("Fruit Freshness Classifier") 
st.write("Upload an image to classify its freshness.")

# 이미지 업로드
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 이미지를 열고 전처리
    image = Image.open(uploaded_file).convert("RGB")
    resized_image = image.resize((240, 240))  # 모델 입력 크기와 맞추기
    image_array = np.array(resized_image) / 255.0  # 정규화

    # 모델 예측
    st.write("Processing the image...")
    predicted_probs = model.predict(np.expand_dims(image_array, axis=0))  # 예측
    predicted_class_index = np.argmax(predicted_probs)  # 가장 높은 확률의 클래스 인덱스
    confidence = predicted_probs[0][predicted_class_index] * 100  # 예측 확률

    # 예측 결과 매핑
    try:
        predicted_label = rev_dict[predicted_class_index]
        st.success(f"Prediction: {predicted_label}")
        st.info(f"Confidence: {confidence:.2f}%")
    except KeyError:
        st.error(f"Unexpected class index: {predicted_class_index}")
        predicted_label = "Unknown"

    # 업로드한 이미지와 결과 표시
    st.image(image, caption=f"Uploaded Image - Prediction: {predicted_label} (Confidence: {confidence:.2f}%)")
