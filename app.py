import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Sınıf etiketleri
CLASS_NAMES = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

# Modeli yükleme (önbelleğe alındı)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_model.keras")
    return model

model = load_model()

# Başlık ve açıklama
st.title("👁️ Retina Hastalığı Sınıflandırıcı")
st.write("Lütfen retina görüntüsü yükleyin. Modelimiz dört sınıftan birini tahmin edecektir:")

# Görüntü yükleme
uploaded_file = st.file_uploader("📤 Retina görüntüsü yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen Görüntü", use_container_width=True)

    # ✅ Görüntüyü 256x256 boyutuna getiriyoruz (modelin istediği şekil)
    img = image.resize((256, 256))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Batch boyutu ekle

    # 🔍 Tahmin
    prediction = model.predict(img_array)[0]
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # ✅ Sonucu göster
    st.markdown("### 🧠 Tahmin Sonucu")
    st.success(f"**{predicted_class.upper()}** sınıfı (%{confidence:.2f} güven)")

    # 📊 Tüm sınıf tahmin oranlarını yazdır
    st.markdown("### 📈 Tüm Sınıf Tahminleri:")
    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f"- {class_name}: %{prediction[i] * 100:.2f}")
