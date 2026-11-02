
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from detector import create_model
from datetime import datetime

st.set_page_config(
    page_title="DeepShield AI",
    page_icon="🛡️",
    layout="wide"
)

# Professional Dark Theme
st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: white;
}
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #38bdf8;
    text-align: center;
}
.sub-title {
    text-align: center;
    color: #94a3b8;
    font-size: 17px;
}
div[data-testid="stMetric"] {
    background: #172033;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #334155;
}
div.stButton > button {
    background: #0284c7;
    color: white;
    border-radius: 10px;
    border: none;
    height: 45px;
    font-weight: bold;
}
div.stButton > button:hover {
    background: #0369a1;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="main-title">🛡️ DeepShield AI</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-title">AI-Powered Image Authenticity Detection</div>',
    unsafe_allow_html=True
)

st.write("")
st.divider()

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    model = create_model().to(device)
    model.load_state_dict(
        torch.load("deepshield_model.pth", map_location=device)
    )
    model.eval()
    return model

# Session History
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar
with st.sidebar:
    st.title("🛡️ DeepShield")
    page = st.radio(
        "Navigation",
        ["🏠 Home", "📊 Dashboard", "📜 History"]
    )
    st.divider()
    st.caption("AI Image Detection System")
    st.caption(f"Running on: {device.upper()}")

# HOME PAGE
if page == "🏠 Home":

    st.header("🔍 Image Detection")
    st.write("Upload an image to identify whether it is Real or AI Generated.")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

            if st.button("🚀 Analyze Image", use_container_width=True):

                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor()
                ])

                image_tensor = transform(image).unsqueeze(0).to(device)

                with st.spinner("AI is analyzing your image..."):
                    with torch.no_grad():
                        outputs = load_model()(image_tensor)
                        probabilities = torch.softmax(outputs, dim=1)[0]

                prediction = outputs.argmax(dim=1).item()
                classes = ["AI GENERATED", "REAL"]
                result = classes[prediction]
                confidence = probabilities[prediction].item() * 100

                st.divider()
                st.subheader("Detection Result")

                if prediction == 0:
                    st.error("🤖 AI-GENERATED IMAGE")
                else:
                    st.success("📷 REAL IMAGE")

                st.metric("AI Model Confidence", f"{confidence:.2f}%")
                st.progress(int(confidence))

                st.caption(
                    "Prediction is not a guarantee of authenticity."
                )

                st.session_state.history.append({
                    "Image": uploaded_file.name,
                    "Result": result,
                    "Confidence": f"{confidence:.2f}%",
                    "Time": datetime.now().strftime("%d-%m-%Y %I:%M %p")
                })

        except Exception as e:
            st.error(f"Unable to process image: {e}")

# DASHBOARD PAGE
elif page == "📊 Dashboard":

    st.header("📊 Detection Dashboard")

    total = len(st.session_state.history)
    ai_count = sum(
        1 for x in st.session_state.history
        if x["Result"] == "AI GENERATED"
    )
    real_count = total - ai_count

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Images", total)
    col2.metric("AI Generated", ai_count)
    col3.metric("Real Images", real_count)

    st.divider()
    st.subheader("Detection Statistics")

    if total > 0:
        st.bar_chart({
            "Images": {
                "AI Generated": ai_count,
                "Real": real_count
            }
        })
    else:
        st.info("Upload and analyze images to see statistics.")

# HISTORY PAGE
elif page == "📜 History":

    st.header("📜 Detection History")

    if st.session_state.history:

        for item in reversed(st.session_state.history):
            with st.container(border=True):
                st.write(f"**Image:** {item['Image']}")
                st.write(f"**Result:** {item['Result']}")
                st.write(f"**Confidence:** {item['Confidence']}")
                st.caption(item["Time"])

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

    else:
        st.info("No detection history available.")

st.divider()
st.caption(
    "© 2026 DeepShield AI | AI-Powered Image Detection"
)
