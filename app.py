
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from detector import create_model

st.set_page_config(
    page_title="DeepShield AI",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ DeepShield AI")
st.subheader("AI-Powered Image Detection")

st.write("Upload an image to check whether it is Real or AI Generated.")

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    model = create_model().to(device)
    model.load_state_dict(
        torch.load("deepshield_model.pth", map_location=device)
    )
    model.eval()
    return model

uploaded_file = st.file_uploader(
    "Upload your image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Detect Image", use_container_width=True):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = load_model()(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        prediction = outputs.argmax(dim=1).item()
        classes = ["Fake", "Real"]
        confidence = probabilities[prediction].item() * 100

        if prediction == 0:
            st.error("AI-GENERATED IMAGE")
        else:
            st.success("REAL IMAGE")

        st.metric("Confidence", f"{confidence:.2f}%")

        st.caption("This is an AI prediction, not a guarantee of authenticity.")