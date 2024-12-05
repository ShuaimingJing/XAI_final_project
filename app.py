import tensorflow as tf
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import requests
import io
import openai
import base64
import os
from openai import OpenAI

# OpenAI API setup
client = OpenAI(api_key=st.secrets['open_ai_key'])

# Load a single model
def load_model(model_name):
    model_mapping = {
        "ResNet50": tf.keras.applications.ResNet50,
        "VGG16": tf.keras.applications.VGG16,
        "EfficientNetB0": tf.keras.applications.EfficientNetB0,
        "MobileNet": tf.keras.applications.MobileNet,
    }
    return model_mapping[model_name](weights="imagenet")

# Preprocess image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    st.image(img, caption="Uploaded Image",)
    img_array = image.img_to_array(img)
    return preprocess_input(np.expand_dims(img_array, axis=0))

# Generate saliency map
def generate_saliency_map(model, img_tensor, class_idx):
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        preds = model(img_tensor)
        top_class_output = preds[:, class_idx]

    grads = tape.gradient(top_class_output, img_tensor)
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0]
    return (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_min(saliency))


# Analyze results using ChatGPT
def analyze_with_chatgpt(model_name, label, score, saliency):
    buffer = io.BytesIO()
    saliency_img = Image.fromarray((saliency.numpy() * 255).astype('uint8')).convert("RGB")
    saliency_img.save(buffer, format="PNG")
    buffer.seek(0)

    saliency_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    prompt = f"""
    Analyze the saliency map and prediction for the image:
    - Model: {model_name}
    - Predicted Label: {label}
    - Confidence Score: {score:.2f}
    - Saliency Map (Base64 encoded): {saliency_base64[:200]}... (truncated)

    Based on these details, explain what the model is focusing on and how its decision-making process can be interpreted.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Analyze the saliency map for the image."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error communicating with OpenAI API: {e}")
        return None

st.sidebar.title("🌟Saliency Map Tutorial")
sections = st.sidebar.radio("Navigate", ["Introduction", "Theory", "Demo"])

if sections == "Introduction":
    st.title("Welcome to the Saliency Map Tutorial!")
    st.markdown("""
    Saliency maps are a key tool in explainable AI, helping visualize which parts of an image contribute most to a neural network's predictions.
    """)
    st.write("""
    In this tutorial, you will:
    - Learn the theory behind saliency maps.
    - Explore their real-world use cases.
    - Interact with models to generate and compare saliency maps.
    """)

if sections == "Theory":
    st.title("Understanding Saliency Maps")

    # What are saliency maps?
    st.markdown("""
    **Saliency maps** help visualize which parts of an image contribute most to a model's prediction. 
    They are computed using gradients of the model's output with respect to the input.
    """)
    st.markdown("### Definition")
    st.write("""
    A saliency map is an interpretability tool used in deep learning, particularly for image classification models, to highlight which pixels or regions of an image are most important for the model's prediction. It visualizes the gradient of the model’s output with respect to the input image, showing how much each pixel influenced the classification decision.

    In other words, a saliency map indicates the areas of focus in an image that the model relies on when making its prediction. Brighter regions in the map correspond to pixels that had a higher impact on the prediction, while darker regions had less influence.
            """)
    
    col1, col2 = st.columns(2)

    # Add the first image to the first column
    with col1:
        st.image("assets/original_image.png", caption="Example of Printer")

    # Add the second image to the second column
    with col2:
        st.image("assets/saliency_map.png", caption="Example of a Saliency Map")

    # Mathematical foundation
    st.markdown("### Mathematical Foundation")
    st.latex(r"S_{ij} = \left| \frac{\partial y}{\partial x_{ij}} \right|")
    st.markdown("""
    - \(y\): Model's output (e.g., confidence for a class).
    - \(x_{ij}\): Pixel at position \((i, j)\).
    """)


    # Applications
    st.markdown("### Applications")
    st.markdown("- Debugging models.")
    st.markdown("- Explaining decisions of models.")
    st.markdown("- Certain areas: Medical imaging, autonomous driving, etc.")



if sections == "Demo":

    # Sidebar for inputs
    st.sidebar.title("Options")
    img_path = st.sidebar.text_input("Enter the path to an image:")
    st.sidebar.markdown("""
    ### Instructions
    1. Enter the full path to the image file you want to analyze.
    2. Some options with images in the github:
    
    assets/umbrella.png
    
    
    assets/wallaby.png
    
    
    assets/elephant.png
    """)
    model_choice = st.sidebar.selectbox("Choose a classification model", ["ResNet50", "VGG16", "EfficientNetB0", "MobileNet"])

    if st.sidebar.button("🔍 Classify and Generate Saliency Map"):
        if img_path and os.path.exists(img_path):
            with st.spinner("Processing image and generating results..."):
                model = load_model(model_choice)
                img_array = preprocess_image(img_path)

                preds = model.predict(img_array)
                decoded_preds = decode_predictions(preds, top=1)[0]
                imagenet_id, label, score = decoded_preds[0]

                st.subheader("Prediction Results")
                st.write(f"**Model:** {model_choice}")
                st.write(f"**Predicted Label:** {label}")
                st.write(f"**Confidence Score:** {score * 100:.2f}%")

                # Generate and display saliency map
                img_tensor = tf.convert_to_tensor(img_array)
                class_idx = tf.argmax(preds[0])
                saliency = generate_saliency_map(model, img_tensor, class_idx)

                st.subheader("Saliency Map")
                plt.figure(figsize=(5, 5))
                plt.imshow(saliency, cmap="hot")
                plt.axis("off")
                st.pyplot(plt)

                # Analyze using ChatGPT
                analysis = analyze_with_chatgpt(model_choice, label, score, saliency)
                st.subheader("ChatGPT Analysis")
                st.write(analysis)
        else:
            st.warning("Please enter a valid image path.")

    if st.sidebar.button("📊 Compare All Models"):
        if img_path and os.path.exists(img_path):
            with st.spinner("Comparing models..."):
                models = {
                    "ResNet50": tf.keras.applications.ResNet50(weights="imagenet"),
                    "VGG16": tf.keras.applications.VGG16(weights="imagenet"),
                    "EfficientNetB0": tf.keras.applications.EfficientNetB0(weights="imagenet"),
                    "MobileNet": tf.keras.applications.MobileNet(weights="imagenet"),
                }
                img_array = preprocess_image(img_path)
                cols = st.columns(len(models))

                for i, (model_name, model) in enumerate(models.items()):
                    with cols[i]:
                        preds = model.predict(img_array)
                        decoded_preds = decode_predictions(preds, top=1)[0]
                        imagenet_id, label, score = decoded_preds[0]

                        st.write(f"**Model:** {model_name}")
                        st.write(f"**Prediction:** {label}")
                        st.write(f"**Confidence:** {score * 100:.2f}%")

                        img_tensor = tf.convert_to_tensor(img_array)
                        class_idx = tf.argmax(preds[0])
                        saliency = generate_saliency_map(model, img_tensor, class_idx)

                        plt.figure(figsize=(4, 4))
                        plt.imshow(saliency, cmap="hot")
                        plt.axis("off")
                        st.pyplot(plt)
        else:
            st.warning("Please enter a valid image path.")
