import gradio as gr

# Function for Prediction
def predict(audio, text, threshold):
    # Placeholder for prediction logic
    prediction = "Keyword Detected"
    probability = 75.0  # Example probability value
    inference_time = "0.023"  # Example inference time

    return prediction, probability, inference_time

# Gradio Interface
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Audio(type="filepath", label="Record or Upload Audio"),
        gr.Textbox(label="Keyword Text"),
        gr.Slider(minimum=0, maximum=1, value=0.5, step=0.1, label="Threshold")
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Number(label="Probability (%)"),
        gr.Textbox(label="Inference Time (seconds)")
    ],
    title="Keyword Spotting using PhonMatchNet",
    description="Record or upload audio and enter keyword text to predict keyword probability.",
    theme="default"
)


# Launch Configuration
if __name__ == "__main__":
    demo.queue()  # Enable queue to support generators
    demo.launch(share=True)