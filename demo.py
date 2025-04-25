import os
import sys
import argparse
import torch
import gradio as gr
import numpy as np

# Make sure to import your model and utilities
sys.path.append(".")  # Add current directory to path
from model import ukws
from dataset import libriphrase
from pathlib import Path

# Initialize the model with parameters matching your trained model
def load_model(checkpoint_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Load the trained model from checkpoint."""
    # Adjust these parameters to match your training configuration
    kwargs = {
        'vocab': 72,  # Adjust based on your actual phoneme vocabulary size
        'text_input': 'g2p_embed',
        'audio_input': 'both',
        'stack_extractor': True,
        'subsequence_phoneme': True,
        'frame_length': 400,
        'hop_length': 160,
        'num_mel': 40,
        'sample_rate': 16000,
        'log_mel': True,
    }
    
    model = ukws.BaseUKWS(**kwargs)
    
    # Load the saved weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # If you saved with accelerator, you may need to extract model state_dict
    if "module" in checkpoint:
        # Handle distributed training checkpoint format
        model.load_state_dict(checkpoint["module"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model

# Audio preprocessing function
def preprocess_audio(audio_array, sample_rate):
    """Preprocess the audio to match the training data format."""
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        # You may need to add resampling code here
        pass
    
    # Convert to float32 if needed
    audio_array = audio_array.astype(np.float32)
    
    # Normalize audio
    if np.abs(audio_array).max() > 0:
        audio_array = audio_array / np.abs(audio_array).max()
    
    return audio_array, 16000

# Text processing function
def process_text(text):
    """Process the text input to match the model's expected format."""
    # This will depend on how your model handles text
    # If using g2p_embed, you might need to convert text to phonemes
    # For simplicity, return the text as is for now
    return text

# Phoneme conversion if needed
def text_to_phonemes(text):
    """Convert text to phonemes using the same method used during training."""
    # You would implement the same g2p conversion used in training
    # For example, if you used a specific library:
    # from g2p_en import G2p
    # g2p = G2p()
    # phonemes = g2p(text)
    # return phonemes
    
    # Placeholder - implement based on your training pipeline
    return text  

# Inference function
def run_inference(model, audio_array, text, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Run inference on audio and text input."""
    with torch.no_grad():
        # Process audio to get features similar to your training pipeline
        # This is a simplified version - you'll need to match your actual preprocessing
        
        # Convert audio to tensor
        x = torch.tensor(audio_array).float().to(device)
        
        # Process text to get features
        # This depends on your model's text_input configuration
        y = text_to_phonemes(text)
        # Convert to tensor (placeholder - implement based on your model)
        y_tensor = torch.tensor([ord(c) for c in y]).long().to(device)
        
        # Calculate lengths
        x_len = torch.tensor([len(audio_array)]).long().to(device)
        y_len = torch.tensor([len(y_tensor)]).long().to(device)
        
        # For "both" audio_input, you'd need Google embeddings too
        # This is a simplified version - you'll need to adapt to your specific model
        
        # Run inference
        # Note: In a real implementation, you'd need to match the exact input format
        # expected by your model based on training configuration
        prob, _, _, _, _, _ = model(x.unsqueeze(0), y_tensor.unsqueeze(0), x_len, y_len)
        
        return prob.item()

# Main Gradio interface
def create_demo(model_path):
    model = load_model(model_path)
    
    def process_inputs(audio, text):
        if audio is None:
            return "Please provide an audio input."
        
        # Process audio
        audio_array, sample_rate = audio
        audio_array, sample_rate = preprocess_audio(audio_array, sample_rate)
        
        # Process text
        if not text:
            return "Please provide a text query."
        
        processed_text = process_text(text)
        
        # Run inference
        match_probability = run_inference(model, audio_array, processed_text)
        
        # Format results
        threshold = 0.5  # Adjust based on your model's performance
        result = f"Match Probability: {match_probability:.4f}\n"
        if match_probability > threshold:
            result += "✅ Audio MATCHES the text query"
        else:
            result += "❌ Audio does NOT match the text query"
            
        return result
    
    demo = gr.Interface(
        fn=process_inputs,
        inputs=[
            gr.Audio(type="numpy", label="Speak the phrase"),
            gr.Textbox(label="Text Query")
        ],
        outputs=gr.Textbox(label="Results"),
        title="PhonMatchNet Keyword Spotting Demo",
        description="Upload or record audio and provide a text query to see if they match.",
        examples=[
            ["example_audio.wav", "hello world"],
            ["example_audio2.wav", "activate smart home"]
        ],
        article="""
        ## How to use this demo
        
        1. Upload an audio file or record yourself saying a phrase
        2. Enter the text you want to check against the audio
        3. Submit and see the match probability
        
        This demo uses a PhonMatchNet-like model trained to detect if an audio snippet contains a specific phrase.
        """
    )
    
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradio Demo for PhonMatchNet")
    parser.add_argument("--checkpoint", type=str, required=True, default="./results/reproduction/checkpoint/epoch_13_best")
    args = parser.parse_args()
    
    demo = create_demo(args.checkpoint)
    demo.launch(share=True)