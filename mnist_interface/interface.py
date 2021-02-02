import os
import torch
import torch.nn.functional as F
from model import Net
from model import transform_input
import gradio as gr
from PIL import Image


def inference(img):
    # transforms ndarray and adds batch dimension
    img = transform_input(img).unsqueeze(0)

    with torch.no_grad():
        # gets probability prediction for each label
        output_probabilities = F.softmax(model(img), dim=1)[0]

    return {labels[i]: float(output_probabilities[i]) for i in range(len(labels))}


# Loads latest model state
model_timestamps = [filename[10:-3]
                    for filename in os.listdir(os.getcwd()+"/mnist_interface/saved_models/")]
latest_timestamp = max(model_timestamps)
model = Net()
model.load_state_dict(torch.load(
    os.getcwd()+f"/mnist_interface/saved_models/mnist-cnn-{latest_timestamp}.pt"))
model.eval()

# Creates gradio interface
labels = range(10)  # 1-9 labels
outputs = gr.outputs.Label(num_top_classes=5)
gr.Interface(fn=inference, inputs='sketchpad', outputs=outputs, title="MNIST interface",
             description="Draw a number from 0-9 in the box and click submit to see the model's predictions.").launch()
