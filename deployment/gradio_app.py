import gradio as gr
import torch
import numpy as np
from pathlib import Path
import tempfile
import json

from utils.mha_utils import load_mha, save_mha, SPINE_LABEL_MAP
from inference import Predictor
from models import create_model
from visualization.visualizer import plot_slice, plot_overlay

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
predictor = None
config = {
    'model': {'name': 'atm_net', 'num_classes': 12, 'in_channels': 1,
              'params': {'features': [32, 64, 128, 256, 512]}},
    'inference': {'tta': True},
}


def load_model_for_gradio(checkpoint_path):
    global predictor
    model = create_model(config)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    predictor = Predictor(model, config, device)
    return "Model loaded successfully"


def predict_mri(file):
    if predictor is None:
        return None, None, "Please load a model first"

    with tempfile.NamedTemporaryFile(suffix='.mha', delete=False) as tmp:
        tmp.write(file)
        tmp_path = tmp.name

    try:
        sitk_img, image_arr = load_mha(tmp_path)
        mask = predictor.predict_volume(tmp_path)

        mid_slice = image_arr.shape[0] // 2

        with tempfile.NamedTemporaryFile(suffix='.mha', delete=False) as out_tmp:
            save_mha(mask, out_tmp.name, ref_image=sitk_img)
            out_path = out_tmp.name

        fig_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
        plot_slice(image_arr, mask, slice_idx=mid_slice, save_path=fig_path)

        overlay_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
        plot_overlay(image_arr, mask, slice_idx=mid_slice, save_path=overlay_path)

        unique = sorted(np.unique(mask).tolist())
        labels_found = [SPINE_LABEL_MAP.get(u, f'Unknown_{u}') for u in unique]

        return fig_path, overlay_path, f"Labels found: {', '.join(labels_found)}"

    except Exception as e:
        return None, None, f"Error: {str(e)}"
    finally:
        Path(tmp_path).unlink(missing_ok=True)


with gr.Blocks(title="Lumbar Spine MRI Segmentation") as demo:
    gr.Markdown("# Lumbar Spine MRI Segmentation")
    gr.Markdown("Upload an MRI volume (.mha) to get segmentation masks.")

    with gr.Row():
        with gr.Column():
            checkpoint_input = gr.File(label="Model Checkpoint (.pt)")
            load_btn = gr.Button("Load Model")

        with gr.Column():
            status = gr.Textbox(label="Status", value="Ready")

    with gr.Row():
        mri_input = gr.File(label="Upload MRI (.mha)")
        predict_btn = gr.Button("Segment", variant="primary")

    with gr.Row():
        with gr.Column():
            slice_view = gr.Image(label="MRI + Segmentation Slices")
        with gr.Column():
            overlay_view = gr.Image(label="Overlay")

    result_info = gr.Textbox(label="Results")

    load_btn.click(
        fn=lambda f: load_model_for_gradio(f.name) if f else "No file selected",
        inputs=[checkpoint_input],
        outputs=[status],
    )

    predict_btn.click(
        fn=lambda f: predict_mri(f) if f else (None, None, "No file uploaded"),
        inputs=[mri_input],
        outputs=[slice_view, overlay_view, result_info],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
