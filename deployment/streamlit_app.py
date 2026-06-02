import streamlit as st
import torch
import numpy as np
from pathlib import Path
import tempfile
import json
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.mha_utils import load_mha, save_mha, SPINE_LABEL_MAP
from inference import Predictor
from models import create_model
from visualization.visualizer import plot_slice, plot_overlay

st.set_page_config(page_title="Lumbar Spine MRI Segmentation", layout="wide")
st.title("Lumbar Spine MRI Segmentation")
st.markdown("Upload a model checkpoint and MRI volume to get segmentations.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'config' not in st.session_state:
    st.session_state.config = {
        'model': {'name': 'atm_net', 'num_classes': 12, 'in_channels': 1,
                  'params': {'features': [32, 64, 128, 256, 512]}},
        'inference': {'tta': True},
    }

with st.sidebar:
    st.header("Model Configuration")
    checkpoint_file = st.file_uploader("Upload Model Checkpoint (.pt)", type=['pt', 'pth'])
    model_name = st.selectbox("Model Architecture", ['atm_net', 'unet', 'attention_unet', 'unet_plus_plus', 'nnunet'])
    use_tta = st.checkbox("Use Test-Time Augmentation", True)

    if st.button("Load Model"):
        if checkpoint_file is not None:
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
                tmp.write(checkpoint_file.read())
                ckpt_path = tmp.name

            st.session_state.config['model']['name'] = model_name
            st.session_state.config['inference']['tta'] = use_tta

            model = create_model(st.session_state.config)
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state['model_state_dict'])
            st.session_state.predictor = Predictor(model, st.session_state.config, device)
            st.success("Model loaded successfully!")
        else:
            st.error("Please upload a checkpoint file")

st.header("Inference")
mri_file = st.file_uploader("Upload MRI Volume (.mha)", type=['mha', 'mhd'])

if mri_file and st.session_state.predictor is not None:
    with tempfile.NamedTemporaryFile(suffix='.mha', delete=False) as tmp:
        tmp.write(mri_file.read())
        mri_path = tmp.name

    with st.spinner("Segmenting..."):
        sitk_img, image_arr = load_mha(mri_path)
        mask = st.session_state.predictor.predict_volume(mri_path)

    mid_slice = image_arr.shape[0] // 2

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("MRI Slice")
        fig_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
        plot_slice(image_arr, slice_idx=mid_slice, save_path=fig_path)
        st.image(fig_path, use_column_width=True)

    with col2:
        st.subheader("Segmentation")
        fig_path2 = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
        plot_slice(image_arr, mask, slice_idx=mid_slice, save_path=fig_path2)
        st.image(fig_path2, use_column_width=True)

    with col3:
        st.subheader("Overlay")
        fig_path3 = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
        plot_overlay(image_arr, mask, slice_idx=mid_slice, save_path=fig_path3)
        st.image(fig_path3, use_column_width=True)

    unique_labels = sorted(np.unique(mask).tolist())
    st.subheader("Detected Structures")
    for u in unique_labels:
        name = SPINE_LABEL_MAP.get(u, f'Unknown_{u}')
        count = int((mask == u).sum())
        st.write(f"- **{name}** (label {u}): {count} voxels")

    with tempfile.NamedTemporaryFile(suffix='.mha', delete=False) as out_tmp:
        save_mha(mask, out_tmp.name, ref_image=sitk_img)
        with open(out_tmp.name, 'rb') as f:
            st.download_button("Download Segmentation (.mha)", f, file_name=f"segmentation_{Path(mri_path).stem}.mha")

    Path(mri_path).unlink(missing_ok=True)

elif mri_file and st.session_state.predictor is None:
    st.warning("Please load a model first in the sidebar.")

st.header("Results Gallery")
st.markdown("*Upload an MRI and run inference to see results here.*")
