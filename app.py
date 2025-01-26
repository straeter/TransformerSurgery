import gc

import streamlit as st
import torch
import transformer_lens

from hooks import get_ablation_hook, get_activation_types
from stream_generator import generate_with_hooks

st.set_page_config(layout="wide")


def init_model(model_name, torch_dtype=torch.bfloat16):
    # Load a model (eg GPT-2 Small)
    with (st.spinner("Loading model...")):
        st.session_state.model_name = model_name
        st.session_state.model = transformer_lens.HookedTransformer.from_pretrained(
            model_name,
            torch_dtype=torch_dtype
        )


def init_session(model_name="gpt2-small"):
    if not "model_name" in st.session_state:
        st.session_state.model_name = model_name
        init_model(model_name)
        st.session_state.output = ""
        st.session_state.output_hooked = ""


def sidebar_settings():
    with st.sidebar:
        st.title("Settings")
        # with st.expander("Settings"):
        model_name = st.selectbox("Model", ["gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl", "gpt-neo-2.7B",
                                            "Llama-2-7b-chat"], index=0)
        st.number_input("Temperature", value=0.0, min_value=0.0, max_value=2.0, step=0.1, key="temperature")
        st.number_input("Max Tokens", value=100, min_value=0, step=1, max_value=st.session_state.model.cfg.n_ctx,
                        key="max_tokens")
        # stop_word = st.text_input("Stop Word", value="")
        # initial_prompt = st.text_area("Initial Prompt", value="")

        if model_name != st.session_state.model_name:
            print("Reloading model...")
            model = st.session_state.model
            model.cpu()
            del model
            gc.collect()
            torch.cuda.empty_cache()
            init_model(model_name)


# Main chat functionality
def main_window():
    st.title("TransformerSurgery")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.selectbox("Layer index", list(range(st.session_state.model.cfg.n_layers)), index=0, key="layer_idx")
    with col2:
        st.selectbox("Activation type",
                     get_activation_types(st.session_state.model.hook_dict, st.session_state.layer_idx), index=0,
                     key="act_type")
    with col3:
        st.selectbox("Head index", list(range(st.session_state.model.cfg.n_heads)), index=0, key="head_idx",
                     disabled=not st.session_state.act_type in ["key", "query", "value"])
    with col4:
        st.number_input("Position", value=-1, min_value=-1, max_value=1000, step=1, key="position")
    with col5:
        st.selectbox("Action", ["zero", "double"], index=0, key="ablation_type")

    prompt = st.text_input("Input prompt")

    col_out_1, col_out_2 = st.columns(2)

    with col_out_1:
        btn_generate = st.empty()
        st.subheader("Normal Output")
        output = st.empty()

        if btn_generate.button("Generate normally"):
            with (st.spinner("Running model...")):
                hooks = []
                model = st.session_state.model
                hooked_generator = generate_with_hooks(model, prompt, hooks, st.session_state.max_tokens,
                                                       st.session_state.temperature)
                st.session_state.output = ""
                for text in hooked_generator:
                    st.session_state.output = text
                    output.text(text)
        output.text(st.session_state.output)

    with col_out_2:
        btn_generate_hooked = st.empty()
        st.subheader("Hooked Output")
        hooked_output = st.empty()
        if btn_generate_hooked.button("Generate with hooks"):
            with (st.spinner("Running model with hooks...")):
                hooks = [get_ablation_hook(
                    act_type=st.session_state.act_type,
                    layer_idx=st.session_state.layer_idx,
                    head_idx=st.session_state.head_idx,
                    position=st.session_state.position,
                    ablation_type=st.session_state.ablation_type
                )]
                model = st.session_state.model
                hooked_generator = generate_with_hooks(model, prompt, hooks, st.session_state.max_tokens,
                                                       st.session_state.temperature)
                st.session_state.output_hooked = ""
                for text in hooked_generator:
                    st.session_state.output_hooked = text
                    hooked_output.text(text)
        hooked_output.text(st.session_state.output_hooked)


init_session()
sidebar_settings()
main_window()
