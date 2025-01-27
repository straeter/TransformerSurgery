import gc

import streamlit as st
import torch
import transformer_lens

from hooks import get_ablation_hook, get_activation_types
from stream_generator import generate_with_hooks
from utils import style_tokens, get_position_list

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

        col_t, col_m = st.columns(2)
        with col_t:
            temperature = st.number_input("Temperature", value=0.0, min_value=0.0, max_value=2.0, step=0.1, key="temperature",
                            help="Temperature scaling for sampling")
        with col_m:
            max_token = st.number_input("Max Tokens", value=100, min_value=0, step=1, max_value=st.session_state.model.cfg.n_ctx,
                            key="max_tokens", help="Maximum number of tokens to generate")

        col_k, col_p = st.columns(2)
        with col_k:
            top_k = st.number_input("Top K", value=None, min_value=1, step=1, key="top_k", help="Top K sampling")
        with col_p:
            top_p = st.number_input("Top P", value=None, min_value=0.0, max_value=1.0, step=0.05, key="top_p",
                            help="Top P sampling")

        st.session_state.model_param = {
            "temperature": temperature,
            "max_tokens": max_token,
            "top_k": top_k,
            "top_p": top_p
        }

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

        st.subheader(f"Model information: {st.session_state.model_name}")

        st.table({
            "Layers": st.session_state.model.cfg.n_layers,
            "Model dimension": st.session_state.model.cfg.d_model,
            "Attention heads": st.session_state.model.cfg.n_heads,
            "Head dimension": st.session_state.model.cfg.d_head,
            "Vocab size": st.session_state.model.cfg.d_vocab,
            "Context size": st.session_state.model.cfg.n_ctx,
            "Tokenizer": st.session_state.model.cfg.tokenizer_name
        })




# Main chat functionality
def main_window():
    st.title("TransformerSurgery")

    model = st.session_state.model

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        layer_idx = st.selectbox("Layer index", list(range(model.cfg.n_layers)), index=0, key="layer_idx", help="Index of the layer / transformer block",)
    with col2:
        act_type = st.selectbox(
            "Activation type", get_activation_types(model.hook_dict, st.session_state.layer_idx), index=0,
            key="act_type", help="Type of activation to ablate (like activation, attention, residual stream...)")
    with col3:
        head_idx = st.selectbox("Head index", list(range(model.cfg.n_heads)), index=0, key="head_idx", help="Head index (if values, keys or queries are selected)",
                     disabled=not st.session_state.act_type in ["key", "query", "value"])
    with col4:
        position = st.text_input("Position(s)", value="all", key="position", help="Affected position(s) in sequence, comma separated list or something like 2,4-7 etc.")
    with col5:
        ablation_type = st.selectbox("Action", ["zero", "double", "flip"], index=0, key="ablation_type", help="Action to perform on selected activations")

    position_list = get_position_list(position)

    prompt = st.text_input("Input prompt")

    tokens = model.tokenizer.tokenize(prompt) if prompt else []
    styled_tokens = style_tokens(tokens, position_list) if tokens else ""

    st.text("Tokenized prompt:")
    st.markdown(styled_tokens, unsafe_allow_html=True)

    col_out_1, col_out_2 = st.columns(2)

    with col_out_1:
        btn_generate = st.empty()
        st.subheader("Normal Output")
        output = st.empty()

        if btn_generate.button("Generate normally"):
            with (st.spinner("Running model...")):
                hooks = []
                hooked_generator = generate_with_hooks(model, prompt, hooks, **st.session_state.model_param)
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
                    act_type=act_type,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    position_list=position_list,
                    ablation_type=ablation_type
                )]
                hooked_generator = generate_with_hooks(model, prompt, hooks, **st.session_state.model_param)
                st.session_state.output_hooked = ""
                for text in hooked_generator:
                    st.session_state.output_hooked = text
                    hooked_output.text(text)
        hooked_output.text(st.session_state.output_hooked)


init_session()
sidebar_settings()
main_window()
