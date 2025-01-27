import gc

import streamlit as st
import torch
import transformer_lens

from hooks import get_ablation_hook, get_activation_aliases, get_layer_indices, act_aliases, get_hook_name
from stream_generator import generate_with_hooks
from styling import streamlit_style
from utils import style_tokens, get_position_list, load_custom_hooks

st.set_page_config(layout="wide")
st.markdown(streamlit_style,unsafe_allow_html=True)


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


default_parameters = {
    "temperature": 0.0,
    "max_tokens": 100,
    "top_k": None,
    "top_p": None
}


def sidebar_settings():
    with st.sidebar:
        st.title("Settings")
        # with st.expander("Settings"):
        model_name = st.selectbox("Model", ["gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl", "gpt-neo-2.7B",
                                            "Llama-2-7b-chat"], index=0)

        if model_name != st.session_state.model_name:
            print("Reloading model...")
            model = st.session_state.model
            model.cpu()
            del model
            gc.collect()
            torch.cuda.empty_cache()
            init_model(model_name)

        if st.button("Reset parameters"):
            st.session_state.temperature = default_parameters["temperature"]
            st.session_state.max_tokens = default_parameters["max_tokens"]
            st.session_state.top_k = default_parameters["top_k"]
            st.session_state.top_p = default_parameters["top_p"]

        col_t, col_m = st.columns(2)
        with col_t:
            st.number_input("Temperature", value=default_parameters["temperature"], min_value=0.0, max_value=2.0,
                            step=0.1, key="temperature",
                            help="Temperature scaling for sampling")
        with col_m:
            st.number_input("Max Tokens", value=default_parameters["max_tokens"], min_value=0, step=1,
                            max_value=st.session_state.model.cfg.n_ctx,
                            key="max_tokens", help="Maximum number of tokens to generate")

        col_k, col_p = st.columns(2)
        with col_k:
            st.number_input("Top K", value=default_parameters["top_k"], min_value=1, step=1, key="top_k",
                            help="Top K sampling")
        with col_p:
            st.number_input("Top P", value=default_parameters["top_p"], min_value=0.0, max_value=1.0, step=0.05,
                            key="top_p",
                            help="Top P sampling")

        # stop_word = st.text_input("Stop Word"

        st.session_state.model_param = {
            "temperature": st.session_state.temperature,
            "max_tokens": st.session_state.max_tokens,
            "top_k": st.session_state.top_k,
            "top_p": st.session_state.top_p
        }

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


def main_window():
    st.title("TransformerSurgery")
    with st.expander("Help"):
        st.write("TransformerSurgery allows you to interact with a transformer model by ablation of activations using TransformerLens."
                 "You can select the model and generation parameters in the sidebaar, choose the activation type and layer, the affected position and/or attention head, and then ablate (zero, double, or flip) the activations in the model."
                 "Then you can generate text from an input prompt with and without the ablation hooks to see the effect of the ablation on the model output."
                 "Beneath the prompt you can see the tokenized prompt with the affected positions highlighted (for custom hooks all positions are highlihgted by default)."
                 "Alternatively, you can also load custom hooks from the custom_hooks folder to apply custom ablation functions to the model activations.")

    model = st.session_state.model

    col_hooks, col_hook, _ = st.columns([1,1,3])
    with col_hooks:
        hook_type = st.selectbox(
            "Hook type", ["ablation hook", "custom hook"],
            key="hook_type", help="Choose hook type, custom hook loads hooks from folder custom_hooks")

    if hook_type == "custom hook":
        with col_hook:
            custom_hooks = load_custom_hooks("custom_hooks")
            custom_hook_key = st.selectbox("Custom hook", list(custom_hooks.keys()), key="custom_hook_key")
            custom_hook = custom_hooks[custom_hook_key]
    else:
        custom_hook = None

    col0, col1, col2, col3, col4, col5 = st.columns(6)
    with col0:
        act_type = st.selectbox(
            "Activation type", list(act_aliases.keys()), index=0,
            key="act_type", help="Type of activation to ablate (like activation, attention, residual stream...)")
    with col1:
        aliases = get_activation_aliases(act_type, model.hook_dict)
        act_name = st.selectbox(
            "Activation name", aliases,
            key="act_alias", help="Name of activation")
    with col2:
        layer_idx = st.selectbox(
            "Layer index", get_layer_indices(act_type, act_name, model.hook_dict), index=0, key="layer_idx",
            help="Index of the layer / transformer block",
            disabled=(act_type == "embedding")
        )

    act_alias = act_aliases[act_type][act_name]
    hook_name: str = get_hook_name(model.hook_dict, act_alias, layer_idx)

    if hook_type == "ablation hook":
        with col3:
            head_idx = st.selectbox(
                "Head index", list(range(model.cfg.n_heads)), index=0, key="head_idx",
                help="Head index (if values, keys or queries are selected)",
                disabled=not (act_type == "attention"))
        with col4:
            position = st.text_input(
                "Position(s)", value="all", key="position",
                help="Affected position(s) in sequence, comma separated list or something like 2,4-7 etc.")
            position_list = get_position_list(position)

        with col5:
            ablation_type = st.selectbox(
                "Action", ["zero", "double", "flip"], index=0, key="ablation_type",
                help="Action to perform on selected activations")
    else:
        head_idx = None
        position_list = None
        ablation_type = None

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
                hooked_generator = generate_with_hooks(model, prompt, [], **st.session_state.model_param)
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
                if hook_type == "custom hook":
                    hooks = [(hook_name, custom_hook)]
                else:
                    hook = get_ablation_hook(
                        act_type=act_type,
                        act_name=act_name,
                        head_idx=head_idx,
                        position_list=position_list,
                        ablation_type=ablation_type
                    )
                    hooks = [(hook_name, hook)]
                hooked_generator = generate_with_hooks(model, prompt, hooks, **st.session_state.model_param)
                st.session_state.output_hooked = ""
                for text in hooked_generator:
                    st.session_state.output_hooked = text
                    hooked_output.text(text)
        hooked_output.text(st.session_state.output_hooked)


init_session()
sidebar_settings()
main_window()
