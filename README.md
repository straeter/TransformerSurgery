# TransformerSurgery

A Library for perform and visualize brain surgery on transformers using mechanistic interpretability library [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) by [Bryce Meyer](https://github.com/bryce13950) and created by [Neel Nanda](https://neelnanda.io/about)

With TransformerSurgery you can ablate attentions in a Transfomer model using TransformerLens hooks. You can then generate text and compare it with the unablated model.

## Quick Start

### Install

Create a virtual environment, then do:
```shell
pip install -r requirements.txt
```

### Use

To run the interactive app, just do:

```shell
streamlit run app.py
```

## Features
- load 6 different transformer models
- compare generated text for ablated and unablated models
- ablate attentions in any layer
- ablate head, residual stream or MLP
- only apply to a fixed position
- zero out or doublle attentions
- custom hooks (coming soon)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details