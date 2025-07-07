# Interpretable Latent Representations of Neural Activity via Sparse Autoencoders
Project repository for: Interpretable Latent Representations of Neural Activity via Sparse Autoencoders

Detailed written report available in PDF format in this repo.

This project explores post-hoc interpretability techniques for neural embeddings derived from population recordings of hippocampal activity. We apply sparse autoencoders (SAEs) to embeddings generated using [CEBRA](https://github.com/AdaptiveMotorControlLab/cebra), and evaluate how sparsification affects attribution to neural and behavioral variables without sacrificing decoding performance.

We find that sparsified representations:
- Activate fewer latent dimensions per sample,
- Yield more concentrated correlations with individual neurons and behaviors,
- Preserve or improve position and direction decoding accuracy.


### Motivation
Latent variable models like CEBRA yield powerful embeddings of high-dimensional neural activity, but they are often hard to interpret. This project investigates whether sparsity-enforcing autoencoders can recover latent structure that is more attributable to observable neural and behavioral factors.


## Methods

- **Embedding generation**: We use CEBRA in supervised mode (labels = position + direction) to create latent embeddings with varying dimensionality (8, 16, 32, 64).
- **Sparsification**: We train SAEs (standard, Top-K, JumpReLU) across a hyperparameter grid to transform dense embeddings into sparse codes.
- **Evaluation**:
  - **Behavioral attribution**: Correlation between latent dimensions and position/direction.
  - **Neuron attribution**: Correlation between latents and raw neural channels.
  - **Decoding**: Linear decoding of position ($R^2$) and direction (accuracy).
  - **Sparsity**: Percent of active latent dimensions per sample.

## Key Results

- SAE latents show improved neuron and behavior attribution.
- Decoding accuracy is preserved across most configurations.
- Higher Top-K and lower embedding size improves behavioral interpretability; larger embeddings improve neural interpretability.

More visualizations and interpretation are included in the [report](./interpretable_neural_embeddings.pdf).


## Contact

If you're interested in collaborating, using this framework, or adapting it to other domains, feel free to reach out:

**Arjun Naik**  
arjunsn@uw.edu  


