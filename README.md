# 👕👗 Autoencoder-Based Embedding Compression for E-commerce Search

## 🎯 Project Overview

This project explores the use of **Autoencoders (AE)** for **Dimensionality Reduction** (compression) of high-dimensional product embeddings. The goal is to significantly reduce the size of vector embeddings originally derived from a powerful Vision-Language model (like CLIP), while preserving their **semantic richness** for downstream tasks, specifically **retrieval and similarity search**.

This optimization is crucial for:
1.  **Reducing Storage Costs:** Storing millions of smaller vectors is cheaper.
2.  **Improving Latency:** Faster vector search lookups in production systems.
3.  **Maintaining Performance:** Ensuring the compressed embeddings maintain high accuracy in retrieval tasks.

## 🛠️ Methodology & Architecture

### 1. Data Source

* **Source:** Product embeddings sourced from a e-commerce catalog.
* **Initial Dimensionality:** The original embeddings are 372-dimensional.
* **Dataset:** A sample of 10,000 product embeddings was used for training and evaluation.

### 2. Autoencoder Design

A custom fully-connected Autoencoder (`MyAutoEncoder` and `DenoisingAutoencoder`) was built using PyTorch (`torch.nn.Module`).

* **Architecture:** A six-layer architecture was implemented with decreasing dimensions in the encoder and symmetrical increasing dimensions in the decoder.
    * **Input Size:** 372
    * **Encoder Layers:** `372 -> F_C -> S_C -> T_C` (Latent Dimension / Bottleneck)
    * **Decoder Layers:** `T_C -> S_C -> F_C -> 372` (Reconstructed Output)
* **Loss Function:** Mean Squared Error (`nn.MSELoss`) was used to minimize the reconstruction error between the original and the decoded embedding.
* **Evaluation Metric:** Retrieval performance was measured using a **K-Nearest Neighbors (KNN) Self-Retrieval Test** (checking if the reconstructed vector retrieves the original input).

### 3. Compression Experimentation (Standard AE)

The core experiment tested various latent dimensions (`T_C`) to find the optimal balance between compression ratio and retrieval accuracy:

| Latent Dimension | Compression Ratio (from 372) | Final Validation MSE | Self-Retrieval Success Rate | Avg. Cosine Similarity |
| :--------------: | :--------------------------: | :------------------: | :-------------------------: | :--------------------: |
| **200** | 46% Reduction                | 0.007890             | **98.55%** | 0.9509                 |
| **156** | 58% Reduction                | 0.008249             | **98.80%** | 0.9488                 |
| **128** | 66% Reduction                | 0.009424             | **98.30%** | 0.9414                 |
| **80** | 78% Reduction                | 0.011186             | **96.10%** | 0.9300                 |
| **64** | 83% Reduction                | 0.011872             | **94.30%** | 0.9256                 |
| **32** | 91% Reduction                | 0.012792             | **92.85%** | 0.9196                 |

### Conclusion

The results show that a **58% compression (to 156 dimensions)** maintains a retrieval success rate of **98.80%**, demonstrating that significant infrastructure optimization can be achieved with minimal loss of semantic integrity.

---
