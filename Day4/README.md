# This repository contains an implementation of the Vision Transformer (ViT) model using PyTorch. The Vision Transformer is a state-of-the-art model for image classification, utilizing transformer-based architectures that have achieved great success in natural language processing and have now been applied to computer vision tasks.

## Overview
The ViT model works by dividing an image into small patches, flattening them, and then feeding them into a transformer architecture. This allows the model to learn contextual relationships between different patches of an image.

## Key Components:
Patch Embedding: The image is split into smaller patches, each flattened into a vector and then linearly embedded into a fixed-dimensional space.
Transformer: A multi-layer transformer encoder is used to process the sequence of embedded patches.
Classification Head: The final representation of the image is processed to predict the class label using a simple fully connected layer.
Structure of the Code
The code is organized into the following main components:

1. FeedForward Class
Implements a simple feed-forward neural network, which is used in the transformer encoder. It includes layer normalization, linear layers, GELU activation, and dropout.
2. Attention Class
Implements multi-head self-attention, which enables the model to focus on different parts of the image simultaneously. It uses layer normalization, linear transformations for the query, key, and value vectors, followed by the attention mechanism.
3. Transformer Class
A transformer architecture that stacks several layers of attention and feed-forward networks to process the input data. Each layer consists of an attention block followed by a feed-forward block.
4. ViT Class
The main Vision Transformer model that ties everything together. It embeds the image patches, adds positional encoding, applies the transformer model, and uses the output to predict the class label using a fully connected layer.
Requirements
PyTorch
einops