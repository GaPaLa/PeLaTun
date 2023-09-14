# Peristaltic Last Layer Tuning (PeLaTun) - Project README

## Overview

If you are memory-poor, you may consider last layer(s) finetuning,
but you may not be able to even fit the entire model in VRAM, let alone for inference or training.

PeLaTun (Peristaltic Last Layer fineTuning) is a notebook designed for finetuning the last transformer block and unembedding layer of the Llama2-7B language model,
with a context length of 4096 tokens and 16-bit precision, while using under 8GB VRAM.


This notebook notices that:

1) to train the last few layers, you only need the preceeding activations and final labels to form your x,y training dataset - to avoid loading the whole model, you can just load the preceeding layers and do inference to get these preceding activations
2) to minimize the VRAM requirement further, you can do this inference with just one layer at a time
3) If you are only loading one layer at a time, it is much faster to load one layer at a time and pass the dataset through it and repeat that for the next layer,
   than to pass a single batch through the entire model while loading one layer at a time, and repeating that for every batch - the first options uses far too much disk I/O

..and implements an appropriate training pipeline to preprocess the dataset and finetune the last layer in this fashion.


### Peristaltic Inference

To further minimize VRAM requirements during the dataset preprocessing stage, we do "peristaltic inference", where we load one transformer layer at a time, processing the embedded dataset through it, and overwrite the embeddings with the output activations/hidden states.
This process continues iteratively for each transformer block going up the model until the desired activations are obtained for training.

In this notebook, only one transformer block is loaded at a time, and batch size 1 is used for both inference and training to fit within the constraints of 8GB VRAM.
With more VRAM, it's possible to load multiple layers at once and increase batch size for faster preprocessing and/or deeper training.

### Fine-Tuning

For the fine-tuning phase, you load the last few transformer layer(s) you wish to train, along with the precomputed activations and labels. The last layers to be trained must be the ones directly after the pre-computed activations from earlier inference.
The fine-tuning process then follows the standard language modelling procedure, but with activations serving as inputs rather than tokens.

## Future Work

PeLaTun could be extended to further reduce VRAM requirements by integrating QLoRa. The moonshot goal is to enable fine-tuning of the last few transformer blocks of a 70B language model with 8GB VRAM.

Finding a way to extend this to finetuning the whole model by taking inspiration from gradient checkpointing seems possible, but that would take up even more disk space (for gradients) and require each layer to be both trained and to have inference re-run on that layer to get the relevant gradients.
Since this is just a fun project to test an idea, this is left out.

## Issues

The primary challenge of PeLaTun is related to I/O bandwidth and storage space. Peristaltic inference involves continuously (over)writing hidden states to disk.
Writing activations to disk is significantly slower than passing it through a transformer block, resulting in waiting times for write operations to complete before starting the next batch (otherwise, you risk a vuffer overflow of cached activations).
As a result, dataset peristalsis is a very time-consuming process.

Storing these activations also uses a substantial amount of storage space; 300 text samples, each with approximately 1000 tokens, with an embeddings size of 4096, at fp16 takes up ~3GB.

Despite optimizations (e.g. threading writes/inference for faster inference, removing activations which are to be padded anyway to reduce stoarge requirements, overwriting the previous layer's activations rather than storing all layers' activations), these problems remain large obstacles for practicality.

## Advantages

The primary advantage of PeLaTun is its ability to make the fine-tuning large language models (at least last layer finetuning) feasible on hardware with very limited resources. 
Preprocessing the dataset as PeLaTun does also speeds up the final layer finetuning phase, in case you need to do multiple training runs after the initial dataset peristalsis, as activations are only passed through the last layer, rather than all preceding layers.
