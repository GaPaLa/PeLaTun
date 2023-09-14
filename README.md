# Peristaltic Last Layer Tuning (PeLaTun) - Project README

## Overview

PeLaTun is a notebook designed for the peristaltic tuning of the last transformer block(s) and unembedding layer of the Llama2-7B language model.
This process is conducted with a context length of 4096 tokens and utilizes 16-bit precision, while using under 8GB VRAM.

### Intuition

I use the term "peristaltic", as we are passing the entire dataset progressively through the model, one transformer block at a time.
This approach contrasts with the conventional practice of processing small sections of the dataset through all layers simultaneously.
The trade-off involves reduced VRAM requirements in exchange for significantly slower overall training time (the data proeprocessing step is very slow due to I/O, the actual finetuning of the last layer is fast) and higher storage demands.

The core idea is that since it is known that you can achieve decent results by fine-tuning only the last few layers of a language model, including the lm_head, normalization layer, and the last transformer block,
this allows for a reduction in VRAM requirements if we do this by pre-processing the dataset through the lower layers of the transformer, loading only one layer at a time.
That results in a dataset comprising activations just before the final layers to be trained, along with the ground truth tokens for prediction, so just the last few layers can be trained with standard language modelling, just taking input activations instead of tokens.

### Peristaltic Inference

To further minimize VRAM requirements during the dataset preprocessing stage, we do "peristaltic inference", where we load one transformer layer at a time, processing the embedded dataset through it, and overwrite the embeddings with the output activations/hidden states.
This process continues iteratively for each transformer block going up the model until the desired activations are obtained for training.

In this notebook, only one transformer block is loaded at a time, and batch size 1 is used for both inference and training to fit within the constraints of 8GB VRAM.
With more VRAM, it's possible to load multiple layers at once and increase batch size for faster preprocessing and/or deeper training.

### Fine-Tuning

For the fine-tuning phase, you load the last few transformer layer(s) you wish to train, along with the precomputed activations and labels.
The fine-tuning process then follows the standard language modelling procedure, with activations serving as inputs rather than tokens.

## Future Work

PeLaTun could be extended to further reduce VRAM requirements by integrating with QLoRa. The goal is to enable fine-tuning of the last few layers of a 70B language model on limited 8GB VRAM, potentially employing low-VRAM optimizers.

Finding a way to extend this to finetuning the whole model by mixing in gradient checkpointing with selective layer loading seems possible but that prcess would take up double the disk space and require each layer to be both train and to have niference re-run on that layer.
Since this is just a fun project to test an idea and practice a little, this is left out.

## Issues

The primary challenge of PeLaTun is related to I/O bandwidth and storage space. Peristaltic inference involves continuously writing and overwriting hidden states on disk.
Writing to disk is significantly slower than processing through a transformer block, resulting in waiting times for write operations to complete before starting the next batch.
Otherwise, you risk overloading VRAM/RAM, leading to system crashes.
As a result, dataset peristalsis is a time-consuming process.

Additionally, PeLaTun consumes a substantial amount of storage space due to the need to store activations. For instance, processing 300 samples, each with approximately 1000 tokens, with an embeddings size of 4096 takes up ~3GB.
Despite optimizations, such as excluding activations to be padded or storing only the most recent layer's activations, significant storage space is still required.

## Advantages

The primary advantage of PeLaTun is its ability to make the fine-tuning large language models (at least last layer finetuning) feasible on hardware with very limited resources. 
Preprocessing the dataset as PeLaTun does also speeds up the final layer finetuning phase, as activations are only passed through the last layer, rather than all preceding layers.
This results in faster iterations if doing multiple training runs after the initial dataset peristalsis.
