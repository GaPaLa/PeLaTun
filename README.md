# "Peristaltic" Last Layer Tuning (PeLaTun?)

## Overview

For very low resource fine tuning, you may consider last layer(s) finetuning, but you may not be able to even fit the model in VRAM, let alone for inference or training.

PeLaTun is a notebook designed for finetuning the last transformer block and unembedding layer of the Llama2-7B language model,
with a context length of 4096 tokens and 16-bit precision, while using under 8GB VRAM.


This notebook notices that:

1) to train the last few layers, you only need the preceeding activations to the layers you are going to train and the final labels in order to form your x,y training dataset - to avoid loading the whole model, you can just load the preceeding layers and do inference on the dataset to get these preceding activations
2) To minimize the VRAM requirement further, you can do this inference with just one layer at a time (it would be more efficient to do this in blocks of N layers, maximising N to how many you layers you can fit in VRAM with enough remaining for inference. For simplicity, this notebook just loads one transformer block at a time.)
3) It is much faster to load layer N, pass the entire dataset through it in batches, unload N, pass those activations forward, and repeat that for the next layer(s), than it is to load layer N, pass a single batch through it, unload N, then load layer N+1, pass the single batch through it, and repeat that for all batches in the dataset. The first options uses much less much disk I/O since weights are much larger than activations. I'll call this peristaltic inference, since the whole dataset goes through the model section by section. (This means last layer finetuning on low VRAM is more feasible then inference, due to inference requiring low latency)

I implements an appropriate pipeline to preprocess the dataset and finetune the last layer in this fashion.


### Peristaltic Inference, step-by-step

1 - First, we take in the tokenized dataset, pass each batch through the embedding layer, then save those embeddings to disk.

2 - We initialize a single transformer block model with the correct params for Llama-2.

3 - Next, we read the state_dict for Llama-2 7B and load the layers for the first transformer block. We overwrite the weights from the transformer block we initialized with these loaded weights.

4 - One batch at a time, we pass the entire dataset through the transformer layer, saving activation to disk (I/O is very slow and it takes up tonsa LOT of disk space. We use threading to get saving/inference done as simultaneously as possible but inference is largely waiting on saving). To reduce the space the activation take up, we remove embeddings which correspond to padded tokens, and we only keep the activations for the currrent layer we are inferring with, so the sample_0 layer_5 activations file immediately overwrites sample_0 layer_4 file.

5 - Once the entire dataset has been passed through this layer, move up a layer. Repeat the last two steps until we reach the layer just before the layers we want to train, so we have the correct input activations to feed them.

In this notebook only one transformer block is loaded at a time and batch size 1 is used for both inference and training to fit within the 8GB VRAM. 
With more VRAM, it's possible to load multiple layers at once during inference and/or training and increase batch size for faster dataset preprocessing and/or training more layers.
To maximise throughput, the batch sizes for inference through the embedding layer; through a transformer block; and for training should be independent. In this code, for simplicity, we set it to be the same for the first two.

### Fine-Tuning

For the fine-tuning phase, you load the last transformer block(s) you wish to train, and initialize a Dataset of the activations from the previous layer, and another Dataset for the tokenized labels (be sure these two datasets' samples are matched correctly as they produce batches - I just do this using shuffle=False :p). The fine-tuning process then follows the standard language modelling procedure, but feeding _activations_ as inputs rather than tokens.

## Future Work

PeLaTun could be extended to further reduce VRAM requirements by integrating QLoRa. The moonshot goal is to enable fine-tuning of the last few transformer blocks of a 70B language model with 8GB VRAM - compared to 7B, each 70B layer has double the hidden activation size, but 4-bit QLoRA should mean it fits in 8GB, and while 70B also has more layers, the number of layers doesn't make a difference to whether it fits in VRAM when using PeLaTun.

It seems infeasible to train the whole model persitalticly - we need all model weights to be updated for each training sample - we can't just train the last layer then train the previous layer. 
Even if we did that, it would also require storing the activations for all samples at all layers in order to calculate the gradients for them, which is just an infeasible amount of storage (multiple TB for a dataset of a few 100 samples).
We could reduce the space used with gradient checkpointing https://github.com/cybertronai/gradient-checkpointing but it would still be very impractical, even slower, and still take up a lot of space.

## Advantages

The primary advantage of PeLaTun is its ability to make the fine-tuning the last layer of large language models feasible on consumer hardware. 
Preprocessing the dataset as PeLaTun does also speeds up the final layer finetuning phase, in case you do multiple training runs after the initial dataset peristalsis, as activations are only passed through the last layer, rather than all preceding layers.
