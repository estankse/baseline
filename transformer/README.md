# Standard Transformer Architecture
Unlike contemporary decoder-only LLMs, the proposed translation model utilizes a standard encoder-decoder Transformer architecture. This approach maintains the cross-attention mechanism between the encoder's contextual representations and the decoder's autoregressive generation process, consistent with the foundational Vaswani et al. (2017) design.


To account for hardware constraints and optimize GPU utilization, a subset of 200,000 samples from the OPUS-100 dataset was selected for training. These parameters can be adjusted accordingly within the data.py

