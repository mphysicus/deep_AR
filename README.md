# DeepAR
﻿<p align="center">
  <img src="figures/Deep AR logo.png" alt="DeepAR Logo" width="300"/>
</p>

---

> **⚠️ Note**: This is an ongoing project under active development. Features, documentation and code may change.

> **Note**: This model was used to perform analysis on the CMIP6 dataset. The code for that can be found in this github repository: [mphysicus/atmospheric-rivers-cmip6](https://github.com/mphysicus/atmospheric-rivers-cmip6)
---
DeepAR is a deep learning model designed for Atmospheric Rivers (AR) detection and segmentation from climate data (using the Climate variable IVT, IVT_u, IVT_v). It utilizes a modified, prompt-less Segment Anything Model (SAM) to generate AR masks.

## Model Architecture
The DeepAR model processes data through a three-stage pipeline:
1. **Input Generator (`IVT2RGB`)**: A CNN that converts 3 channel climate data (Integrated Vapor Transport: `ivt`, `ivtu`, `ivtv`) into a 3 channel RGB-like image suitable for the image encoder.
2. **Segmentation (`SamAR`)**: A modified [SAM model](https://github.com/facebookresearch/segment-anything) that operates without prompts. It uses a learned `no_mask_embedding` (This replaces the prompt encoder of the original SAM model) to generate segmentation masks from the features produced by the image encoder.

The diagram below illustrates the architecture:
﻿<p align="center">
  <img src="figures/model_architecture.png" alt="DeepAR Architecture" width="800"/>
</p>
Below is the architecture of the `IVT2RGB` module:
﻿<p align="center">
  <img src="figures/IVT2RGB.png" alt="IVT2RGB architecture" width="800"/>
</p>


## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/mphysicus/deep_AR.git
    cd deep_AR
    ```
2. Install the package:
    ```bash
    pip install -e .
    ```
## Usage
The model is designed to work with NetCDF files (.nc) containing `ivt`, `ivtu` and `ivtv` variables. Use the dataset class `ARInferenceDataset` for loading and preprocessing the data during inference.

For a quick preview of how to use the model for inference, please refer to this notebook file in the repository: [demo.ipynb](https://github.com/mphysicus/deep_AR/blob/main/notebook/demo.ipynb)

## Pretrained Models
✨Coming Soon✨We will be uploading pre-trained model weights on Huggingface soon.













## 🙏 Acknowledgment
We are thankful to [Segment Anything](https://github.com/facebookresearch/segment-anything) for releasing their code as open-source contributions.
