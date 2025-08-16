# BConformeR: A Conformer Based on Mutual Sampling for Unified Prediction of Continuous and Discontinuous Antibody Binding Sites

Accurate prediction of antibody-binding sites (epitopes) on antigens is crucial for vaccine design, immunodiagnostics, therapeutic antibody development, antibody engineering, research into autoimmune and allergic diseases, and for advancing our understanding of immune responses. Despite in silico methods that have been proposed to predict both linear (continuous) and conformational (discontinuous) epitopes, they consistently underperform in predicting conformational epitopes. In this work, we propose a conformer-based model trained on antigen sequences derived from 1,080 antigen-antibody complexes, leveraging convolutional neural networks (CNNs) to extract local features and Transformers to capture long-range dependencies within antigen sequences.

![Figure 1](./fig/overview.png)

## Installation

1. Clone this repository:

```bash
git clone https://github.com/Jerry-alt19/BConformeR.git
cd BConformeR

2. Install dependencies:

```bash
conda create -n bconformer python=3.9
conda activate bconformer
pip install -r requirements.txt
