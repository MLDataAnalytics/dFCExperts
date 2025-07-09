![DOI](https://img.shields.io/badge/DOI-10.1101%2F2024.12.20.629773-blue?style=for-the-badge&logo=biorxiv)
# 🧠 dFCExpert: Learning Brain Dynamic Functional Connectivity with Modularity and State Experts

This repository contains the official implementation of "[dFCExperts: Learning Dynamic Functional Connectivity Patterns with Modularity and State Experts](https://doi.org/10.1101/2024.12.20.629773)."

**dFCExpert** is a novel deep learning framework designed to robustly characterize **dynamic functional connectivity (dFC)** patterns from functional Magnetic Resonance Imaging (fMRI) data. Understanding these patterns is crucial for advancing research in imaging neuroscience and medicine.

While recent **Graph Neural Network (GNN)** approaches combined with **transformers** or **recurrent neural networks (RNNs)** have shown promise, they often struggle to capture the **modular organization** of brain networks and the diversity of **dFC states**. **dFCExpert** addresses these challenges through two key innovations:

---

## 🔹 Modularity Experts

We introduce a **mixture of experts (MoE)** mechanism within a **GNN** framework. Each expert specializes in learning features from brain nodes within the same functional module, significantly enhancing the model's ability to capture the brain's modular structure.

---

## 🔹 State Experts

To model temporal dynamics, we employ a **soft prototype clustering** approach. This method aggregates dFC features into a set of interpretable connectivity states, providing insights into how brain functions vary across time and conditions.

---

## 🚀 Key Features

* **Modular GNN Architecture:** Utilizes expert specialization for enhanced feature learning.
* **Soft Clustering:** Enables discovery of dynamic brain states.
* **Enhanced Interpretability:** Provides clearer insights into learned dFC representations.
* **Validated Performance:** Proven effective on three large-scale fMRI datasets.
* **Diverse Applications:** Applicable to developmental neuroscience, sex differences, and Autism Spectrum Disorder (ASD) research.

---

## 🖼️ Model Overview

![The deep learning model](https://github.com/MLDataAnalytics/dFCExperts/blob/main/Figures/model.png)

**dFCExpert** consists of **modularity experts** and **state experts**. Taking the dFC graphs as input, the modularity experts leverage a combination of **GIN** and **MoE** to learn brain graph features for each FC segment, aiming to capture the brain modularity mechanism effectively.

Building upon the outputs of the **modularity experts**, the state experts adaptively group the temporal graph features into distinctive states using a **soft prototype clustering** method. This approach allows the model to learn expressive state features by assigning soft clusters to the temporal graph features.

Finally, the learned state features are passed through an **MLP** layer to predict a specific task, such as predicting sex in a classification setting or predicting an intelligence measure in a regression setting.

Formally, the objective of **dFCExpert** is to train a neural network: $f: (G_1, ..., G_T) -> h_{state}$, where $G_t = \{A_t, X_t\}$ for $t = 1, ..., T$ represents the sequence of constructed **dFC** graphs with $T$ segments, and $h_{state} ∈ R^{(K×D)}$ is the output feature matrix from the **state experts**.

We formulate $f = s ∘ m$ as a composition of **modularity experts** $m$ that learn dFC graph representations and $h_{modularity} = (h_{G1}, ..., h_{GT})$ and **state experts** $s$ that aggregate these learned temporal graph features into the final state features $h_{state}$.

---

## 📦 Installation

To get started, clone the repository and install the required dependencies:

```bash```
```
git clone https://github.com/MLDataAnalytics/dFCExperts.git
cd dFCExperts
pip install -r requirements.txt
```

## 📁 Dataset
The fMRI data used in this paper can be downloaded from the Human Connectome Project and ABCD_ABCC.

### Example Data Folder Structure
Your data folder (specified by the --sourcedir option) should follow this structure:

```
data
├─── hcp1200
│    ├─── label.csv
│    ├─── hcp_rest_datasplit_5folds.pth
│    ├─── hcp_rfMRI_REST1_LR_fc_Schaefer2018_400Parcels.pt
│    └─── hcp_rfMRI_REST1_LR_tc_Schaefer2018_400Parcels.pt
└─── abcd_abcc
     ├─── 6195_timeseries-?x352.pth
     ├─── datasplit_5folds.pth
     └─── label.csv
```

## 🏋️ Training
To train the model as described in the paper, run the following command:

```Bash```
```
python3 main.py --exp_name 'hcp_c' \
                --dataset 'hcp-dyn' \
                --targetdir './result' \
                --target_feature 'Gender' \
                --gin_type 'moe_gin' \
                --num_gin_experts 5 \
                --num_states 7 \
                --state_ex_loss_coeff 10 \
                --orthogonal \
                --freeze_center \
                --project_assignment \
                --fc_hidden 256 \
                --num_epochs 30 \
                --minibatch_size 8 \
                --train \
                --validate \
                --test \
                --test_model_name 'model_val_acc'
```

## 🛠️  Usage Examples

### Train the Model

```Python```
```
from dfcexpert import DFCEngine

# Load your fMRI data
data = load_fmri_data('path/to/data')

# Initialize and train the model
model = DFCEngine(config='config.yaml')
model.train(data)
```

### Extract dFC Representations

```Python```
```
# Extract learned features
representations = model.extract_features(data)
```

### Visualize Connectivity States

```Python```

```
model.visualize_states(output_dir='results/visualizations')
```

## 📄 Citation
If you use this work in your research, please cite our paper:

```
@article{Chen2024.12.20.629773,
	author = {Chen, Tingting and Li, Hongming and Zheng, Hao and Fan, Yong},
	title = {dFCExpert: Learning Dynamic Functional Connectivity Patterns with Modularity and State Experts},
	elocation-id = {2024.12.20.629773},
	year = {2024},
	doi = {10.1101/2024.12.20.629773},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Characterizing brain dynamic functional connectivity (dFC) patterns from functional Magnetic Resonance Imaging (fMRI) data is of paramount importance in imaging neuroscience and medicine. Recently, many graph neural network (GNN) models, combined with transformers or recurrent neural networks (RNNs), have shown great potential for modeling the dFC patterns. However, these methods face challenges in effectively characterizing the modularity organization of brain networks and capturing varying dFC state patterns. To address these limitations, we propose dFCExpert, a novel method designed to learn robust representations of dFC patterns in fMRI data with modularity experts and state experts. Specifically, the modularity experts optimize multiple experts to characterize the brain modularity organization during graph feature learning process by combining GNN and mixture of experts (MoE), with each expert focusing on brain nodes within the same functional network module. The state experts aggregate temporal dFC features into a set of distinctive connectivity states using a soft prototype clustering method, providing insight into how these states support diverse brain functions and how they vary across brain conditions. Experiments on three large-scale fMRI datasets have demonstrated the superiority of our method over existing alternatives. The learned dFC representations not only enhance interpretability but also hold promise for advancing our understanding of brain function across a range of conditions, including development, sex difference, and Autism Spectrum Disorder. Our code is publicly available at \href{https://github.com/MLDataAnalytics/dFCExperts}{MLDataAnalytics/dFCExpert}.},
	URL = {[https://www.biorxiv.org/content/early/2024/12/21/2024.12.20.629773](https://www.biorxiv.org/content/early/2024/12/21/2024.12.20.629773)},
	eprint = {[https://www.biorxiv.org/content/early/2024/12/21/2024.12.20.629773.full.pdf](https://www.biorxiv.org/content/early/2024/12/21/2024.12.20.629773.full.pdf)},
	journal = {bioRxiv}
}
``` 

---

## 🤝 Contributing
We welcome contributions! If you have suggestions for improvements, bug reports, or would like to contribute code, please feel free to open an issue or submit a pull request.

---

## 📬 Contact

For any questions or support, please open an issue in this repository.

---

