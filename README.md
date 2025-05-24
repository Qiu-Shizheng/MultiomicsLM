# MultiomicsLM: A foundational model for characterizing the multi-omics landscape of the general population

This package provides an ensemble inference module for multiple disease models based on multiomics BERT. It enables you to load pre‐finetuned models for several diseases and, given a new multiomics sample (protein and metabolite data), outputs the predicted probability for each disease.

## Supported Diseases

The following disease codes and names are supported:

- dementia → Dementia  
- parkinson → Parkinson’s disease  
- copd → COPD  
- asthma → Asthma  
- RA → Rheumatoid arthritis  
- obesity → Obesity  
- diabetes → T2D  
- gout → Gout  
- hypertension → Hypertension  
- heart_failure → Heart failure  
- ischaemic_heart_disease → Ischaemic heart disease  
- atrial_fibrillation → Atrial fibrillation  
- stroke → Stroke  
- cerebral_infarction → Cerebral infarction  
- Breast_cancer → Breast cancer  
- Colorectal_cancer → Colon cancer  
- Lung_cancer → Lung cancer  
- Prostate_cancer → Prostate cancer  
- skin_cancer → Skin cancer  
- glaucoma → Glaucoma  


## System requirements
torch 2.4.1+cu124  
tqdm 4.66.4  
scikit-learn 1.4.2  
scipy 1.13.1  
seaborn 0.12.2  
python 3.11.9  
pytorch-cuda 12.4    
optuna 3.6.1     
numpy 1.26.4
matplotlib 3.8.4 

## Installation

Clone the repository and install with pip:

```bash
git clone https://github.com/yourusername/multiomics_ensemble.git
cd multiomics_ensemble
pip install -e .

![](https://github.com/Qiu-Shizheng/MultiomicsLM/blob/main/Figure/Figure%201.jpeg)


