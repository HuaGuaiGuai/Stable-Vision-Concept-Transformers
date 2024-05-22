## Stable Vision Concept Transformers Models
 We propose the Stable Vision Concept Transformer (SVCT) framework, which uses ViT as a backbone, generates the concept layer, and fuses the concept features as well as the image features. SVCT mitigates the information leakage problem caused by CBM and maintains accuracy. We also make our model more stable by inserting a DDS plug-in. Comprehensive experiments show that SVCT can provide stable interpretations despite perturbations to the inputs, with less performance degradation than CBMs, and maintaining higher accuracy.
![framework](https://github.com/HuaGuaiGuai/Stable-Vision-Concept-Transformers/assets/115633787/d3e1e65d-17e4-4dc6-a48c-ddc36adbfa78)

## Setup
1. Install Python (3.9) and PyTorch (1.13).
2. Install dependencies by running `pip install -r requirements.txt`
3. Please clone https://github.com/openai/guided-diffusion and put it in the same directory as this repo. Note that we leverage the pre trained diffusion model of `256x256 diffusion (not class conditional): 256x256_diffusion_uncond.pt`, and please put it in the `guided_diffusion/models`.  
4. We provide pre-trained vit model pth files for the four medical datasets in the `backbone` folder for training CBMs.
5. We provide the covid dataset in the `data` folder, other datasets need to be downloaded at the time of use.


## Running the models

### 1. Creating Concept sets (optional)
A. Create initial concept set using GPT-3 - `GPT_initial_concepts.ipynb`, do this for all 3 prompt types (can be skipped if using the concept sets we have provided). NOTE: This step costs money and you will have to provide your own `openai.api_key`.

B. Process and filter the conceptset by running `GPT_conceptset_processor.ipynb` (Alternatively get ConceptNet concepts by running ConceptNet_conceptset.ipynb)

C. You need to generate as many concepts as possible, with a minimum number of concepts of 10 times the number of classification categories

### 2. Train VCT 
Train a Vision Concept Transformer Model on covid by running:  `train_vct.ipynb`

### 3. Evaluate VCT

Evaluate the trained models by running `evaluate_vct.ipynb`. This measures model accuracy, creates barplots explaining individual decisions and prints final layer weights which are the basis for creating weight visualizations.

### 4. Transform VCT into SVCT
Convert vct to SVCT and test model performance by running `experiments.ipynb`

## Results

Interpretability while maintaining a high level of accuracy:

|           Method  |HAM10000 |Covid19-CT|BloodMNIST|  OCT2017  |          
|-------------------|---------|----------|--------- |-----------|
|Standard(No interpretability)   | 99.13%  | 81.62%   | 97.05%   | 99.70%    |
| Label-free CBM (LF-CBM)        | 93.61%  | 79.75%   | 94.97%   | 97.50%    |
| Post-hoc CBM (P-CBM)           | 97.60%  | 76.26%   | 94.83%   | 98.60%    |
|Vision Concept Transformer (VCT)| 99.00%  | 80.62%   | 96.21%   | 99.10%    |
| **SVCT**                       | **99.05%**  | **81.37%**   | **96.96%**   | **99.50%**    |
| $\rho_u = 8/255$ - LF-CBM      | 90.08%  | 67.98%   | 80.53%   | 91.88%$   |
| $\rho_u = 8/255$ - P-CBM       | 90.96%  | 70.66%   | 77.55%   | 91.70%    |
| $\rho_u = 8/255$ - VCT         | 95.80%  | 69.78%   | 89.45%   | 96.80%    |
| $\rho_u = 8/255$ -**SVCT**     | 97.97%  | 74.45%   | 94.07%   | 98.70%    |
| $\rho_u = 10/255$ - LF-CBM     | 88.70%  | 65.12%   | 75.63%   | 90.58%    |
| $\rho_u = 10/255$ - P-CBM      | 90.21%  | 66.32%   | 74.27%   | 90.10%    |
| $\rho_u = 10/255$ - VCT        | 95.28%  | 68.85%   | 87.71%   | 96.25%    |
| $\rho_u = 10/255$ -**SVCT**    | 97.24%  | 71.65%   | 92.65%   | 98.48%    |

Result of cfs: 
|           Method  |HAM10000 |Covid19-CT|BloodMNIST|  OCT2017  |          
|-------------------|---------|----------|--------- |-----------|
| $\rho_u = 6/255$ -LF-CBM       | 0.3335  | 0.6022   | 0.5328   | 0.3798    |
| $\rho_u = 6/255$ -VCT          | 0.3361  | 0.6761   | 0.5432   | 0.3625    |
| $\rho_u = 6/255$ -**SVCT**     | 0.1354  | 0.5555   | 0.3589   | 0.3257    |
| $\rho_u = 8/255$ -LF-CBM       | 0.3719  | 0.6707   | 0.6280   | 0.3941    |
| $\rho_u = 8/255$ -VCT          | 0.4109  | 0.8114   | 0.7162   | 0.3812    |
| $\rho_u = 8/255$ -**SVCT**     | 0.1555  | 0.6446   | 0.4383   | 0.3459    |
| $\rho_u = 10/255$ -LF-CBM      | 0.4027  | 0.7224   | 0.6906   | 0.4055    |
| $\rho_u = 10/255$ -VCT         | 0.4637  | 0.8943   | 0.8057   | 0.3949    |
| $\rho_u = 10/255$ -**SVCT**    | 0.1725  | 0.7096   | 0.5058   | 0.3620    |

Result of cpcs:
|           Method  |HAM10000 |Covid19-CT|BloodMNIST|  OCT2017  |          
|-------------------|---------|----------|--------- |-----------|       
| $\rho_u = 6/255$ -LF-CBM       | 0.9405  | 0.8117   | 0.8511   | 0.9254    |
| $\rho_u = 6/255$ -VCT          | 0.9394  | 0.7650   | 0.8436   | 0.9314    |
| $\rho_u = 6/255$ -**SVCT**     | 0.9900  | 0.8359   | 0.9320   | 0.9468    |
| $\rho_u = 8/255$ -LF-CBM       | 0.9256  | 0.7710   | 0.7947   | 0.9196    |
| $\rho_u = 8/255$ -VCT          | 0.9098  | 0.6743   | 0.7328   | 0.9240    |
| $\rho_u = 8/255$ -**SVCT**     | 0.9867  | 0.7818   | 0.8977   | 0.9387    |
| $\rho_u = 10/255$ -LF-CBM      | 0.9123  | 0.7336   | 0.7545   | 0.9145    |
| $\rho_u = 10/255$ -VCT         | 0.8844  | 0.6155   | 0.6670   | 0.9179    |
| $\rho_u = 10/255$ -**SVCT**    | 0.9836  | 0.7389   | 0.8625   | 0.9321    |

## References

Our code implementation is based on the following awesome material:

1. https://arxiv.org/abs/2005.00928
2. https://github.com/openai/guided-diffusion
3. https://github.com/Trustworthy-ML-Lab/Label-free-CBM
