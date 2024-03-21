# Faithful Vision Concept Transformers Models
 We propose the Faithful Vision Concept Transformer (FVCT) framework, which uses ViT as a backbone, generates the concept layer, and fuses the concept features as well as the image features. FVCT mitigates the information leakage problem caused by CBM and maintains accuracy. We also make our model more stable by inserting a DDS plug-in. Comprehensive experiments show that FVCT can provide stable interpretations despite perturbations to the inputs, with less performance degradation than CBMs, and maintaining higher accuracy.
![framework](https://github.com/HuaGuaiGuai/Faithful-Vision-Concept-Transformers/assets/115633787/e4409afe-47cb-45f3-890a-7dc256e7655e)


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

### 3. Evaluate FVCT

Evaluate the trained models by running `evaluate_vct.ipynb`. This measures model accuracy, creates barplots explaining individual decisions and prints final layer weights which are the basis for creating weight visualizations.

### 4. Transform VCT into FVCT
Convert vct to FVCT and test model performance by running `experiments.ipynb`

## Results

Interpretability while maintaining a high level of accuracy:

|           Method  |HAM10000 |Covid19-CT|BloodMNIST|  OCT2017  |          
|-------------------|---------|----------|--------- |-----------|
|Standard(No interpretability)   | 99.13%  | 81.62%   | 97.05%   | 99.70%    |
| Label-free CBM                 | 93.61%  | 79.75%   | 94.97%   | 97.50%    |
| **FVCT**                       | **99.05%**  | **81.37%**   | **96.96%**   | **99.50%**    |
| $\rho_u = 6/255$ -base         | 91.98%  | 73.09%   | 86.85%   | 93.22%    |
| $\rho_u = 6/255$ -**FVCT**     | 98.45%  | 76.32%   | 98.32%   | 98.90%    |
| $\rho_u = 7/255$ -base         | 90.88%  | 69.74%   | 83.56%   | 92.40%    |
| $\rho_u = 7/255$ -**FVCT**     | 98.28%  | 74.77%   | 94.70%   | 98.85%    |
| $\rho_u = 8/255$ -base         | 90.08%  | 67.98%   | 80.53%   | 91.88%    |
| $\rho_u = 8/255$ -**FVCT**     | 97.97%  | 74.45%   | 94.07%   | 98.70%    |
| $\rho_u = 9/255$ -base         | 89.45%  | 66.30%   | 77.95%   | 91.16%    |
| $\rho_u = 9/255$ -**FVCT**     | 97.60%  | 72.27%   | 93.34%   | 98.60%    |
| $\rho_u = 10/255$ -base        | 99.70%  | 65.12%   | 75.63%   | 90.58%    |
| $\rho_u = 10/255$ -**FVCT**    | 97.24%  | 71.65%   | 92.65%   | 98.48%    |

result of cfs: 
|           Method  |HAM10000 |Covid19-CT|BloodMNIST|  OCT2017  |          
|-------------------|---------|----------|--------- |-----------|
| $\rho_u = 6/255$ -base         | 0.3335  | 0.6022   | 0.5328   | 0.3798    |
| $\rho_u = 6/255$ -**FVCT**     | 0.1354  | 0.5555   | 0.3589   | 0.3257    |
| $\rho_u = 7/255$ -base         | 0.3577  | 0.6403   | 0.5864   | 0.3871    |
| $\rho_u = 7/255$ -**FVCT**     | 0.1462  | 0.6052   | 0.3995   | 0.3370    |
| $\rho_u = 8/255$ -base         | 0.3719  | 0.6707   | 0.6280   | 0.3941    |
| $\rho_u = 8/255$ -**FVCT**     | 0.1555  | 0.6446   | 0.4383   | 0.3459    |
| $\rho_u = 9/255$ -base         | 0.3881  | 0.6985   | 0.6623   | 0.4001    |
| $\rho_u = 9/255$ -**FVCT**     | 0.1641  | 0.6798   | 0.4733   | 0.3542    |
| $\rho_u = 10/255$ -base        | 0.4027  | 0.7224   | 0.6906   | 0.4055    |
| $\rho_u = 10/255$ -**FVCT**    | 0.1725  | 0.7096   | 0.5058   | 0.3620    |

result of cpcs:
|           Method  |HAM10000 |Covid19-CT|BloodMNIST|  OCT2017  |          
|-------------------|---------|----------|--------- |-----------|
|           Method  |HAM10000 |Covid19-CT|BloodMNIST|  OCT2017  |          
|-------------------|---------|----------|--------- |-----------|
| $\rho_u = 6/255$ -base         | 0.9405  | 0.8117   | 0.8511   | 0.9254    |
| $\rho_u = 6/255$ -**FVCT**     | 0.9900  | 0.8359   | 0.9320   | 0.9468    |
| $\rho_u = 7/255$ -base         | 0.9329  | 0.7900   | 0.8203   | 0.9225    |
| $\rho_u = 7/255$ -**FVCT**     | 0.9883  | 0.8061   | 0.9154   | 0.9422    |
| $\rho_u = 8/255$ -base         | 0.9256  | 0.7710   | 0.7947   | 0.9196    |
| $\rho_u = 8/255$ -**FVCT**     | 0.9867  | 0.7818   | 0.8977   | 0.9387    |
| $\rho_u = 9/255$ -base         | 0.9188  | 0.7528   | 0.7730   | 0.9171    |
| $\rho_u = 9/255$ -**FVCT**     | 0.9852  | 0.7587   | 0.8801   | 0.9354    |
| $\rho_u = 10/255$ -base        | 0.9123  | 0.7336   | 0.7545   | 0.9145    |
| $\rho_u = 10/255$ -**FVCT**    | 0.9836  | 0.7389   | 0.8625   | 0.9321    |
