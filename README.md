# Faithful Vision Concept Transformers Models
 We propose the Faithful Vision Concept Transformer (FVCT) framework, which uses ViT as a backbone, generates the concept layer, and fuses the concept features as well as the image features. FVCT mitigates the information leakage problem caused by CBM and maintains accuracy. We also make our model more stable by inserting a DDS plug-in. Comprehensive experiments show that FVCT can provide stable interpretations despite perturbations to the inputs, with less performance degradation than CBMs, and maintaining higher accuracy.
![framework](https://github.com/HuaGuaiGuai/Faithful-Vision-Concept-Transformers/assets/115633787/e4409afe-47cb-45f3-890a-7dc256e7655e)


## Setup
1. Install Python (3.9) and PyTorch (1.13).
2. Install dependencies by running `pip install -r requirements.txt`
3. Please clone https://github.com/openai/guided-diffusion and put it in the same directory as this repo. Note that we leverage the pre trained diffusion model of `256x256 diffusion (not class conditional): 256x256_diffusion_uncond.pt`. 
4. We provide pre-trained vit model pth files for the four medical datasets in the `backbone` folder for training CBMs.
5. We provide the ham10000 and covid datasets in the `data` folder, the oct dataset needs to be downloaded at the time of use.


## Running the models

### 1. Creating Concept sets (optional)
A. Create initial concept set using GPT-3 - `GPT_initial_concepts.ipynb`, do this for all 3 prompt types (can be skipped if using the concept sets we have provided). NOTE: This step costs money and you will have to provide your own `openai.api_key`.

B. Process and filter the conceptset by running `GPT_conceptset_processor.ipynb` (Alternatively get ConceptNet concepts by running ConceptNet_conceptset.ipynb)

C. You need to generate as many concepts as possible, with a minimum number of concepts of 10 times the number of classification categories

### 2. Train VCT 
Train a Vision Concept Transformer Model on covid by running:

`sadasddddddddddddddddddddddddddddddddddddddddddddd`

### 3. Transform VCT into FVCT



### 4. Evaluate FVCT

Evaluate the trained models by running `evaluate_vct.ipynb`. This measures model accuracy, creates barplots explaining individual decisions and prints final layer weights which are the basis for creating weight visualizations.


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
