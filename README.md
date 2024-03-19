# Faithful Vision Concept Transformers Models

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

|           Method  |HAM10000 |Covid19-ct|BloodMNIST|  OCT2017  |          
|-------------------|---------|----------|--------- |-----------|
|Standard(No interpretability)| CIFAR10 | CIFAR100 | CUB200   | Places365 |
| Label-free CBM              | 88.80%  | 70.10%   | 76.70%   | 48.56%    |
|-------------------|---------|----------|--------- |-----------|
| FVCT                       | 82.96%  | 58.34%   | **75.96%**  | 38.46%    |
| Label-free CBM              | **86.37%** | **65.27%**   | 74.59%  | **43.71%**   |
| Label-free CBM              | **86.37%** | **65.27%**   | 74.59%  | **43.71%**   |
