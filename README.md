# MFBO Tutorial
**A Multi-Fidelity Bayesian Method for Complex Model Optimization: An Industrial Case Study**
MFBO is an efficient method for reducing uncertainty using multi fidelity and Bayesian optimization with a general overview of possible acquisition functions. It is developed for a undergraduate project at [Politecnico di Milano](https://www.polimi.it/) in a joint collaboration with [Università della Svizzera italiana](https://www.usi.ch)

### Installation
This tutorial required Anaconda or mamba being installed. After cloning the repository, set up the python environment using the command:
```sh
conda env create -f advpy.yml
```
Or fork the repository for having your own copy.

### Abstract
Computational models are often evaluated via stochastic simulation or numerical approximation. Fitting these models implies a difficult optimization problem over complex, possibly noisy parameter landscapes. The need for multiple realizations, as in uncertainty quantification or optimization, makes surrogate models an attractive option. For expensive high-fidelity models, however, even performing the number of simulations needed for fitting a surrogate may be too expensive. Inexpensive but less accurate low-fidelity data or models are often also available. Multi-fidelity combine high-fidelity and low-fidelity in order to achieve accuracy at a reasonable cost. Here we consider an hybrid method based on Multi-fidelity for implementing a Bayesian optimization algorithm. At the heart of this algorithm is maximizing the information criterion called acquisition function, and a list of the possible available choices is presented. Multi-fidelity Bayesian optimization achieves competitive performance with an affordable computational overhead for the running time of non-optimized models.

Here, it is shown the initial green uncertainty level described by the standard deviation before the MFBO method is applied:
![What is this](output_img/only_gp_wing0_PCA_resampled_same.png)

Following, it is shown the EI acquisition function outcome applied to one input (angle of the cargo-wing industrial test case):
![What is this](output_img/output_test4_alpha0.png)

Finally, it is shown the result provided again by EI acquisition function, this time applied to all the inputs:
![What is this](output_img/EI_model_pca_3000_singlobj.png)

