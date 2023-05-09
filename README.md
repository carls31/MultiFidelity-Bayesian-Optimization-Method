# A Multi-Fidelity Bayesian Method for Complex Model Optimization: An Industrial Case Study
Computational models are often evaluated via stochastic simulation or numerical approximation. Fitting these models implies a difficult optimization problem over complex, possibly noisy parameter landscapes. The need for multiple realizations, as in uncertainty quantification or optimization, makes surrogate models an attractive option. For expensive high-fidelity models, however, even performing the number of simulations needed for fitting a surrogate may be too expensive. Inexpensive but less accurate low-fidelity data or models are often also available. Multi-fidelity combine high-fidelity and low-fidelity in order to achieve accuracy at a reasonable cost. Here we consider an hybrid method based on Multi-fidelity for implementing a Bayesian optimization algorithm. At the heart of this algorithm is maximizing the information criterion called acquisition function, and a list of the possible available choices is presented. Multi-fidelity Bayesian optimization achieves competitive performance with an affordable computational overhead for the running time of non-optimized models.

### Installing
After cloning the repository, launch the command:
```sh
conda env create -f advpy.yml
```
