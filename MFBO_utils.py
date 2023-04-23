# MultiFidelity Bayesian Optimization library for helper functions
import os
SMOKE_TEST = os.environ.get("SMOKE_TEST")

from sklearn.decomposition import PCA
   
# Compute PCA 
def PCA_transformation(pts_original):
    pca = PCA(n_components=4, svd_solver='full').fit(pts_original)
    pts_transformed = pca.transform(pts_original)
    return pts_transformed

import numpy as np
from pyDOE import lhs
from scipy.interpolate import NearestNDInterpolator

# Sample RSMs
def data_resample(pts, obs):
    dim = pts.shape[1]
    N = 10000
    lb = np.min( pts,axis=0)
    ub = np.max( pts,axis=0)
    bounds = {'lb': lb, 'ub': ub}
    # Generate latin-hypercube
    new_pts = lb + (ub - lb) * lhs(dim, N) 
    # pts are not in convex hull of pts (LD interpolator does not extrapolate) 
    r =  NearestNDInterpolator( pts, obs)
    # pts has to be inside region of interpolation .
    valuesTrasf = r(new_pts) 
    #valuesTrasf.reshape(-1,1).T
    return new_pts,valuesTrasf

import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.model_selection import train_test_split

class TestcaseDataset(Dataset):
    def __init__(self,new_pts,new_obs,dim):
        data = np.hstack([new_pts,new_obs]) 
        hifi, lofi = train_test_split(data, test_size=1e-3, shuffle=True)
        hifi, test = train_test_split(hifi, test_size=1e-1)
        size=dim-data.shape[1]
        # Cast them
        self.X_hifi = torch.Tensor(hifi[:,:size]) 
        self.X_lofi = torch.Tensor(lofi[:,:size]) 
        self.X_test = torch.Tensor(test[:,:size]) 
        self.Y_hifi = torch.Tensor(hifi[:,size:])#.unsqueeze(-1)
        self.Y_lofi = torch.Tensor(lofi[:,size:])#.unsqueeze(-1)
        self.Y_test = torch.Tensor(test[:,size:])#.unsqueeze(-1)

        self.hifi_dataset = TensorDataset(self.X_hifi, self.Y_hifi)
        self.test_dataset = TensorDataset(self.X_test, self.Y_test)
    def __call__(self):
        return (self.hifi_dataset, 
                self.test_dataset, 
                self.X_hifi, 
                self.X_lofi, 
                self.X_test, 
                self.Y_hifi, 
                self.Y_lofi, 
                self.Y_test, 
                )


from torch import nn # Neural Network Module
# Define hyperparameters

hidden_size = 128
learning_rate = 1e-2
# Define the neural network model (input, hidden, output size)
class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(NeuralNet, self).__init__()
        # Define layers
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.LeakyReLU() # activation function nn.LeakyReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        #out = self.sigmoid(out)
        return out
    

import gpytorch
device = "cuda" if torch.cuda.is_available() else "cpu" 
from botorch.models import SingleTaskGP, FixedNoiseGP, ModelListGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

# define a function to initialize the Gaussian Process model
def initialize_model(X, Y, noise, state_dict=None):
  # move X and Y to the device
  train_x, train_obj = X.to(device), Y.to(device)
  # create an empty list of models
  models = []
  # define a Gaussian likelihood with specified noise level
  likelihood = gpytorch.likelihoods.GaussianLikelihood(noise=noise)#, learn_additional_noise=False)
  
  # loop over each output dimension in Y
  for i in range(train_obj.shape[-1]):
      train_y = train_obj[..., i : i + 1]
      # create a single task Gaussian Process for each output dimension
      models.append(
          SingleTaskGP(
              train_x, train_y
          )
      )
  # create a model list GP by passing all the created SingleTaskGP models
  model = ModelListGP(*models)
  # define the marginal log-likelihood as sum of marginal log-likelihoods of all the models
  mll = SumMarginalLogLikelihood(model.likelihood, model)

  #mll = ExactMarginalLogLikelihood(likelihood, single_model) # OTHER LIKELIHOOD ?
  ''' load state dict if it is passed '''
  if state_dict is not None:
    model.load_state_dict(state_dict)
  return mll, model


from botorch.optim import optimize_acqf
from botorch.utils.sampling import manual_seed

NUM_POINTS = 1 if not SMOKE_TEST else 1
NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 32

def opt_acq_and_get_next_obj(acqf,bounds,nn_model):
  """Optimizes the acquisition function, and returns a new candidate."""
  with manual_seed(1234):
    candidates, acq_value = optimize_acqf(
        acq_function=acqf, 
        bounds=bounds,
        q=NUM_POINTS,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        options={"batch_limit": 10, "maxiter": 200},
    )
  ''' get a new obj '''
  with torch.no_grad():
    predicted_optimal_obj = nn_model(candidates)
  return candidates, predicted_optimal_obj
    

# Define the Kernel of Gaussian Process
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self,X_train, Y_train, likelihood=gpytorch.likelihoods.GaussianLikelihood()):
        super(ExactGPModel, self).__init__(X_train, Y_train, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) 
       
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x,covar_x)
    

def gp_visualization(x_train,y_train,alpha=0,iter=1000,acqf='acqf'):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    train_multi_yvar = noise**2*torch.ones(x_train.unsqueeze(-1).shape[0],1, device=device)
    ''' Define the model '''
    #gp_model = FixedNoiseGP(x_train.unsqueeze(-1), y_train, train_multi_yvar)
    #gp_model = SingleTaskGP(x_train.unsqueeze(-1), y_train, likelihood = likelihood  )
    gp_model = ExactGPModel(x_train, y_train.squeeze(), likelihood)
    ''' Find optimal model hyperparameters '''
    gp_model.train()
    likelihood.train()
    ''' Use the adam optimizer '''  # Includes GaussianLikelihood parameters
    optimizer = torch.optim.Adam(gp_model.parameters(),lr=1e-3)
    ''' "Loss" for GPs - the marginal log likelihood '''
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)
    training_iter = iter
    for i in range(training_iter):
        ''' Zero gradients from previous iteration '''
        optimizer.zero_grad()    
        ''' Output from model '''
        output = gp_model(x_train)
        ''' compute loss and backprop gradients '''
        loss = -mll(output, y_train.squeeze())
        loss.backward()
        if (i+1) % 100 == 0:
            print(f'Wing{alpha+1}: {acqf} - Iter {i + 1}/{training_iter}')# - Loss: {loss.item():.5f} LenghtParam {gp_model.covar_module.base_kernel.lengthscale.detach().numpy()[0,0]:.5f}')
        optimizer.step()
    print('\n')
    return gp_model.eval(), likelihood.eval()

import matplotlib.pyplot as plt
def plot_gp(gp_model_acqf,acqf):
    fig = plt.figure(figsize=(21, 9))

    for alpha in range(2):#X_lofi.shape[1]):

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(X_hifi[:,alpha].min(), X_hifi[:,alpha].max(), 100)
            # Make predictions by feeding model through likelihood / gp_model
            observed_pred_acqf = gp_model_acqf((test_x ))  
            # Initialize plot 
            ax = fig.add_subplot(2, 2, alpha+1)      
            ax.scatter(X_test[:,alpha].numpy(), Y_test[:,-1].numpy(), alpha=0.8, label = 'Test')
            
            # Get upper and lower confidence bounds # Plot predictive means as blue line
            lower, upper = observed_pred_acqf.confidence_region() 
            ax.plot(test_x.numpy(), observed_pred_acqf.mean.numpy(), label = f'Mean_{acqf}')    
            ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5, label = f'Confidence_{acqf}')  
            ax.scatter(eval(f'X_lofi_{acqf}')[:,alpha].numpy(), eval(f'Y_lofi_{acqf}')[:,-1].numpy(), s=50,label = f'Observed Data_{acqf}')

            ax.set_ylim([-4,3])
            plt.xlabel(f"PCA-Wing{alpha+1}")
            ax.legend()