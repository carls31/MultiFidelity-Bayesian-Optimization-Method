import torch
print('Running on PyTorch {}'.format(torch.__version__))

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

import pandas as pd
from sklearn.model_selection import train_test_split
#df = pd.read_csv('Case3_2nd_launch_WingsConvCoeffs_Info.csv', sep=";")
df = pd.read_excel('Case5_1st_launch_WingsConvCoeffs_Info.xlsx')
df_conv = df[df['Unnamed: 47'] == 'CONVERGED']
hifi, lofi = train_test_split(df_conv, test_size=0.008, random_state=42 )

hifi_x = hifi.alpha0.to_numpy()
lofi_x  = lofi.alpha0.to_numpy()
hifi_y = hifi.Cy0Mean.to_numpy()
lofi_y  = lofi.Cy0Mean.to_numpy()

# normalize features
mean = hifi_x.mean( )
std = hifi_x.std( ) + 1e-6 # prevent dividing by 0
hifi_x = (hifi_x - mean) / std
lofi_x = (lofi_x - mean) / std

# normalize labels
mean, std = hifi_y.mean(), hifi_y.std()
hifi_y = (hifi_y - mean) / std
lofi_y = (lofi_y - mean) / std

# Cast them
X_hifi = torch.FloatTensor(hifi_x).unsqueeze(-1)
X_lofi = torch.FloatTensor(lofi_x).unsqueeze(-1)
Y_hifi = torch.FloatTensor(hifi_y).unsqueeze(-1)
Y_lofi = torch.FloatTensor(lofi_y).unsqueeze(-1)

best_obs_value = lofi_y.max()
bounds = torch.FloatTensor([[-1.9], [1.6]])

from torch import nn
# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(1, 24)
        self.fc2 = nn.Linear(24, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

model = NeuralNetwork().to(device)
print(model)

#--- OPTIMIZING THE PARAMETERS OF THE NN ---#

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# In a single training epoch, the model makes predictions on the training dataset (fed to it in batches)
# and backpropagates the prediction error to update the model’s parameters
def train(X,Y, model, loss_fn, optimizer):
    size = len(X)
    model.train()

    # Compute prediction error
    pred = model(X)
    loss = loss_fn(pred, Y)

    # Backpropagation
    optimizer.zero_grad() # clears old gradients from the last step
    loss.backward()
    optimizer.step()

    loss, current = loss.item(), len(X)
    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def pred(new_point, model):
    model.eval()
    with torch.no_grad():
            predicted_p = model(new_point)
    return predicted_p

from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood   
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim import optimize_acqf
def get_next_points(X_lofi, Y_lofi,best_obs_value, bounds, n_points=1):
  single_model = SingleTaskGP(X_lofi, Y_lofi)
  mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)

  fit_gpytorch_mll(mll, retain_graph=True)

  EI = qExpectedImprovement( model = single_model, best_f = best_obs_value )

  candidates, _ = optimize_acqf(
    acq_function = EI,
    bounds = bounds,
    q = n_points,
    num_restarts= 200,
    raw_samples = 512 )
  return candidates

epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(X_hifi, Y_hifi, model, loss_fn, optimizer)
print("Done!")

n_runs = 3
for i in range(n_runs):

   print(f"No of optimization: {i+1}")
   new_point = get_next_points(X_lofi, Y_lofi, best_obs_value, bounds, 16)
   print(f"New candidates are: {new_point}")
   
   predicted_optimal_point = pred(new_point,model) 
   print(f"New predicted points are: {predicted_optimal_point}")

   X_lofi = torch.cat([X_lofi, new_point ])
   Y_lofi = torch.cat([Y_lofi, predicted_optimal_point])

import matplotlib.pyplot as plt
f, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.plot(X_hifi.numpy(), Y_hifi.numpy(), 'r*',label = 'target')
ax.plot(X_lofi.numpy(), Y_lofi.numpy(), 'k*',label = 'predicted')
plt.legend(loc="lower left")
plt.show()