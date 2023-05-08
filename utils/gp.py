import torch
import matplotlib.pyplot as plt
import gpytorch

def raw_gp(X_train, Y_train, X_test, Y_test):

    # Define the Kernel of GP
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self,X_train, Y_train,likelihood):
            super(ExactGPModel, self).__init__(X_train, Y_train, likelihood)
            # this serve for prior
            #self.mean_module = gpytorch.means.ZeroMean()
            #PeriodicKernel =  gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
            #RQKernel =  gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())
            #self.covar_module = gpytorch.kernels.ProductKernel(PeriodicKernel,RQKernel)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) 

        
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x,covar_x)


    ## FIT THE MODEL.. change X train and Y train with a subset ..

    #likelihood = gpytorch.likelihoods.GaussianLikelihood()
    noise_level = 0.01
    noise = noise_level*torch.ones(X_train.shape[0])
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noise, learn_additional_noise=False)

    model = ExactGPModel(X_train, Y_train, likelihood)


    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters())  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    training_iter  = 500
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(X_train)
        # Calc loss and backprop gradients
        loss = -mll(output, Y_train)
        loss.backward()
        if (i+1) % 100 == 0:
            print('Iter {}/{} - Loss: {} LenghtParam {} '.format(
                i + 1, training_iter, loss.item(), model.covar_module.base_kernel.lengthscale.detach().numpy()[0,0] ))
        optimizer.step()


    model.eval()
    likelihood.eval()
    #test_x = torch.FloatTensor(np.linspace(X_train.min(),X_train.max(),100))
    test_x = torch.linspace(X_train.min(), X_train.max(), 100)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Obtain the predictive mean and covariance matrix
        f_preds = model(test_x)
        Y_mean = f_preds.mean
        Y_cov = f_preds.covariance_matrix
        # Make predictions by feeding model through likelihood
        observed_pred = likelihood(model(test_x))
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(8, 6))
        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        
        ax.plot(X_train.numpy(), Y_train.numpy(), 'k*')
        ax.plot(X_test.numpy(), Y_test.numpy(), 'r*')

        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        # ax.set_ylim([-3, 3])
        ax.legend(['Observed Data','Test', 'Mean', 'Confidence'])