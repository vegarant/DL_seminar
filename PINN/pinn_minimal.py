# Setup
import torch
import torch.nn as nn
import torch.nn.functional as F
# pip install tqdm py-pde
from tqdm import tqdm
from pde import CartesianGrid, solve_laplace_equation

#torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.manual_seed(0)
torch.set_num_threads(1)


# Neural network
class MLP(nn.Module):
  def __init__(self, n):
    super().__init__()
    self.W1 = nn.Parameter( torch.randn(2, n) )
    self.b1 = nn.Parameter( torch.randn(n) )
    self.W2 = nn.Parameter( torch.randn(n, n)/n**.5 ) # n**.5 = sqrt(n)
    self.b2 = nn.Parameter( torch.randn(n)/n**.5 )
    self.W3 = nn.Parameter( torch.randn(n)/n**.5 )
    self.b3 = nn.Parameter( torch.randn(1)/n**.5 )

  def forward(self, pos):
    h1 = F.tanh(pos @ self.W1 + self.b1)
    h2 = F.tanh(h1  @ self.W2 + self.b2)
    return      h2  @ self.W3 + self.b3

neurons = 20
net = MLP(neurons)

# Check that it works:
#print(net(torch.tensor([1.,1.])))

# Alternative approach to make an MLP using pytorch magic
#net = nn.Sequential(nn.Linear(2,neurons), nn.Tanh(), nn.Linear(neurons,neurons), nn.Tanh(), nn.Linear(neurons,1), nn.Flatten(0))


# Generate training data
# Interior points
sample_int = torch.rand(10**3,2)

# Boundary points - we use sine boundary conditions
sample_bnd = torch.rand(4,10**2,2)
sample_bnd[0,:,0] = 0
sample_bnd[1,:,0] = 1
sample_bnd[2,:,1] = 0
sample_bnd[3,:,1] = 1

pi = torch.pi
bnd_vals = torch.cat([torch.sin(2*pi * sample_bnd[0,:,1]),
                      torch.sin(2*pi * sample_bnd[1,:,1]),
                      torch.sin(2*pi * sample_bnd[2,:,0]),
                      torch.sin(2*pi * sample_bnd[3,:,0])])
sample_bnd = sample_bnd.reshape(-1,2)


# Loss / objective function
def Lint():
  # We use finite differences to approximate the Laplacian
  eps = 1e-2
  dx = torch.tensor([eps,0])
  dy = torch.tensor([0,eps])
  return ( ( net(sample_int+dx) + net(sample_int+dy) + net(sample_int-dx) + net(sample_int-dy) - 4*net(sample_int) ) / eps**2 ).pow(2).mean()

# Boundary conditions
def Lbnd():
  return (net(sample_bnd)-bnd_vals).pow(2).mean()

def L():
  return Lint()+Lbnd() * 1e3


# Optimization
# This is basically just "boilerplate" code for optimization,
# i.e. it's just the syntax for pytorch optimization
opt = torch.optim.LBFGS(net.parameters(), history_size=10, max_iter=100)
#opt = torch.optim.Adam(net.parameters(), lr=1e-3) # This is an alternative optimizer
def closure():
  opt.zero_grad()
  loss = L()
  loss.backward()
  return loss

print(Lint(), Lbnd())
for it in tqdm(range(10)): # tqdm gives a nice loading bar
  opt.step(closure)
  print(Lint(), Lbnd())


# Evaluation of solution
N = 64
grid = CartesianGrid([[0, 1]] * 2, N)
bcs = [{"value": "sin(2*pi * y)"}, {"value": "sin(2*pi * x)"}]

# Use py-pde to calculate ground truth solution for comparison
res = solve_laplace_equation(grid, bcs)
#res.plot()

# Evaluate our solution
torch_grid = torch.cartesian_prod(torch.linspace(0,1,N), torch.linspace(0,1,N))
torch_pred = net(torch_grid).view(N,N).detach().numpy()

# Write out RMSE (Root Mean Square Error) and plot
print('RMSE:', torch.tensor(torch_pred-res.data).norm().item()/N)
res.data[:,:] = torch_pred
res.plot()
