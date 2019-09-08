import torch
import numpy as np
from matplotlib import pylab as plt

problems = ['burgers','KdV','heat','wave']

#
# Networks we're evaluating
#
class PureStencil(torch.nn.Module):
    def __init__(self, Nx, width=3):
        super(PureStencil,self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(1,1,width,bias=False),
        )
    def forward(self,x):
        return torch.nn.functional.pad(self.net(x),(1,1))
  
class PureLinear(torch.nn.Module):
    def __init__(self, Nx, width=3):
        super(PureLinear,self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(Nx,Nx-2,bias=False),
        )
    def forward(self,x):
        return torch.nn.functional.pad(self.net(x),(1,1))
    
class DeepStencil(torch.nn.Module):
    def __init__(self,Nx,width=3):
        super(DeepStencil,self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(1,15,width),
            torch.nn.ReLU(),
            torch.nn.Conv1d(15,15,1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(15,15,1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(15,1,1)
        )
    def forward(self,x):
        return torch.nn.functional.pad(self.net(x),(1,1))
    
class FCMLP(torch.nn.Module):
    def __init__(self, Nx):
        super(FCMLP,self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(Nx,100),
            torch.nn.ReLU(),
            torch.nn.Linear(100,100),
            torch.nn.ReLU(),
            torch.nn.Linear(100,100),
            torch.nn.ReLU(),
            torch.nn.Linear(100,Nx-2)
        )
    def forward(self,x):
        return torch.nn.functional.pad(self.net(x),(1,1))
    
class AutoencoderFC(torch.nn.Module):
    def __init__(self, Nx):
        super(AutoencoderFC,self).__init__()
        self.net = torch.nn.Sequential(
            
        )
    def forward(self,x):
        return self.net(x)
    
class AutoencoderConv(torch.nn.Module):
    def __init__(self, Nx):
        super(AutoencoderConv,self).__init__()
        self.net = torch.nn.Sequential(
            
        )
    def forward(self,x):
        return self.net(x)
    
class DiscriminatorFC(torch.nn.Module):
    def __init__(self, Nx, width):
        super(DiscriminatorFC,self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(Nx,100),
            torch.nn.ReLU(),
            torch.nn.Linear(100,50),
            torch.nn.ReLU(),
            torch.nn.Linear(50,20),
            torch.nn.ReLU(),
            torch.nn.Linear(20,10),
            torch.nn.ReLU(),
            torch.nn.Linear(10,1),
            torch.nn.Sigmoid(),
        )
    def forward(self,x):
        return self.net(x)
    
    
class DiscriminatorConv(torch.nn.Module):
    def __init__(self, Nx, width):
        super(DiscriminatorConv,self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(1,5,width),
            torch.nn.LeakyReLU(),
            torch.nn.AvgPool1d(2),
            torch.nn.Conv1d(5,5,width),
            torch.nn.LeakyReLU(),
            torch.nn.AdaptiveAvgPool1d(16),
            torch.nn.Conv1d(5,1,width),
            torch.nn.LeakyReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Sigmoid(),
        )
        
    def forward(self,x):
        return self.net(x)
    
class ConditionalDiscriminatorConv(torch.nn.Module):
    def __init__(self, Nx, width):
        super(ConditionalDiscriminatorConv,self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(2,5,width),
            torch.nn.LeakyReLU(),
            torch.nn.AvgPool1d(2),
            torch.nn.Conv1d(5,5,width),
            torch.nn.LeakyReLU(),
            torch.nn.AdaptiveAvgPool1d(16),
            torch.nn.Conv1d(5,1,width),
            torch.nn.LeakyReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Sigmoid(),
        )
        
    def forward(self,x,y):
        return self.net( torch.cat((x,y),dim=1) )
    
models = {"PureStencil":PureStencil,
         "PureLinear":PureLinear,
          "DeepStencil":DeepStencil,
         "FCMLP":FCMLP}
#
# Training Utilities
#
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def select_batch_idcs(num, Ntraj,Nt, Npast=1,Nfuture=1):
    ii = torch.LongTensor([(np.random.choice(Ntraj),
                            np.random.choice(Nt-Npast-Nfuture+1))
                            for _ in range(num)])
    return ii

Ntraj_test = 2
Ntraj_val = 2
def get_batch(num, dataset, Npast=1, Nfuture=1):
    Ntraj,Nt,Nx = dataset.shape
    ii = select_batch_idcs(num, Ntraj-Ntraj_test-Ntraj_val, Nt)
    xx = torch.cat([dataset[j,(t-Npast+1):(t+1),:].unsqueeze(0) for j,t in ii])
    yy = torch.cat([dataset[j,(t+1):(t+Nfuture+1),:].unsqueeze(0) for j,t in ii])
    return xx,yy

def test_batch(dataset, Npast=1, Nfuture=1):
    Ntraj,Nt,Nx = dataset.shape
    ii = torch.LongTensor([(i,j) for i in range(Ntraj-Ntraj_val-Ntraj_test, Ntraj-Ntraj_val)
                          for j in range(Nt-Npast-Nfuture+1)])
    xx = torch.cat([dataset[j,(t-Npast+1):(t+1),:].unsqueeze(0) for j,t in ii])
    yy = torch.cat([dataset[j,(t+1):(t+Nfuture+1),:].unsqueeze(0) for j,t in ii])
    return xx,yy

#
# Evaluation Utilities
#
def do_a_path(model, dataset, samp, Nstep=-1):
    u0 = dataset[samp,(0,),:]
    u0 = u0.reshape((1,1,dataset.shape[-1]))
    #u0 = torch.tensor(u0,dtype=torch.float32)
    plt.figure()
    plt.plot(u0.cpu().numpy().flatten())
    try:
        Npast = model.Npast
    except:
        Npast = 1
    if Nstep <= 0:
        Nstep = dataset.shape[1]-Npast
    Nplot = Nstep//10
    errors = np.zeros((Nstep,))
    with torch.no_grad():
        for i in range(Nstep):
            uN = model(u0)
            u0 += uN
            ucpu = u0.cpu().numpy().ravel()
            dcpu = dataset[samp,i+1,:].cpu().numpy().ravel()
            errors[i] = np.linalg.norm((ucpu - dcpu)/(np.abs(dcpu)+1.0e-5))
            if i%Nplot==Nplot-1:
                plt.plot(ucpu)
                plt.plot(dcpu,'--')
    plt.show()
    return errors
    
