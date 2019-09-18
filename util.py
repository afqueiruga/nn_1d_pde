import torch
import numpy as np
from matplotlib import pylab as plt
    
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
    ii = select_batch_idcs(num, Ntraj-Ntraj_test-Ntraj_val, Nt, 
                           Npast=Npast, Nfuture=Nfuture)
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
def do_a_path(model, dataset, samp, Nstep=-1,Nplot=-1,label=None,x=None,marker=None):
    u0 = dataset[samp,(0,),:]
    u0 = u0.reshape((1,1,dataset.shape[-1]))
    #u0 = torch.tensor(u0,dtype=torch.float32)
    #plt.figure()
    #plt.plot(x,u0.cpu().numpy().flatten())
    try:
        Npast = model.Npast
    except:
        Npast = 1
    if Nstep <= 0:
        Nstep = dataset.shape[1]-Npast
    if Nplot<=0:
        Nplot=10
    plot_freq = max(Nstep//Nplot,1)
    if x is None:
        x=range(41)
    errors = np.zeros((Nstep,))
    with torch.no_grad():
        for i in range(Nstep):
            u0[0,0,0] = dataset[samp,i,0]
            u0[0,0,-1] = dataset[samp,i,-1]
            uN = model(u0)
            u0 += uN
            ucpu = u0.cpu().numpy().ravel()
            dcpu = dataset[samp,i+1,:].cpu().numpy().ravel()
            # Apply BCs

            errors[i] = np.linalg.norm((ucpu - dcpu),2)
            if i%plot_freq==plot_freq-1:
                plt.plot(x,ucpu,label=label,marker=marker)
                #plt.plot(x,dcpu,'--')
    #plt.show()
    return errors
    
def do_an_unknown_path(model, u0, Nstep=-1,Nplot=-1,label=None,x=None,marker=None):
    u0 = u0.reshape((1,1,-1))
    try:
        Npast = model.Npast
    except:
        Npast = 1
    if Nstep <= 0:
        Nstep = 100
    if Nplot<=0:
        Nplot=10
    plot_freq = max(Nstep//Nplot,1)
    if x is None:
        x=range(41)
    #plt.plot(x,u0.cpu().numpy().ravel())
    with torch.no_grad():
        for i in range(Nstep):
            uN = model(u0)
            u0 += uN
            ucpu = u0.cpu().numpy().ravel()
            if i%plot_freq==plot_freq-1:
                plt.plot(x,ucpu,label=label,marker=marker)
    return None
    
