import torch


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
    
class LeakyDeepStencil(torch.nn.Module):
    def __init__(self,Nx,width=3,channels=15,depth=3,act="LeakyReLU",
                front_bias=True):
        super(LeakyDeepStencil,self).__init__()
        ActFuncDict = {
            "LeakyReLU":torch.nn.LeakyReLU,
            "ReLU":torch.nn.ReLU,
            "Tanh":torch.nn.Tanh,
            "Sigmoid":torch.nn.Sigmoid,
            "CELU":torch.nn.CELU,
        }
        ActFunc = ActFuncDict[act]
        layers = [[ torch.nn.Conv1d(1,channels,width,bias=front_bias),ActFunc() ]] + \
                 [ [torch.nn.Conv1d(channels,channels,1),ActFunc() ]
                   for _ in range(depth-1) ] + \
                 [[ torch.nn.Conv1d(channels,1,1) ]]
        self.net = torch.nn.Sequential(*[i for l in layers for i in l])
        self.act = act
        self.depth = depth
        self.width = width
        self.channels = channels
    def forward(self,x):
        return torch.nn.functional.pad(self.net(x),
                        (self.width//2,self.width//2))
    
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
    
class LeakyFCMLP(torch.nn.Module):
    def __init__(self, Nx):
        super(LeakyFCMLP,self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(Nx,100),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(100,100),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(100,100),
            torch.nn.LeakyReLU(),
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
    
models = {"PureStencil":PureStencil,
         "PureLinear":PureLinear,
          "DeepStencil":DeepStencil,
          "LeakyDeepStencil":LeakyDeepStencil,
          "LeakyFCMLP":LeakyFCMLP,
          "FCMLP":FCMLP
         }
    
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

discriminators = {
    "ConditionalConv":ConditionalDiscriminatorConv,
}
