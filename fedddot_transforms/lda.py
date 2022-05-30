import torch
from torch import nn, optim
import pandas as pd
import numpy as np
from random import randint
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.stats import norm

class LDAData(Dataset):
    def __init__(self, dataFrame: pd.DataFrame, coordLables: list):
        self.__X = torch.tensor(dataFrame[coordLables].values, dtype = torch.float)
        self.__len = len(dataFrame)
    
    def __len__(self) -> int:
        return self.__len
    
    def __getitem__(self, idx: int):
        return self.__X[idx, :]

class LDA(nn.Module):
    def __init__(self, Hin: int):
        super(LDA, self).__init__()
        assert Hin >= 2, ValueError('Parameter Dim must be greater than zero!')
        self.__Hin = Hin
        self.__params = nn.parameter.Parameter(
            data = torch.rand(size = (self.__Hin + 1, 1), dtype = torch.float),
            requires_grad = True
        )
    
    @property
    def Hin(self) -> int:
        return self.__Hin
    
    @property
    def params(self) -> torch.Tensor:
        return self.__params.data.detach()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        ndim = X.ndim
        assert ndim == 1 or ndim == 2, ValueError('Dimensionality of input vector must be 1 or 2!')
        P = None
        if ndim == 1:
            P = X.unsqueeze(0)
        else:
            P = X
        (Nbatch, Hin) = P.shape
        assert Hin == self.__Hin, ValueError(f'Dimension of input vector is incorrect (given Hin = {Hin}, required Hin = {self.__Hin})')
        
        # h = (Ax + By + Cz + D) / SQRT(A^2 + B^2 + C^2)
        P_emb = torch.cat(
            (P, torch.full(size = (Nbatch, 1), fill_value = 1.0, dtype = torch.float)),
            dim = 1
        )
        numerator = P_emb @ self.__params
        denominator_params = self.__params[: -1]
        denominator_SS = denominator_params.transpose(0, 1) @ denominator_params
        denominator = torch.sqrt(denominator_SS)
        h = numerator / denominator
        return h
    
    def train(self, dataFrame: pd.DataFrame, coordLables: list,
                    Nbatch: int, schedule: list):
        vmData = LDAData(dataFrame, coordLables)
        dataLoader = DataLoader(
            dataset = vmData,
            batch_size = Nbatch,
            shuffle = True
        )

        Etr = []
        for schedule_element in schedule:
            (lr, Nepoch) = schedule_element
            opt = optim.SGD(
                params = [self.__params],
                lr = lr,
                momentum = 0.9
            )
            for epoch in range(Nepoch):
                X = next(iter(dataLoader))
                h = self(X)#; print(f'h = {h}')

                h_pos_masked = h * (h > 0).float()
                h_neg_masked = h * (h < 0).float()

                hpos = h_pos_masked[torch.nonzero(h_pos_masked, as_tuple = True)]
                hneg = h_neg_masked[torch.nonzero(h_neg_masked, as_tuple = True)]
                
                mpos = 0.0; Vpos = 0.0
                if hpos.numel() >= 1:
                    mpos = hpos.mean()
                if hpos.numel() >= 2: 
                    Vpos = hpos.var()
                
                mneg = 0.0; Vneg = 0.0
                if hneg.numel() >= 1:
                    mneg = hneg[torch.nonzero(hneg)].mean()
                if hneg.numel() >= 2:
                    Vneg = hneg[torch.nonzero(hneg)].var()   

                lossC = nn.MSELoss()(self.__params[2], torch.full_like(self.__params[2], fill_value = 1.0))
                lossD = nn.MSELoss()(self.__params[3], torch.full_like(self.__params[3], fill_value = 0.0))
                lossV = (Vpos + Vneg)
                lossM = (mpos - mneg)
                
                loss = 1.0 * lossV + 1.0 * lossM + 0.3 * lossC + 0.3 * lossD
                
                Etr.append(loss.item())
                opt.zero_grad()
                loss.backward()
                opt.step()
 
        return Etr
    
    def plot(self, ax: plt.Axes, x_range: np.ndarray, y_range: np.ndarray):
        A = self.params[0].item(); B = self.params[1].item(); C = self.params[2].item(); D = self.params[3].item()
        X = []; Y = []; Z = []
        for i in range(x_range.shape[0]):
            for j in range(y_range.shape[0]):
                x = x_range[i]; y = y_range[j]
                z = - (A/C) * x - (B/C) * y - (D/C)
                X.append(x); Y.append(y); Z.append(z)
        ax.scatter(X, Y, Z, label = 'LDA Plane')
        return
