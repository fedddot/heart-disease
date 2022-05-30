import pandas as pd
import numpy as np

def reorder_eugendata(L: np.ndarray, V: np.ndarray) -> tuple([np.ndarray, np.ndarray]):
    if L.size == 1:
        return L, V
    else:
        Lmax = np.argmax(L)
        Lsub, Vsub = reorder_eugendata(
            np.delete(L, Lmax, 0),
            np.delete(V, Lmax, 1)
        )
        Lord = np.concatenate(
            (L[[Lmax]], Lsub),
            axis = 0 
        )
        Vord = np.concatenate(
            (V[:, Lmax].reshape(-1, 1), Vsub),
            axis = 1
        )
        return Lord, Vord

def pca(X = pd.DataFrame, pc_num = int) -> pd.DataFrame:
    assert pc_num > 0, ValueError('pc_num must be greater than zero!')
    Xnorm = X.copy(deep = True)
    for col in Xnorm.columns:
        Xnorm[col] = Xnorm[col] - Xnorm[col].mean()
        Xnorm[col] = Xnorm[col] / Xnorm[col].std()

    cov = Xnorm.cov()
    Lraw, Vraw = np.linalg.eig(cov)
    L, V = reorder_eugendata(Lraw, Vraw)
    (I, J) = V.shape
    if pc_num <= J:
        V = V[:, : pc_num]
    (I, J) = V.shape
    Xpca_np = Xnorm.values @ V
    pca_columns = [f'PC{j}' for j in range(J)]
    Xpca = pd.DataFrame(data = Xpca_np, columns = pca_columns, index = X.index)
    return Xpca, L, V

