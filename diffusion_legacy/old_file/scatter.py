# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from typing import Optional
from torch_geometric.typing import OptTensor

import torch
from torch.nn import Parameter, Conv1d

from torch_geometric.nn.inits import zeros
from torch_geometric.utils import get_laplacian
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
from torch_geometric.transforms.to_sparse_tensor import ToSparseTensor
# adapted from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/cheb_conv.html
def diffusion_scat(f, W, K=3, t=3, layer=3):
    G = nx.from_scipy_sparse_matrix(W)
    D = np.array([np.max((d, 1)) for (temp, d) in list(G.degree)])
    Dhalf = np.diag(1 / np.sqrt(D))
    A = np.matmul(np.matmul(Dhalf, W.todense()), Dhalf)
    T = np.matmul(W.todense(),np.diag(1/D))#(np.eye(np.shape(D)[0]) + A) / 2 ###sym diffusion operator np.matnul(A,np.diag(1 / D)
    U = np.linalg.matrix_power(T, t)
    psi = []
    for idx in range(K):
        if idx == 0:
            psi.append(np.eye(np.shape(D)[0]) - T)
        else:
            T0 = T
            T = np.matmul(T0, T0)
            psi.append(T0 - T)

    y_next = [f]
    y_out = [np.matmul(U, np.absolute(f))]###UX alone

    for i in range(layer - 1):
        for k in range(len(y_next)):
            y_next_new = []
            ftemp = y_next.pop(0)
            ftemp = np.absolute(ftemp)
            y = [np.matmul(fltr, ftemp) for fltr in psi]
            y_out.extend([np.matmul(U, np.absolute(y_temp)) for y_temp in y])
            y_next_new.extend(y)
        y_next = y_next_new
    y_out = np.concatenate(tuple(y_out), axis=0)  # use this to form a single matrix
    return y_out


class ScatAgg(MessagePassing):
    r"""
    """

    def __init__(self,
                 J: int = 3,  # max scale of scattering
                 layer: int = 1,  # layer of scattering
                 nlp: int = 1,  # nonlinear power
                 normalization: Optional[str] = 'sym',
                 **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        assert J > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.normalization = normalization

        self.J = J
        self.layer = layer
        self.nlp = nlp
    def forward(self,
                x,
                edge_index: OptTensor = None,
                # edge_index, edge_weight: OptTensor = None,
                batch: OptTensor = None,
                ):
        """"""
        # if self.normalization != 'sym' and lambda_max is None:
        #     raise ValueError('You need to pass `lambda_max` to `forward() in`'
        #                      'case the normalization is non-symmetric.')

        # if lambda_max is None:
        #     lambda_max = torch.tensor(2.0, dtype=x.dtype, device=x.device)
        # if not isinstance(lambda_max, torch.Tensor):
        #     lambda_max = torch.tensor(lambda_max, dtype=x.dtype,
        #                               device=x.device)
        # assert lambda_max is not None

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        out = torch.empty(x.shape[0], x.shape[1], 0).cuda()

        y_prop_now = x.unsqueeze(-1)
        for _ in range(self.layer):
            y_prop_next = torch.empty(x.shape[0], x.shape[1], 0).cuda()
            for idx_prop in range(y_prop_now.shape[-1]):
                y_out, y_prop = self.scatprop(y_prop_now[:, :, idx_prop], edge_index)
                y_prop_next = torch.cat((y_prop_next, y_prop), -1)
                out = torch.cat((out, y_out.unsqueeze(-1)), -1)
            y_prop_now = y_prop_next

        return out

    def scatprop(self,
                 x_in,
                 edge_index: OptTensor = None,
                 ):

        y_prop = torch.empty(x_in.shape[0], x_in.shape[1], self.J).cuda()

        if len(x_in.shape) > 2:
            x_in = x_in.squeeze()

        x_sub = x_in

        for idx_j in range(self.J):

            x_temp = x_sub
            if idx_j == 0:
                x_sub = self.propagate(edge_index, x=x_sub)
            else:
                for _ in range(2 ** (idx_j - 1)):
                    x_sub = self.propagate(edge_index, x=x_sub)

            y_pass_add = x_temp - x_sub
            y_pass_add = torch.pow(torch.abs(y_pass_add), self.nlp)

            y_prop[:, :, idx_j] = y_pass_add  # high-pass

        y_out = x_sub  # low-pass

        return y_out, y_prop

    def message(self, x_j):
        return 0.5 * x_j  # message = 0.5(I+A) following diffusion scattering

    def __repr__(self) -> str:
        # return (f'{self.__class__.__name__}({self.in_channels}, '
        #         f'{self.out_channels}, K={len(self.lins)}, '
        #         f'normalization={self.normalization})')
        return (f'{self.__class__.__name__}(J={self.J}, '
                f'layer={self.layer}, '
                f'power={self.nlp})'
                # f'normalization={self.normalization})'
                )







# =============================================================================
# scattering transform
# =============================================================================
''' get one-step scattering coefficients using Paley-Littlewood wavelet '''
def propLayer(f, lamb, V, K=4, N=1):
    # K: scale; N: bandwidth
    idx = []
    for k in range(K):
        if k == 0:
            idx.append(lamb < N)
        else:
            idx.append((lamb >= 2 ** (k - 1) * N) * (lamb < 2 ** k * N))

    y = []
    for k in range(K):
        y.append(np.matmul(np.matmul(V[:, idx[k]], V[:, idx[k]].T), f))

    return y


''' get one-step scattering coefficients using a general wavelet '''
''' change the name of the function propLayerHaar to propLayer in order to use '''
''' using haar wavelet as an example, replace it with any wavelet '''


def phi(lamb):
    phi = np.sinc(2 * lamb)
    return phi
def psi(lamb):
    psi = np.sinc(lamb) * (1 - np.cos(np.pi * lamb))
    return psi

def propLayerHaar(f, lamb, V, J=3):  # to replace propLayer
    y = []
    for k in range(J):
        j = J - k
        if j == J:
            H = phi(2 ** j * lamb)
        else:
            H = psi(2 ** (-j) * lamb)
        H = np.diag(H)
        y.append(np.matmul(np.matmul(np.matmul(V, H), V.T), f))
    return y


''' get all scattering coefficients '''


def getRep(f, lamb, V, layer=3, N=1):
    y_out = []
    y_next = []
    y = propLayer(f, lamb, V, N=N)
    y_out.append(y.pop(0))
    y_next.extend(y)
    for i in range(layer - 1):
        for k in range(len(y_next)):
            ftemp = y_next.pop(0)
            ftemp = np.absolute(ftemp)
            y = propLayer(ftemp, lamb, V, N=N)
            y_out.append(y.pop(0))
            y_next.extend(y)
    y_out = np.concatenate(tuple(y_out), axis=1)  # use this to form a single matrix
    return y_out


# =============================================================================
# diffusion transform
# =============================================================================

def diffusion_scat(f, W, K=3, t=3, layer=3):
    G = nx.from_scipy_sparse_matrix(W)
    D = np.array([np.max((d, 1)) for (temp, d) in list(G.degree)])
    Dhalf = np.diag(1 / np.sqrt(D))
    A = np.matmul(np.matmul(Dhalf, W.todense()), Dhalf)
    T = (np.eye(np.shape(D)[0]) + A) / 2 ###(I+T)/2,sym diffusion
    U = np.linalg.matrix_power(T, t)  ###U=T^3(2^2-1=3)
    psi = []
    for idx in range(K):
        if idx == 0:
            psi.append(np.eye(np.shape(D)[0]) - T) ###I-T
        else:
            T0 = T
            T = np.matmul(T0, T0)
            psi.append(T0 - T)###T^(2^(j-1))-T^(2^j)
    ##psi={psi_0,psi_1,psi_2}
    y_next = [f]
    y_out = [np.matmul(U, np.absolute(f))]  ###Ux

    for i in range(layer - 1):
        for k in range(len(y_next)):
            y_next_new = []
            ftemp = y_next.pop(0)
            ftemp = np.absolute(ftemp) ####|high pass|
            y = [np.matmul(fltr, ftemp) for fltr in psi]###filter num=3
            y_out.extend([np.matmul(U, np.absolute(y_temp)) for y_temp in y])
            y_next_new.extend(y)
        y_next = y_next_new
    y_out = np.concatenate(tuple(y_out), axis=0)  # use this to form a single matrix
    return y_out


# =============================================================================
# gaussianization
# =============================================================================

def gaussianization_whiten(A, pca=True, num_of_components=8):
    '''A is data matrix with size (# of sample) X (# of dimension)'''
    if not pca:
        num_of_components = A.shape[1]
    pca = PCA(n_components=num_of_components)
    pca.fit(A)
    A_mean = pca.mean_
    V = pca.components_
    lamb = pca.explained_variance_
    lamb_invhalf = 1 / np.sqrt(lamb)
    Sigma_invhalf = np.matmul(np.diag(lamb_invhalf), V)
    A_gaussian = np.matmul(Sigma_invhalf, (A - A_mean).T)
    return A_gaussian.T


def gaussianization_spherize(A, pca=True, num_of_components=8):
    '''A is data matrix with size (# of sample) X (# of dimension)'''

    A = A - np.mean(A, axis=0)
    if pca:
        pca_model = PCA(n_components=num_of_components)
        A = pca_model.fit_transform(A)
    return normalize(A)
