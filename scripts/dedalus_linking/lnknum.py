

import numpy as np
from numba import jit, prange
import pickle

def compute_link_DS(s1, s2):
    """Compute linking number between two Streamline objects."""
    # Make closed polylines from streamline nodes
    x1 = np.vstack((s1.x, s1.x[0:1]))
    x2 = np.vstack((s2.x, s2.x[0:1]))
    # Call compiled linking number computation
    return _compute_link_DS(x1, x2)


def compute_partial_link_DS(s1, s2, idx, filename, dL=1):
    """Compute partial linking numbers between two Streamline objects. 
    Both Streamline objects should have the same lenght L attribute.
    
    s1, s2: Streamline object
    
    idx: tuple of ints
        contains the indices of the streamlines for saving to disk intermediate results
        
    dL: float or int
    """
    # partition s1 and s2 every dL
    L_values = np.arange(0,s1.L + dL,dL)
    nL = len(L_values)
    def split_s(s):
        coord = s.s
        split_coords = np.split(coord, L_values[:-1])
        idx = [0]
        for k in range(1,nL):
            idx.append(np.max(np.where(coord <= L_values[k])) + 1)
        # split polylne
        split_x = [s.x[idx[i]:idx[i + 1]] for i in range(len(idx) - 1)] 
        # get the normalization factor for the line going from 0 to L_values[k]
        split_Tf = np.array([s.T[idx[i]:idx[i + 1]][-1] for i in range(len(idx) - 1)])
        return split_x, split_Tf
    
    ls, ls_Tf = split_s(s1)
    ks, ks_Tf = split_s(s2)
    s1_start = s1.x[0:1] 
    s2_start = s2.x[0:1]
    
    # add s1.x[0:1] and s2.x[0:1] at the beginning of each subpolyline, except first
    for i in range(1,len(ls)):
        ls[i] = np.vstack((s1_start, ls[i]))
        ks[i] = np.vstack((s2_start, ks[i]))
    # add s1.x[0:1] and s2.x[0:1] at the end of each subpolyline, including last
    for i in range(len(ls)):
        ls[i] = np.vstack((ls[i], s1_start))
        ks[i] = np.vstack((ks[i], s2_start))  
    
    # compute linking number between
    lnkNum = 0.
    norm_factor = 0.
    for a in range(nL-1):
        running_L = L_values[a]
        for n in range(a**2, (a+1)**2): # this loop could be parallelized
            i,j = _pairing(n)
            lnkNum += _compute_link_DS(ls[i], ks[j])
        norm_factor = ls_Tf[a] * ks_Tf[a]
        # save to disk (running_L, lnkNum, normfactor)
        #filename_extended = filename + "_L{0}_i{1}j{2}.pickle".format(running_L,
        #                                                                   idx[0],
        #                                                                   idx[1])
        with open(filename, "rb") as file:
            lnks, nfs, Ls = pickle.load(file)
        lnks[idx[0],idx[1],a+1] = lnkNum
        nfs[idx[0],idx[1],a+1] = norm_factor
        with open(filename, "wb") as file:
            pickle.dump((lnks, nfs, Ls), file)
        #with open(filename, "wb") as file:
        #     pickle.dump((running_L, lnkNum, norm_factor), file)
    return lnkNum


@jit(nopython=True, parallel=True)
def _compute_link_DS(ls, ks):
    """
    Compute linking number between two closed polylines with shape (N, 3).
    The last point should equal the first point for each line.
    """
    λ = 0
    Nl = ls.shape[0]
    Nk = ks.shape[0]
    for i in prange(Nk-1):
        for j in range(Nl-1):
            a = ls[j, :] - ks[i, :]
            b = ls[j, :] - ks[i + 1, :]
            c = ls[j + 1, :] - ks[i + 1, :]
            d = ls[j + 1, :] - ks[i, :]
            p = np.dot(a, np.cross(b, c))
            an = np.sqrt(np.dot(a, a))
            bn = np.sqrt(np.dot(b, b))
            cn = np.sqrt(np.dot(c, c))
            dn = np.sqrt(np.dot(d, d))
            d1 = an * bn * cn + np.dot(a, b) * cn + np.dot(b, c) * an + np.dot(c, a) * bn
            d2 = an * dn * cn + np.dot(a, d) * cn + np.dot(d, c) * an + np.dot(c, a) * dn
            λ += (np.arctan2(p, d1) + np.arctan2(p, d2))
    return λ / (2 * np.pi)

@jit(nopython=True)
def _pairing(n):
    fsqrtn = int(np.floor(np.sqrt(n)))
    a = n - fsqrtn * fsqrtn
    if a <= fsqrtn:
        return a, fsqrtn
    else:
        return fsqrtn, 2*fsqrtn - a
