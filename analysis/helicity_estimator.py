import numpy as np
from dedalus_linking.streamline import Streamline
import dedalus_linking.bcc_utils as bcc
from dedalus_linking.lnknum import _pairing
from dedalus_linking.load_dedalus import build_cartesian_vorticity_interpolator
import pyfln
import multiprocessing
from joblib import Parallel, delayed
import pickle
import time


def split_strm(s, L_values):
    nL = len(L_values)
    coord = s.s
    idx = [0]
    for k in range(1, nL):
        idx.append(np.max(np.where(coord <= L_values[k])) + 1)
    # split polylne
    split_x = [s.x[idx[i]:idx[i + 1] + 1] for i in range(len(idx) - 1)]
    # get the normalization factor for the line going from 0 to L_values[k]
    split_Tf = np.array([s.T[idx[i]:idx[i + 1] + 1][-1] for i in range(len(idx) - 1)])
    return split_x, split_Tf



if __name__ == '__main__':
    ###### Build vorticity field interpolator
    filename = "snapshot.npz"
    # Interpolation parameters
    interp_scales = 4  # Controls refinement of spectral sampling prior to linear interpolation
    # Build cartesian vorticity interpolator
    ω_interp = build_cartesian_vorticity_interpolator(filename, interp_scales)


    # set seed
    np.random.seed(1986) # cf Arnold 1986

    ###### Integrates streamlines for various streamline integration tolerances & settings
    # Test parameters

    N_sample = 20
    L_bounds = 1
    dL = 10

    atol = 1e-6
    maxdt = 1e-2
    mdt = int(-np.log10(maxdt))
    atol_print = int(-np.log10(atol))

    ## Sample initial conditions
    sampled = np.random.rand(3*N_sample)
    u = 2 * sampled[::3] - 1
    p = 2 * np.pi * sampled[1::3]
    r = sampled[2::3] ** (1 / 3.)

    x_0 = r * np.cos(p) * np.sqrt(1 - u * u)
    y_0 = r * np.sin(p) * np.sqrt(1 - u * u)
    z_0 = r * u

    points = [np.array((x_0[i], y_0[i], z_0[i])) for i in range(N_sample)]
    rtol = 1e-12
    
    cpu_counts = multiprocessing.cpu_count()
    
    t_start = time.time()
    print("Starting Streamline integration")
    ## integrate streamlines
    def wrap_streams(x0):
        s = Streamline(x0, L_bounds, rtol, atol, maxdt)
        s.integrate(ω_interp)
        return s
    streams = [wrap_streams(x) for x in points]
    #streams = Parallel(n_jobs=cpu_counts, prefer="threads")(delayed(wrap_streams)(x) for x in points)
    args = (N_sample, L_bounds, int(-np.log10(maxdt)), int(-np.log10(atol)))
    with open('strms_N{0}_L{1}_mdt{2}_atol{3}.pickle'.format(*args), 'wb') as file:
        pickle.dump(streams, file)
        print("Integration & pickling successful: atol={0}, max_step={1}".format(atol, maxdt))
    # load streamlines

    initial_points = []
    for s in streams:
        initial_points.append(s.x[0:1])
    
    with open('initial_strms_N{0}_L{1}_mdt{2}_atol{3}.pickle'.format(*args), 'wb') as file:
        pickle.dump(initial_points, file)
    print("Initial points saved successfully")

    print("Streamline integration complete in ", time.time() - t_start, " s")

    # split them

    L_values = np.arange(0, L_bounds + dL, dL)

    splits = []
    for s in streams:
        splits.append(split_strm(s, L_values))

    # make the list of arrays
    print("Splitting...")
    arr_list = []
    idx_list = []
    for i, tup_arr in enumerate(splits):
        idx_list.append(len(arr_list))
        s_start = streams[i].x[0:1]
        arr_list.append(tup_arr[0][0])
        arrays = [np.vstack((s_start, tup_arr[0][i])) for i in range(1, len(tup_arr[0]))]
        arr_list.extend(arrays)
    idx_list.append(-1)
    filename_bcc =  'strms_N{0}_L{1}_mdt{2}_atol{3}_dL{4}'.format(*args, dL) + "_split.bcc"
    # export bcc file

    bcc.export_closed_BCC(filename_bcc, arr_list)

    # run pyfln

    pyfln(filename_bcc)
    lnks = bcc.readtxt_certificate(filename_bcc[:-4] + '_cert.txt',
                                   sparse=True)

    # compute partial linking numbers

    links = {}
    nL = len(L_values)
    for i in range(N_sample):
        for j in range(i):
            ls_Tf = splits[i][1]
            ks_Tf = splits[j][1]
            lnkNum = 0.
            lnkNum_list = np.zeros(nL)
            norm_factor = 0.
            norm_list = np.zeros(nL)
            #lnk_block = lnks[idx_list[i]:idx_list[i+1], idx_list[j]:idx_list[j+1]]  # larger first i> j
            for a in range(nL-1): 
                for n in range(a**2, (a+1)**2): # this loop could be parallelized
                    k,l = _pairing(n)
                    #lnkNum += lnk_block[k, l]
                    lnkNum += lnks[idx_list[i]+k, idx_list[j]+l]
                norm_list[a+1] = ls_Tf[a] * ks_Tf[a]
                lnkNum_list[a+1] = lnkNum
            links[(i,j)] = {"lnk":np.array(lnkNum_list), "norm_factor":np.array(norm_list)}

    lnk_file = "lnknums_pyfln_N{0}_L{1}_mdt{2}_atol{3}_dL{4}.pickle".format(N_sample, L_bounds, mdt, atol_print, dL)
    with open(lnk_file, 'wb') as file:
        pickle.dump(links, file)
    print("Saved partial linking numbers.")


