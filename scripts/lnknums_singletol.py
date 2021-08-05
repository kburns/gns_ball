import numpy as np
import pickle
from src.lnknum import compute_partial_link_DS
import time

if __name__ == '__main__':
    """ Computes partial linking numbers of 50 test streamlines for a given tolerance of the streamline integrator."""

    N_sample = 50
    L_bounds = 100

    mdt = 2
    atol = 6
    dL = 1

    print("Starting mdt={0}, atol={1}...".format(mdt,atol))
    t0 = time.time()
    filename = 'longT/strms2_N{0}_L{1}_mdt{2}_atol{3}.pickle'.format(N_sample,
                                                        L_bounds,
                                                        mdt,
                                                        atol)
    with open(filename, 'rb') as file:
        streams = pickle.load(file)
        print("Pickling Successful for mdt={0}, atol={1}".format(mdt, atol))
    # run linking numbers
    Ls = np.arange(0,L_bounds + dL,dL)
    lnk = np.zeros((N_sample, N_sample, len(Ls)))
    nfs = np.zeros((N_sample, N_sample, len(Ls)))

    save_filename = "longT/lnk_nums_N{0}_L{1}_mdt{2}_atol{3}".format(N_sample, L_bounds, mdt, atol)
    with open(save_filename, 'wb') as file:
        pickle.dump((lnk, nfs, Ls), file)
    count = 0
    tot = N_sample *(N_sample - 1)/2
    for i in range(1, N_sample):
        for j in range(i):
            t1 = time.time()
            lnk[i, j] = compute_partial_link_DS(streams[i], 
                                                streams[j],
                                                (i, j),
                                                save_filename,
                                                dL)
            count += 1
            print("Streamline #{0}/{1} done in {2:0.2f} s".format(count, tot, time.time() - t0))
    print("Finished mdt={0}, atol={1} in {2} seconds.".format(mdt,atol, time.time()-t0))