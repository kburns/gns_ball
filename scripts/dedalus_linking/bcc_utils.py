import numpy as np
import scipy.sparse as sp

def _create_header(Nc, TC):
    """
    Creates bcc headers. 
    
    Inputs:
    
      Nc: int (must fit in 32 bit unsigned). 
      Number of curves to store
      
      TC:int (must fit in 32 bit unsigned).
      Total number of control points 
      
    """
    
    file_info = bytes(40)
    header = (bytes("BCC", "ascii") # BCC File header
          + (0x44).to_bytes(1, byteorder='little')  # 32-bit ints and floats
          + bytes("PL", 'ascii') # curve format: polylines
          + (3).to_bytes(1, byteorder='little') # number of dimensions
          + (2).to_bytes(1, byteorder='little') # up direction is z
          + (Nc).to_bytes(8, byteorder='little', signed=False) # total number of curves as unsigned 64-bit int
          + (TC).to_bytes(8, byteorder='little', signed=False) # total number of control points as unsigned 64-bit int
          + file_info
         )
    if len(header) != 64:
        raise ValueError("Invalid header")
    return header

def _create_datablock(curves, closed=True):
    """
    BCC file export helper function: Returns binary datablock compatible with .bcc format 
    from a list of (Nx3) or (3xN) numpy float arrays. numpy arrays are downcasted to single precision floats.
    
    arr_list: list of (N x 3) or (3 x N) numpy array. 
    
    closed: bool. Defaults to True
        Indicates whether the curves are closed;
    
    """
    
    content_stream = bytes(0)
    for i, curve in enumerate(curves):
        if curve.shape[1] != 3:
            if curve.shape[0] == 3:
                curve = curve.T
            else:
                raise ValueError("Invalid curve shape")
        curve = curve.astype(np.float32)
        if closed:
            content_stream += (-curve.shape[0]).to_bytes(4,byteorder='little', signed=True)
        else:
            content_stream += (curve.shape[0]).to_bytes(4,byteorder='little', signed=True)
        content_stream += curve.ravel().tobytes() # implicitely uses 'C' memory layout order 
    return content_stream
    
def export_closed_BCC(filename, arr_list):
    """
    Exports collection of closed curves represented by a list of numpy arrays to the binary .bcc format
    
    filename: str
        output will be written to filename. if .bcc extension is not already present it is added.
        
    arr_list: collection of 2d numpy array with one dimension 3
        Closed curves to output
    """
    
    Nc = len(arr_list)
    TC = 0 # total number of curve points
    for curve in arr_list:
        if curve.shape[1] != 3:
            if curve.shape[0] == 3:
                curve = curve.T
            else:
                raise ValueError("Invalid curve shape")
        TC += curve.shape[0]
                
    binary = _create_header(Nc, TC) + _create_datablock(arr_list)
    if filename[-4:] != '.bcc':
        filename += '.bcc'
    with open(filename, "wb") as file:
        for byte in binary:
            file.write(byte.to_bytes(1, byteorder='little'))
    print("{0} curves with {1} total points succesfully exported to bcc".format(Nc, TC))
    
def read_BCC(filename):
    if filename[-4:] != '.bcc':
        filename += '.bcc'
    with open(filename, "rb") as file:
        head = file.read(64)
        
        if head[0:3] != b'BCC':
            raise ValueError("File is not a Binary Curve Collection (BCC)")

        Ncurves = int.from_bytes(head[8:16], byteorder="little", signed=False)
        Tot_points = int.from_bytes(head[16:24], byteorder="little", signed=False)

        is_closed = np.zeros(Ncurves)
        len_curves = np.zeros(Ncurves, dtype=int)
        list_curves = []
        for i in range(Ncurves):
            print(i)
            len_curves[i] = int.from_bytes(file.read(4), byteorder="little", signed=True)
            if len_curves[i] < 0: # if curve is closed
                is_closed[i] = 1
                len_curves[i] *= -1
            curve = np.frombuffer(file.read(12*len_curves[i]), dtype=np.float32, count=-1)
            list_curves.append(np.reshape(curve, (len_curves[i], 3)))
        chunk = file.read(1) 
        if chunk != b'':
            raise ValueError("Error loading")
        return list_curves, is_closed
    
def readtxt_certificate(filename, sparse=False):
    if filename[-4:] != '.txt':
        filename += '.txt'
    with open(filename, 'r') as file:
        Ncurve = int(file.readline())
        if sparse:
            lnks = sp.dok_matrix((Ncurve, Ncurve), dtype=np.int32)
        else:
            lnks = np.zeros((Ncurve, Ncurve))
        for l in file:
            i,j, lnk = l.split(',')
            i = int(i)
            j = int(j)
            if i < j:
                i,j = j,i
            lnks[i, j] = int(lnk)
    return lnks