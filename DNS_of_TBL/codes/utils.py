import numpy as np
from scipy.io import loadmat

def get_data(file_name):
    data = loadmat(file_name)
    
    sd = -301
    nd = -101
    nx = 85
    
    u = data['u'][:nx, sd:nd]
    uv = data['uv'][:nx, sd:nd]
    uu = data['uu'][:nx, sd:nd]
    vv = data['vv'][:nx, sd:nd]
    x = data['x'][:, sd:nd]
    y = data['y'][:nx]

    x, y = np.meshgrid(x, y)
    
    data = [x, y, u, uv, uu, vv]
    
    nskip = 1
    cp = np.stack([x[:, ::nskip].flatten(), y[:, ::nskip].flatten()], 1)
    
    ind_bc = np.zeros(x.shape, dtype = bool)
    ind_bc[[0, -1], ::20] = True; ind_bc[:, [0, -1]] = True
    
    bc = np.stack([d[ind_bc].flatten() for d in data], 1)
    return bc, cp



