import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def plot_discharge(data,
                    save_fig=False,
                    fig_filename='discharge.png',
                    dpi=150,
                    close_fig=False):

    etaMIN = -9.
    etaMAX = 1.

    eta = data['eta']
    qw = data['qw']

    cmap = mpl.cm.get_cmap('Greys')

    eta_index = 1 - ((eta - etaMIN) / (etaMAX - etaMIN))
    eta_rgb = np.array(map(cmap, eta_index))[:,:,:3]

    min_qw = 5. * 1. * 0.01

    kk = np.minimum(1, qw / (min_qw * 100.))

    IMG = eta_rgb.copy()

    IMG[:,:,0] = np.minimum(1, (1-kk) * IMG[:,:,0])
    IMG[:,:,1] = np.minimum(1, (0.72*kk + (1-kk)) * IMG[:,:,1])
    IMG[:,:,2] = np.minimum(1, (kk + (1-kk)) * IMG[:,:,2])

    IMG[qw < min_qw] = eta_rgb[qw < min_qw]
    
    plt.figure(figsize=(10,10))
    plt.imshow(IMG)
    
    if save_fig:
        plt.savefig(fig_filename, dpi = dpi)
        plt.show()
        
    if close_fig:
        plt.close()