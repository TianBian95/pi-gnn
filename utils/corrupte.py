import sys,os
sys.path.append(os.getcwd())
import numpy as np

def uniform_mix_C(mixing_ratio, num_classes):
    '''
    returns a linear interpolation of a uniform matrix and an identity matrix
    '''
    return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
        (1 - mixing_ratio) * np.eye(num_classes)

def uniform_mix_C_revised(mixing_ratio, num_classes):
    '''
    returns a linear interpolation of a uniform matrix and an identity matrix
    '''

    off_diagnal= mixing_ratio * np.full((num_classes, num_classes), 1 / (num_classes-1))
    np.fill_diagonal(off_diagnal, 0)
    data = np.eye(num_classes) * (1 - mixing_ratio) + off_diagnal
    return data
    # return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
    #     (1 - mixing_ratio) * np.eye(num_classes)


def flip_labels_C(corruption_prob, num_classes, seed=1):
    '''
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    '''
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
    return C