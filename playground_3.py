import csv

import numpy as np

if __name__ == '__main__':
    header, data = np.loadtxt('data/Cal_fire.csv', delimiter=',', usecols=(1,2))