#!/usr/bin/env python

import os
import sys
from team_code import training_code

if __name__ == '__main__':
    # Parse arguments.
    # if len(sys.argv) != 3:
    #     raise Exception('Include the data and model folders as arguments, e.g., python train_model.py data model.')
    #
    # data_directory = sys.argv[1]
    # model_directory = sys.argv[2]
    os.chdir('../../')
    data_directory = './cinc-2021_data/'
    model_directory = './models/ISIBrnoAIMT/'
    training_code(data_directory, model_directory) ### Implement this function!

    print('Done.')
