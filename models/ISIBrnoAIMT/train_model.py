#!/usr/bin/env python

import os
import sys
from team_code import training_code

if __name__ == '__main__':
    # Parse arguments.

    os.chdir('../../')
    data_directory = './cinc-2021_data/'
    model_directory = './models/ISIBrnoAIMT/'
    training_code(data_directory, model_directory) ### Implement this function!

    print('Done.')
