#!/usr/bin/env python

import os
import sys
from team_code import training_code

if __name__ == '__main__':
    # Parse arguments.

    os.chdir('../../')
    data_directory = './data_cinc-2021/'
    model_directory = './models/m01_classification_ISIBrnoAIMT/'
    training_code(data_directory, model_directory) ### Implement this function!

    print('Done.')
