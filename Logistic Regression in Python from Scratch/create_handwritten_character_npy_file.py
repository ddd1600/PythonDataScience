import numpy as np
import pandas as pd
import scipy.io

#source of data: https://www.nist.gov/itl/products-and-services/emnist-dataset sorted by class -- I was able to snatch the matlab formatted database (*.mat) file
ascii_table = pd.read_excel("ascii-table.xls")














#MATLAB CONVERSION CURRENTLY ABANDONED IN FAVOR OF A SIMPLE COMPILATION APPROACH
#mat = scipy.io.loadmat("./handwritten_characters_dataset/matlab/emnist-byclass.mat")
#matstuff = list(matstuff.items()) #len(data) => 4
#FOR data
#index  desc     data
#0      header     ('__header__', b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Sun Dec 18 07:27:57 2016')
#1      version     ('__version__', '1.0')
#2      globals     ('__globals__', [])
#3      dataset    everything
#dataset = matstuff[3][1][0][0] #[1][0][0] takes us into the actual data (more headers and stuff)
#len(dataset) => 3
#FOR dataset
#index      desc            more info
#0          pictures (X     to get to an actual picture: dataset[0][0][0][0][0]





