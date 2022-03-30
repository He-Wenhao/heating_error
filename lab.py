import pandas as pd
import numpy as np

df = pd.read_csv('./results/amp_opt.csv',sep = ';',skipinitialspace=True)
df.index = df['N']

a = df['amp'][2]
b = eval(a)
print(b[0][0])