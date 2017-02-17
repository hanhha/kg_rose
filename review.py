import pandas as pd 
import numpy as np
from collections import Counter as cnt

# Read training data
train_set = pd.read_csv ('Data/train.csv', sep=',', header=0, skip_blank_lines=True, quotechar='"')
M = train_set.shape [0] # Size of training set

for k in train_set:
	print "Variable %s:" % k
	count_val = cnt(train_set[k])
	print "NaN: %d" % count_val[np.nan]
	#for kv in count_val:
	#	print "'---- %s: %d" % (kv, count_val[kv])  
