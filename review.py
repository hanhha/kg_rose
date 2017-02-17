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

# Figure cabin distribution depends on Pclass and Fare
pclass_fare_cabin_train_set = pd.DataFrame(columns=['Pclass','Fare','Cabin'])
avg_pclass_fare_cabin = pd.DataFrame(columns=['Pclass','Fare','Cabin'])
for idx, row in train_set[pd.notnull(train_set['Cabin'])][['Pclass','Fare','Cabin']].iterrows():
	cabins = row['Cabin'].split(' ')
	if len(cabins) > 1:
		for i in cabins:
			new_row = [row['Pclass'], row['Fare']/len(cabins), i]
			pclass_fare_cabin_train_set.loc[len(pclass_fare_cabin_train_set)] = new_row
	else:
		pclass_fare_cabin_train_set.loc[len(pclass_fare_cabin_train_set)] = row

for k in cnt(pclass_fare_cabin_train_set['Cabin']).keys():
	interested = pclass_fare_cabin_train_set[pclass_fare_cabin_train_set['Cabin'] == k]
	row = [interested['Pclass'].sum()/len(interested), interested['Fare'].sum()/len(interested), k]
	avg_pclass_fare_cabin.loc[len(avg_pclass_fare_cabin)] = row

		
