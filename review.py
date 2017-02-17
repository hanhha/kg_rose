from collections import Counter as cnt
from ggplot import *
import pandas as pd 
import numpy as np
import math
import re

def distance_pclass_fare (row, ref):
	print math.pow(row['Pclass'] -ref['Pclass'], 2) + math.pow(row['Fare'] - ref['Fare'], 2)
	return math.pow(row['Pclass'] -ref['Pclass'], 2) + math.pow(row['Fare'] - ref['Fare'], 2)

# Read training data
train_set = pd.read_csv ('Data/train.csv', sep=',', header=0, skip_blank_lines=True, quotechar='"')
M = train_set.shape [0] # Size of training set

# Predict factors:
#   - Sex (1 - Male)
#   - Age
#   - SibSp
#   - Parch
#   - Pclass
#   - Floor

m_train_set = pd.DataFrame(columns=['Sex','Age', 'SibSp','Parch', 'Pclass','Floor','Survived'])

for idx, row in train_set[['Survived','Sex','Age','SibSp','Parch','Pclass','Cabin']].iterrows():
	cabins   = row['Cabin'].split(' ') if pd.notnull(row['Cabin']) else ['Z']
	sex      = 1 if row['Sex'] == 'female' else 0
	age      = row['Age'] if pd.notnull(row['Age']) else 0
	sibsp    = row['SibSp']
	parch    = row['Parch']
	pclass   = row['Pclass']
	survived = row['Survived']

	if len(cabins) > 1:
		for i in cabins:
			floor = ord(re.findall('[A-Z]',i)[0]) - 65
			new_row = {'Pclass':pclass, 'Floor':floor, 'Sex':sex, 'Age':age, 'SibSp':sibsp, 'Parch':parch, 'Survived':survived}
			m_train_set.loc[len(m_train_set)] = new_row
	else:
		floor = ord(re.findall('[A-Z]',cabins[0])[0]) - 65
		new_row = {'Pclass':pclass, 'Floor':floor, 'Sex':sex, 'Age':age, 'SibSp':sibsp, 'Parch':parch, 'Survived':survived}
		m_train_set.loc[len(m_train_set)] = new_row
	

#pclass_fare_floor_train_set['pidx'] = pclass_fare_floor_train_set.groupby('Floor').cumcount() + 1

#p = ggplot(pclass_fare_floor_train_set, aes(x='Pclass',y='Fare',color='Floor')) 
#p = p + geom_point()
#print (p)


#avg_pclass_fare_floor = pd.DataFrame(columns=['Pclass','Fare','Floor'])
#for k in cnt(pclass_fare_floor_train_set['Floor']).keys():
#	interested = pclass_fare_cabin_train_set[pclass_fare_cabin_train_set['Cabin'] == k]
#	row = [interested['Pclass'].sum()/len(interested), interested['Fare'].sum()/len(interested), k]
#	avg_pclass_fare_cabin.loc[len(avg_pclass_fare_cabin)] = row

		
# Assume cabin for NaN cabin passengers
#m_train_set = train_set
#for idx, row in m_train_set[pd.isnull(m_train_set['Cabin'])].iterrows():
#	avg_pclass_fare_cabin[['Pclass', 'Fare']].apply (lambda ref: distance_pclass_fare(row, ref), axis = 1)


