from collections import Counter as cnt
from ggplot import *
import pandas as pd 
import numpy as np
import math
import re

#==================================
print "Reading training data ..."
#==================================
train_set = pd.read_csv ('Data/train.csv', sep=',', header=0, 
			skip_blank_lines=True, quotechar='"')
M = train_set.shape [0] # Size of training set

# Predict factors:
#   - Sex (1 - Male)
#   - Age
#   - SibSp
#   - Parch
#   - Pclass
#   - Floor

#==================================
print "Extracting and numerizing features ..."
#==================================
m_train_set = pd.DataFrame(columns=[
			'Sex','Age', 'SibSp','Parch', 'Pclass',
			'Floor'])
m_survived = pd.DataFrame(columns=['Survived'])

for idx, row in train_set[['Survived','Sex','Age','SibSp',
			'Parch','Pclass','Cabin']].iterrows():
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
			new_row = {'Pclass':pclass, 'Floor':floor, 'Sex':sex, 'Age':age, 
								'SibSp':sibsp, 'Parch':parch}
			m_train_set.loc[len(m_train_set)] = new_row
			m_survived.loc[len(m_survived)] = {'Survived':"Survived"} if survived == 1 else {'Survived':"Dead"}
	else:
		floor = ord(re.findall('[A-Z]',cabins[0])[0]) - 65
		new_row = {'Pclass':pclass, 'Floor':floor, 'Sex':sex, 
							'Age':age, 'SibSp':sibsp, 'Parch':parch}
		m_train_set.loc[len(m_train_set)] = new_row
		m_survived.loc[len(m_survived)] = {'Survived':"Survived"} if survived == 1 else {'Survived':"Dead"}

#)==================================
print "Visulizing training data in 2D space"
#==================================
# Normalizing and scaling data
s_train_set = (m_train_set - m_train_set.mean()) / (m_train_set.max() -
		m_train_set.min())
sigma = s_train_set.cov() # Compute covariance matrix of preprocessed data
U, s, V = np.linalg.svd(sigma, full_matrices=True, compute_uv=True)
Ureduce = U [:, range(0,2)]
pa_train_set = s_train_set.dot(Ureduce).rename(columns={0:'P0',1:'P1'})

# Visualize Principal Component of traning set
p = ggplot(pa_train_set.assign(Survived=m_survived['Survived']), aes(x='P0',
y='P1', color = 'factor(Survived)'))
p = p + geom_point() + scale_color_manual(values=["red","green"]) + ggtitle("Survived")
p.show()

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


