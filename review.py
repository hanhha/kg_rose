from collections import Counter as cnt
from ggplot import *
import pandas as pd 
import numpy as np
import math
import re
import sklearn as skl
from sklearn.neural_network import MLPClassifier as NN
from sklearn.metrics import classification_report,confusion_matrix

def sigmoid(x):
  return 1 / (1 + np.ex(-x))
  
def dsigmoid(y):
  return y * (1.0 - y)

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
			m_survived.loc[len(m_survived)] = {'Survived':survived}
	else:
		floor = ord(re.findall('[A-Z]',cabins[0])[0]) - 65
		new_row = {'Pclass':pclass, 'Floor':floor, 'Sex':sex, 
							'Age':age, 'SibSp':sibsp, 'Parch':parch}
		m_train_set.loc[len(m_train_set)] = new_row
		m_survived.loc[len(m_survived)] = {'Survived':survived}

# Normalizing and scaling data
s_train_set = (m_train_set - m_train_set.mean()) / (m_train_set.max() -
		m_train_set.min())

#==================================
print "Visulizing training data in 2D space"
#==================================
# Compute PCA
print "Using PCA to reduce to 2D dataset"
sigma = s_train_set.cov() # Compute covariance matrix of preprocessed data
U, s, V = np.linalg.svd(sigma, full_matrices=True, compute_uv=True)
Ureduce = U [:, range(0,2)]
pa_train_set = s_train_set.dot(Ureduce).rename(columns={0:'P0',1:'P1'})
variance_retained = s[range(0,2)].sum() / s.sum()
print "Variance retained: %f" % variance_retained
m_lbl_survived = m_survived.applymap(lambda e: 'Survived' if e == 1 else 'Dead')
p = ggplot(pa_train_set.assign(Survived=m_lbl_survived['Survived']), aes(x='P0',
y='P1', color = 'factor(Survived)'))
p = p + geom_point() + scale_color_manual(values=["red","green"]) + ggtitle("Survived")
p.show()

#==================================
print "Split training set to training and test sets"
#==================================
Y = m_survived['Survived'].values
X = s_train_set[list(s_train_set.columns)].values

#==================================
print "Training Neural Network"
#==================================
X_train, X_test, Y_train, Y_test = skl.model_selection.train_test_split (X, Y)
mlp = NN(hidden_layer_sizes=(10,10))
mlp.fit(X_train, Y_train)
predicts = mlp.predict(X_test)
print (confusion_matrix(Y_test, predicts))
print (classification_report(Y_test, predicts))
