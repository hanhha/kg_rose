from collections import Counter as cnt
from ggplot import *
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd 
import numpy as np
import re
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

def visualize_data(X, Y):
  # Compute PCA
  print "Using PCA to reduce to 3D dataset"
  sigma = X.cov() # Compute covariance matrix of preprocessed data
  U, s, V = np.linalg.svd(sigma, full_matrices=True, compute_uv=True)
  Ureduce = U [:, range(0,3)]
  pa_train_set = X.dot(Ureduce).rename(columns={0:'P0',1:'P1',2:'P2'})
  variance_retained = s[range(0,3)].sum() / s.sum()
  print "Variance retained: %f" % variance_retained
  m_lbl_survived = Y['Survived'].apply(lambda e: 'Survived' if e == 1 else 'Dead')
  data_plot = pa_train_set.assign(Survived=m_lbl_survived)
  fig = pylab.figure()
  ax = Axes3D(fig)
  ax.scatter(data_plot['P0'], data_plot['P1'], data_plot['P2'])
  plt.show()
  #print data_plot
  #p = ggplot(data_plot, aes(x='P0',
  #y='P1',z='P2', color = 'factor(Survived)'))
  #p = p + geom_point() + scale_color_manual(values=["red","green"]) + ggtitle("Survived")
  #p.show()
  
#==================================
print "Reading training data ..."
#==================================
train_set = pd.read_csv ('Data/train.csv', sep=',', header=0, 
			skip_blank_lines=True, quotechar='"')

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
visualize_data (s_train_set, m_survived)

#==================================
print "Split training set to training and test sets"
#==================================
Y = m_survived['Survived'].values
X = s_train_set[list(s_train_set.columns)].values

#==================================
print "Trying multiple classifiers:"
#==================================
scores = [0] * 7
names = ["KNN", "Linear SVM", "RBF SVM", "Decision Tree","Random Forest",
		"Neural Network", "AdaBoost", "Naive Bayes"]
classifiers = pd.DataFrame([[1,2,3,4,5,6,7,8],
              names,
               [KNeighborsClassifier(11),
                 SVC(kernel="linear", C=0.025),
							 SVC(gamma=2, C=1),
							 DecisionTreeClassifier(max_depth=6),
							 RandomForestClassifier(max_depth=6, n_estimators=10,
								 max_features=None),
							 MLPClassifier(alpha=1),
							 AdaBoostClassifier(),
							 GaussianNB()],
							 scores]).T.rename(columns={0:'Idx', 1:'Name', 2:'Clf', 3:'Score'})

X_train, X_test, Y_train, Y_test = train_test_split (X, Y)

for idx, clf in classifiers.iterrows():
	print "%s" % clf['Name']
	clf['Clf'].fit(X_train, Y_train)
	clf['Score'] = clf['Clf'].score(X_test, Y_test)
	print "   score: %f" % clf['Score']
	
plt.plot (classifiers['Idx'], classifiers['Score'], 'ro')
plt.xticks(classifiers['Idx'], classifiers['Name'], rotation =
		'vertical')
plt.margins(0.2)
plt.subplots_adjust(bottom=0.3)
plt.show()

#==================================
print "Pick the most accuracy model ..."
#==================================
selected_clf      = classifiers.loc[classifiers['Score'].idxmax()]
print "%s is selected" %selected_clf['Name']

#==================================
print "Reading test data ..."
#==================================
test_set = pd.read_csv ('Data/test.csv', sep=',', header=0, 
			skip_blank_lines=True, quotechar='"')

#==================================
print "Extracting and numerizing features and then classifying test data ..."
#==================================
m_test_set = pd.DataFrame(columns=[
			'Sex','Age', 'SibSp','Parch', 'Pclass',
			'Floor'])
m_pred_survived = pd.DataFrame(columns=['PassengerId', 'Survived'])

for idx, row in test_set[['PassengerId','Sex','Age','SibSp',
			'Parch','Pclass','Cabin']].iterrows():
	cabins   = row['Cabin'].split(' ') if pd.notnull(row['Cabin']) else ['Z']
	sex      = 1 if row['Sex'] == 'female' else 0
	age      = row['Age'] if pd.notnull(row['Age']) else 0
	sibsp    = row['SibSp']
	parch    = row['Parch']
	pclass   = row['Pclass']
	pid      = row['PassengerId']

	if len(cabins) > 1:
	  tmp_predict = 0
	  location = len(m_pred_survived)
	  for i in cabins:
			floor = ord(re.findall('[A-Z]',i)[0]) - 65
			sample = pd.Series({'Pclass':pclass, 'Floor':floor, 'Sex':sex, 'Age':age, 
								'SibSp':sibsp, 'Parch':parch})
			# Normalizing and scaling data
			sample = (sample - m_train_set.mean()) / (m_train_set.max() - m_train_set.min())
			#print sample[['Sex','Age','SibSp','Parch','Pclass','Floor']].values
			if (tmp_predict == 0):
			  m_test_set.loc[location] = sample
			  tmp_predict = selected_clf['Clf'].predict(sample[['Sex','Age','SibSp','Parch','Pclass','Floor']].values.reshape(1,-1))[0]
			  m_pred_survived.loc[location]={'PassengerId':pid,'Survived':tmp_predict}
	else:
		floor = ord(re.findall('[A-Z]',cabins[0])[0]) - 65
		sample = pd.Series({'Pclass':pclass, 'Floor':floor, 'Sex':sex, 'Age':age, 
								'SibSp':sibsp, 'Parch':parch})
		# Normalizing and scaling data
		sample = (sample - m_train_set.mean()) / (m_train_set.max() - m_train_set.min())
		m_test_set.loc[len(m_test_set)] = sample
		#print sample[['Sex','Age','SibSp','Parch','Pclass','Floor']].values
		m_pred_survived.loc[len(m_pred_survived)]={'PassengerId':pid,'Survived':selected_clf['Clf'].predict(sample[['Sex','Age','SibSp','Parch','Pclass','Floor']].values.reshape(1,-1))[0]}

#==================================
print "Visulizing classified test data in 2D space"
#==================================
visualize_data (m_test_set, m_pred_survived)

#==================================
print "Writing to submission file ..."
#==================================
m_pred_survived.fillna(0).astype(int).to_csv('Data/submission.csv', index=False)