#Bibs
import pandas as pd
import numpy as np
import sklearn
print('Bibliotheken erfolgreich geladen\n')



#load features
ft = np.load('FeatureData.npy')
lb = np.load('Label.npy')
print('Features, Label erfolgreich geladen')
print('Shape Feature Set: ', ft.shape)
print('Verteilung Klassen::     Normal: ',np.count_nonzero(lb==0), '     Slugging: ', np.count_nonzero(lb==1), '\n')

#Datensatz ml-bearbeitbar transformieren
X = ft.reshape(len(lb), len(ft[0,0,:])*len(ft[0,:,0]))
y = lb.reshape(len(lb))
print('Feature Set erfolgreich transofmiert\n')

#DataS splitten
from sklearn.model_selection import train_test_split
seed = 69
tr_sz = 0.7
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=tr_sz, random_state = seed)
print('Datensatz gesplittet')
print('Verteilung Klassen Train::     Normal: ',np.count_nonzero(y_train==0), '     Slugging: ', np.count_nonzero(y_train==1), '\n')
print('Verteilung Klassen Test::     Normal: ',np.count_nonzero(y_test==0), '     Slugging: ', np.count_nonzero(y_test==1), '\n')

#Train
#Tree
from sklearn import tree
clf =  tree.DecisionTreeClassifier
clf.fit(X_train, y_train)
tree.plot_tree(clf)
print('Model erfolgreich trainiert\n')

#SVC

#Eval
print('Score: ', clf.score(X_test, y_test))


from sklearn.metrics import confusion_matrix
y_pred = clf.predict(X_test)
cf_matrix = confusion_matrix(y_test, y_pred)

import seaborn as sns
import matplotlib.pyplot as plt
#confusion-matrix
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

ax.set_title('Confusion Matrix\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')

ax.xaxis.set_ticklabels(['Normal','Slugging'])
ax.yaxis.set_ticklabels(['Normal','Slugging'])

plt.show()

print('X')
print('Y')
