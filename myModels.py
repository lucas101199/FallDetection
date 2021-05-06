import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

#chargement des données avec pandas
feature_norm = pd.read_csv("fichier.csv")

#transformation en matrice numpy - seul reconnu par scikit-learn
data = feature_norm.values
#X matrice des var. explicatives
X = data[:,0:4]
#y vecteur de la var. à prédire
y = data[:4,]

#subdivision des données éch_train et éch_test
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

#création d'une instance des différentes models
RandomForest = RandomForestClassifier(random_state=0)
AdaBoost =  AdaBoostClassifier(random_state=0)
SVM =  SVC(random_state=0)
KNN = KNeighborsClassifier()

#dictionnaire des différents models utilisé.
dict_of_models = {'RandomForest': RandomForest,
                  'AdaBoost' : AdaBoost,
                  'SVM': SVM,
                  'KNN': KNN
                 }

#la méthode d'évaluation des différentes model avec un rapport de classification
def evaluation(model):
  #exécution de l'instance sur les données d'apprentissage  
  model.fit(X_train, y_train)
  #la prediction sur les données de test
  ypred = model.predict(X_test)
    
  #matrice de confusion
  #confrontation entre Y obs. sur l’éch. test et la prédiction
  cm = confusion_matrix(y_test,y_pred)
  print(cm)
  #taux de succès
  acc = accuracy_score(y_test,y_pred)
  print(acc)
  print(classification_report(y_test, ypred))
    
    
  plt.figure(figsize=(12, 8))
  plt.plot(N, train_score.mean(axis=1), label='train score')
  plt.plot(N, val_score.mean(axis=1), label='validation score')
  plt.legend()
    

for name, model in dict_of_models.items():
    print(name)
    evaluation(model)

#fonfion de prediction si le sujet est tombé ou non.
def fallOrNotFall(model, TimeStamp,X_Axis,Y_Axis,Z_Axis):
  x = np.array([TimeStamp,X_Axis,Y_Axis,Z_Axis]).reshape(1, 4)
  print(model.predict(x))
  print(model.predict_proba(x))