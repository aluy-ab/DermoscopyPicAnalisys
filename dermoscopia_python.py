from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from statistics import mean
from numpy import std
import pandas as pd


#importa arquivo arff
y_true = arff.loadarff('project2.arff')
df = pd.DataFrame(y_true[0]).to_numpy()
#df.head()
#print(df)

atributos = df[:,19]
classes = df[:,0:18]
#print(atributos)
#print(classes)

atrib = atributos.reshape(-1,1)
print(atrib)

# definir o modelo/classificador
model = RandomForestClassifier(n_estimators=20)
model.fit(atrib, classes)
y_pred = model.predict(atrib)

# avaliar o modelo
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1) #definição das informações para validação 
# cruzada (cv na função abaixo)
# n_splits: nro de folds
# n_repeats: quantas vezes é feita a validação
# random_state: controla a geração de estados aleatórios para cada repetição

#para sensibilidade e especificidade
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

#---------ACURÁCIA---------

#primeira opção:
n_scoresA = cross_val_score(model, y_true, y_pred, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise') 
# n_jobs: nro de processos a serem executados em paralelo
# error_score: valor se acontecer um erro no ajuste do estimador. "raise" valor aumenta

# imprimir acurácia
print('Acuracia: %.3f (%.3f)' % (mean(n_scoresA), std(n_scoresA))) #std: standard deviation

#segunda opção:
#print('Acuracia: %.3f (%.3f)' % (((tp+tn)/(tp+tn+fp+fn)), std(n_scores)))

#------SENSIBILIDADE-------

n_scoresS = cross_val_score(model, y_true, y_pred, scoring='recall', cv=cv, n_jobs=-1, error_score='raise')

print('Sensibilidade: %.3f (%.3f)' % (mean(n_scoresS), std(n_scoresS)))

#------ESPECIFICIDADE------

#print('Especificidade: %.3f (%.3f)' % ((tn/(tn+fp)), std(n_scores)))

#------PRECISAO------

n_scoresP = cross_val_score(model, y_true, y_pred, scoring='precision', cv=cv, n_jobs=-1, error_score='raise') 

print('Precisao: %.3f (%.3f)' % (mean(n_scoresP), std(n_scoresP)))