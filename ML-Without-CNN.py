#import keras
import cv2
import os
import numpy as np

X = []
Y = []

path = r"C:\Users\Ronith\Desktop\Data Science 2020\Projects\Hand Sign\Final_Project\Hand Signs"

for i in os.listdir(path) :
    
    np_path = r"C:\Users\Ronith\Desktop\Data Science 2020\Projects\Hand Sign\Final_Project\Hand Signs\{}".format(i)
    
    for ii in os.listdir(np_path) :
        
        filename = r"C:\Users\Ronith\Desktop\Data Science 2020\Projects\Hand Sign\Final_Project\Hand Signs\{}\{}".format(i,ii)
        
        img = cv2.imread(filename, 0)

        img = cv2.resize(img,(64,64))
        
        X.append(np.asarray(img).flatten())
        
        #X.append(img.reshape(1,64,64,1))
        
        Y.append(i)
        
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)


# Classification :

# Catboost
from catboost import CatBoostClassifier
classifier1 = CatBoostClassifier()
classifier1.fit(X_train, Y_train)

pred1 = classifier1.predict(X_test)
 
classifier1.save_model("Catboost")


# XGBoost
from xgboost import XGBClassifier
classifier2 = XGBClassifier()
classifier2.fit(X_train, Y_train)

pred2 = classifier2.predict(X_test)

classifier2.save_model("XGBoost")


# LightGBM
import lightgbm as lgb
d_train = lgb.Dataset(X_train, label = Y_train)
params = {}
params['learning_rate'] = 0.125
params['boosting_type'] = 'gbdt'
params['objective'] = 'multiclass'
params['metric'] = 'multi_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 5
params['min_data'] = 50
params['max_depth'] = 8
params["num_class"] = 230
classifier3 = lgb.train(params, d_train, 1000)

pred_lgb = classifier3.predict(X_test)

pred3 = []

for i in (pred_lgb) :
    j = 0
    xo = 1
    while xo == 1:
        if i[j] == i.max():
            pred3.append(j)
            xo = 0
        else :
            j = j + 1


# Naive_Bayes
from sklearn.naive_bayes import GaussianNB
classifier4 = GaussianNB()
classifier4.fit(X_train, Y_train)

pred4 = classifier4.predict(X_test)


# KNearistNegihbor
from sklearn.neighbors import KNeighborsClassifier
classifier5 = KNeighborsClassifier(n_neighbors = 1, metric = "minkowski")
classifier5.fit(X_train, Y_train)

pred5 = classifier5.predict(X_test)


# Decison Tree :
from sklearn.tree import DecisionTreeClassifier
classifier6 = DecisionTreeClassifier()
classifier6.fit(X_train, Y_train)

pred6 = classifier6.predict(X_test)


# Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier7 = RandomForestClassifier()
classifier7.fit(X_train, Y_train)

pred7 = classifier7.predict(X_test)


# ANN
from keras.models import Sequential
from keras.layers import Dense , Dropout

classifier8 = Sequential()
classifier8.add(Dense(output_dim = 512, input_dim = 8, activation = "relu"))
classifier8.add(Dropout(0.25))
classifier8.add(Dense(output_dim = 512, activation = "relu"))
classifier8.add(Dropout(0.5))
classifier8.add(Dense(output_dim = 1, activation = 'sigmoid'))

classifier8.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ["accuracy"])
classifier8.fit(X_train,Y_train,batch_size = 32, epochs = 10)

pred8 = classifier8.predict(X_test)


pred = [pred1,pred2,pred3,pred4,pred5,pred6,pred7]
# Accuracy :
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements

acc = []
from sklearn.metrics import confusion_matrix
for i in pred :
    cm = confusion_matrix(Y_test, i)
    acc.append(accuracy(cm))