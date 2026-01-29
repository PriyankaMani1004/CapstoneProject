import pandas as pd
from sklearn.model_selection import train_test_split 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pickle
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xg
import lightgbm as lg
from xgboost import XGBClassifier
from lightgbm  import LGBMClassifier

class FeatureSelection():
    def rfeFeature(indep_X,dep_Y,n):
            rfelist=[]
            
            log_model = LogisticRegression(solver='lbfgs')
            svc_model = SVC(kernel = 'linear', random_state = 0)
            RF = RandomForestClassifier(criterion = 'entropy', random_state = 0)      
            DT= DecisionTreeClassifier(criterion = 'gini',splitter='best',random_state = 0)            
            xgb_model = XGBClassifier( learning_rate=0.1,n_estimators=200, random_state=0)
            lgb_model = LGBMClassifier(learning_rate=0.1, n_estimators=200,random_state=0)
    
            rfemodellist=[log_model,svc_model,xgb_model,RF,DT,lgb_model] 
            for i in   rfemodellist:
                print(i)
                log_rfe = RFE(estimator=i, n_features_to_select=n)
                log_fit = log_rfe.fit(indep_X, dep_Y)
                log_rfe_feature=log_fit.transform(indep_X)
                rfelist.append(log_rfe_feature)
            return rfelist
    
    def split_scalar(indep_X,dep_Y):
            X_train, X_test, y_train, y_test = train_test_split(indep_X, dep_Y, test_size = 0.25, random_state = 0)
                    
            #Feature Scaling
            #from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            
            return X_train, X_test, y_train, y_test
    
    def cm_prediction(classifier,X_test):
         y_pred = classifier.predict(X_test)
            
            # Making the Confusion Matrix
         from sklearn.metrics import confusion_matrix
         cm = confusion_matrix(y_test, y_pred)
            
         from sklearn.metrics import accuracy_score 
         from sklearn.metrics import classification_report 
            #from sklearn.metrics import confusion_matrix
            #cm = confusion_matrix(y_test, y_pred)
            
         Accuracy=accuracy_score(y_test, y_pred )
            
         report=classification_report(y_test, y_pred)
         return  classifier,Accuracy,report,X_test,y_test,cm
    
    def logistic(X_train,y_train,X_test):       
            
            from sklearn.linear_model import LogisticRegression
            classifier = LogisticRegression(random_state = 0)
            classifier.fit(X_train, y_train)
            classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test)
            return  classifier,Accuracy,report,X_test,y_test,cm  
    
    def svm_linear(X_train,y_train,X_test):
                    
            from sklearn.svm import SVC
            classifier = SVC(kernel = 'linear', random_state = 0)
            classifier.fit(X_train, y_train)
            classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test)
            return  classifier,Accuracy,report,X_test,y_test,cm
        
    def svm_NL(X_train,y_train,X_test):
                    
            from sklearn.svm import SVC
            classifier = SVC(kernel = 'rbf', random_state = 0)
            classifier.fit(X_train, y_train)
            classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test)
            return  classifier,Accuracy,report,X_test,y_test,cm
    
    def Decision(X_train,y_train,X_test):
            
            from sklearn.tree import DecisionTreeClassifier
            classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
            classifier.fit(X_train, y_train)
            classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test)
            return  classifier,Accuracy,report,X_test,y_test,cm  
    
    def random(X_train,y_train,X_test):
            
            from sklearn.ensemble import RandomForestClassifier
            classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
            classifier.fit(X_train, y_train)
            classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test)
            return  classifier,Accuracy,report,X_test,y_test,cm
    
    def Xgb(X_train,y_train,X_test):
            
            from sklearn.ensemble import XGBClassifier
            classifier = XGBClassifier(max_depth=5,learning_rate=0.05,n_estimators=500, eval_metric='auc', random_state = 0)
            classifier.fit(X_train, y_train)
            classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(classifier,X_test)
            return  classifier,Accuracy,report,X_test,y_test,cm
    
    def Lgb(X_train,y_train,X_test):
        from sklearn.ensemble import LGBClassifier
        classifier=LGBClassifier(learning_rate=0.05, n_estimators=500,objective='binary', metric='auc',random_state=0)
        classifier.fit(X_train,y_train)
        classifier,Accuracy,report,X_test,y_test,cm=cm_prediction(Classifier,X_test)
        return  classifier,Accuracy,report,X_test,y_test,cm
    
    def rfe_classification(acclog,accsvml,accsvmnl,accdes,accrf,accxgb,acclgb): 
        
        rfedataframe=pd.DataFrame(index=['Logistic','SVC','Random','DecisionTree','XGBClassifier','LGBClassifier'],columns=['Logistic','SVMl','SVMnl',
                                                                                            'Decision','Random','XGB','LGB'])
    
        for number,idex in enumerate(rfedataframe.index):
            
            rfedataframe['Logistic'][idex]=acclog[number]       
            rfedataframe['SVMl'][idex]=accsvml[number]
            rfedataframe['SVMnl'][idex]=accsvmnl[number]       
            rfedataframe['Decision'][idex]=accdes[number]
            rfedataframe['Random'][idex]=accrf[number]
            rfedataframe['XGB'][idex]=accxgb[number]
            rfedataframe['LGB'][idex]=acclgb[number]
        return rfedataframe
