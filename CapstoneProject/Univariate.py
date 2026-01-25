import pandas as pd
import numpy as np
class Univariate():

    def quanQual(dataset):
        quan=[]
        qual=[]
        for columnName in dataset.columns:
            if(dataset[columnName].dtype=='O'):
                quan.append(columnName)
            else:
                qual.append(columnName)

        return quan,qual

    def IQR(quan,dataset):
        descrptive=pd.DataFrame(index=["Mean","Median","Mode","Q1:25%","Q2:50%","Q3:75%","99%","Q4:100%",
                                   "IQR","1.5Rule","Lesser","Greater","Min","Max","Kurtosis","Skew","Variance",
                                    "StandardDeviation"],columns=quan)
        for columnName in quan:
            descrptive[columnName]["Mean"]=dataset[columnName].mean()
            descrptive[columnName]["Median"]=dataset[columnName].median()
            descrptive[columnName]["Mode"]=dataset[columnName].mode()[0]
            descrptive[columnName]["Q1:25%"]=dataset.describe()[columnName]["25%"]
            descrptive[columnName]["Q2:50%"]=dataset.describe()[columnName]["50%"]
            descrptive[columnName]["Q3:75%"]=dataset.describe()[columnName]["75%"]
            descrptive[columnName]["99%"]=np.percentile(dataset[columnName],99)
            descrptive[columnName]["Q4:100%"]=dataset.describe()[columnName]["max"]
            descrptive[columnName]["IQR"]=descrptive[columnName]["Q3:75%"]-descrptive[columnName]["Q1:25%"]
            descrptive[columnName]["1.5Rule"]=1.5*descrptive[columnName]["IQR"]
            descrptive[columnName]["Lesser"]=descrptive[columnName]["Q1:25%"]-descrptive[columnName]["1.5Rule"]
            descrptive[columnName]["Greater"]=descrptive[columnName]["Q3:75%"]+descrptive[columnName]["1.5Rule"]
            descrptive[columnName]["Min"]=dataset[columnName].min()
            descrptive[columnName]["Max"]=dataset[columnName].max()
            descrptive[columnName]["Kurtosis"]=dataset[columnName].kurtosis()
            descrptive[columnName]["Skew"]=dataset[columnName].skew()
            descrptive[columnName]["Variance"]=dataset[columnName].var()
            descrptive[columnName]["StandardDeviation"]=dataset[columnName].std()
        return descrptive

    def findOutlier(quan,descrptive):
        lesser=[]
        greater=[]
        for columnName in quan:
            if(descrptive[columnName]["Min"]<descrptive[columnName]["Lesser"]):
                lesser.append(columnName)
            if(descrptive[columnName]["Max"]>descrptive[columnName]["Greater"]):
                greater.append(columnName)
        return lesser,greater

    def replaceOutlier(lesser,greater,dataset,descrptive):
        for columnName in lesser:
            dataset[columnName][dataset[columnName]<descrptive[columnName]["Lesser"]]=descrptive[columnName]["Lesser"]    
                    
        for columnName in greater:
            dataset[columnName][dataset[columnName]>descrptive[columnName]["Greater"]]=descrptive[columnName]["Greater"]
        return 

    def freqTable(columnName,dataset):
        freqTable=pd.DataFrame(columns=["Unique_Values","Frequency","Relative_Frequency","Cumsum"])
        freqTable["Unique_Values"]=dataset["columns"].value_counts().index
        freqTable["Frequency"]=dataset["columns"].value_counts().values
        freqTable["Relative_Frequency"]=(freqTable["Frequency"]/103)
        freqTable["Cumsum"]=freqTable["Relative_Frequency"].cumsum()
        return freqTable