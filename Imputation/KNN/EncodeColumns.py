
# coding: utf-8

# In[26]:

import numpy as np
import pandas as pd
from pandas import DataFrame as df
from sklearn.feature_extraction import DictVectorizer as DV


def encodeDF(df, fileName) :
    encodedDF = []
    for column in df.columns:
        print(column)
        if column in listOfColumnsToEncode :
            print(" ** Encode Column")
            tmp = pd.get_dummies(df[[column]])
            # condition for native-country column, since it has a missmatch in test & train data
            if(column == 'native-country') :
                tmp = tmp.reindex(columns=nativeCountryColumnList, fill_value=0)
        else :
            tmp = df[[column]]
        
        if column not in duplicateColumnList :
            encodedDF.append(tmp)
        
    encodedDF = pd.concat(encodedDF, axis=1)

    # Encode the Class Lable (Target variable)
    encodedDF[targetLabelToEncode] = encodedDF[targetLabelToEncode].astype('category')
    encodedDF[targetLabelToEncode] = encodedDF[targetLabelToEncode].cat.codes

    #print(len(encodedDF.columns))
    #encodedDF.head(20)

    encodedDF.to_csv("Encoded-"+fileName, index=False)

listOfColumnsToEncode = ["WorkClass", "marital-status", "occupation", "relationship", 
                         "race", "sex", "native-country"]
duplicateColumnList = ["education"]
targetLabelToEncode = "Class"
nativeCountryColumnList = ['native-country_ United-States', 'native-country_ Cambodia', 'native-country_ Canada', 
                           'native-country_ China', 'native-country_ Columbia', 'native-country_ Cuba', 
                           'native-country_ Dominican-Republic', 'native-country_ Ecuador', 'native-country_ El-Salvador', 
                           'native-country_ England', 'native-country_ France', 'native-country_ Germany', 
                           'native-country_ Greece', 'native-country_ Guatemala', 'native-country_ Haiti', 
                           'native-country_ Holand-Netherlands', 'native-country_ Honduras', 'native-country_ Hong', 
                           'native-country_ Hungary', 'native-country_ India', 'native-country_ Iran', 
                           'native-country_ Ireland', 'native-country_ Italy', 'native-country_ Jamaica', 
                           'native-country_ Japan', 'native-country_ Laos', 'native-country_ Mexico', 
                           'native-country_ Nicaragua', 'native-country_ Outlying-US(Guam-USVI-etc)', 
                           'native-country_ Peru', 'native-country_ Philippines', 'native-country_ Poland', 
                           'native-country_ Portugal', 'native-country_ Puerto-Rico', 'native-country_ Scotland', 
                           'native-country_ South', 'native-country_ Taiwan', 'native-country_ Thailand', 
                           'native-country_ Trinadad&Tobago', 'native-country_ Vietnam', 'native-country_ Yugoslavia']


filesToEncode = ["census-income.data_WithImputation_UsingKNNClutering.csv", 
                 "census-income.data_WithImputation_UsingRandomForest.csv", 
                 "census-income.test_WithImputation_UsingKNNClutering.csv", 
                 "census-income.test_WithImputation_UsingRandomForest.csv"]

for file in filesToEncode :
    df = pd.read_csv(file)
    encodeDF(df,file)

