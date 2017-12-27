import time
import gc

import pandas as pd
import numpy as np
#from sklearn import cross_validation, linear_model
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
import matplotlib.pyplot as plt
import category_encoders as ce
from examples.source_data.loaders import get_mushroom_data, get_cars_data, get_splice_data

plt.style.use('ggplot')

#pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

workDir = "./EncodeTest/"
dataSetList = ["train_imputed_knn.csv","test_imputed_knn.csv","train_imputed_knn_zscore.csv","test_imputed_knn_zscore.csv",
               "train_imputed_randomforest.csv","test_imputed_randomforest.csv"]
#dataSetList = ["train_imputed_knn.csv", "test_imputed_knn.csv"]
columnHeader = ["Age", "WorkClass", "fnlwgt", "education", "educationnum", "maritalstatus", "occupation", "relationship", 
                "race", "sex", "capitalgain", "capitalloss", "hoursperweek", "nativecountry", "Class"]
targetLabel = "Class"
duplicateColumn = ["education"]
columnsToEncode = ["WorkClass", "maritalstatus", "occupation", "relationship", "race", "sex", "nativecountry"]

def writeEncodedData(Xnew, ynew):
    print("In method")
    result = pd.concat([Xnew, ynew], axis=1)
    result.to_csv("encoded.csv")

#This Takes in a classifier that supports multiclass classification, and X and a y, and returns a cross validation score.
def score_models(clf, X, y, encoder, encoder_name, dataSetName):
    scores = []

    runs=1
    X_test = None
    for _ in range(runs):
        #X_test = encoder().fit_transform(X, y)
        encoder.fit(X, y)
        X_test = encoder.transform(X)
       
        # Saving encoded file with the target label (function is not working here, Dont know why?)
        result = pd.concat([X_test, y], axis=1)
        fileName = workDir + encoder_name + "_" + dataSetName
        result.to_csv(fileName, index=False)
        # Saving end. 
        #writeEncodedData(X_test, y[0])
        
        #scores.append(cross_validation.cross_val_score(clf, X_test, y[0], n_jobs=1, cv=5))
        score = cross_val_score(clf, X_test, y, n_jobs=-1, cv=5)
        #score = []
        print(score)
        scores.append(score)
        print(scores)
        gc.collect()

    scores_ = [y for z in [x for x in scores] for y in z]
    print(score)
    print(scores_)

    return float(np.mean(scores)), float(np.std(scores)), scores, X_test.shape[1]

#def main(loader, name):
#Iterate through the datasets and score them with a classifier using different encodings.
def main(dataSetName, X, y):    

    scores = []
    raw_scores_ds = {}

    # Loading logistic regression classifier
    clf = linear_model.LogisticRegression()

    # try every encoding method available
    #encoders = ce.__all__
    encoders = ["BackwardDifferenceEncoder", "BinaryEncoder", "HashingEncoder", "HelmertEncoder", "OneHotEncoder", 
                "OrdinalEncoder", "SumEncoder", "PolynomialEncoder", "BaseNEncoder", "LeaveOneOutEncoder"]
    print(encoders)

    for encoder_name in encoders:
        print(encoder_name)
        if(encoder_name == "BackwardDifferenceEncoder"):
            encoder = ce.BackwardDifferenceEncoder(cols=columnsToEncode)
        if(encoder_name == "BinaryEncoder"):
            encoder = ce.BinaryEncoder(cols=columnsToEncode)
        if(encoder_name == "HashingEncoder"):
            encoder = ce.HashingEncoder(cols=columnsToEncode)
        if(encoder_name == "HelmertEncoder"):
            encoder = ce.HelmertEncoder(cols=columnsToEncode)
        if(encoder_name == "OneHotEncoder"):
            encoder = ce.OneHotEncoder(cols=columnsToEncode)
        if(encoder_name == "OrdinalEncoder"):
            encoder = ce.OrdinalEncoder(cols=columnsToEncode)
        if(encoder_name == "SumEncoder"):
            encoder = ce.SumEncoder(cols=columnsToEncode)
        if(encoder_name == "PolynomialEncoder"):
            encoder = ce.PolynomialEncoder(cols=columnsToEncode)
        if(encoder_name == "BaseNEncoder"):
            encoder = ce.BaseNEncoder(cols=columnsToEncode)
        if(encoder_name == "LeaveOneOutEncoder"):
            encoder = ce.LeaveOneOutEncoder(cols=columnsToEncode)
        #encoder = getattr(category_encoders, encoder_name)
        print(encoder)
        start_time = time.time()
        score, stds, raw_scores, dim = score_models(clf, X, y, encoder, encoder_name, dataSetName)
        scores.append([encoder_name, dataSetName, dim, score, stds, time.time() - start_time])
        raw_scores_ds[encoder_name] = raw_scores
        gc.collect()

    results = pd.DataFrame(scores, columns=['Encoding', 'Dataset', 'Dimensionality', 'Avg. Score', 'Score StDev', 'Elapsed Time'])

    #print(raw_scores_ds)
    #raw = pd.DataFrame.from_dict(raw_scores_ds)
    #print(raw)
    #ax = raw.plot(kind='box', return_type='axes')
    #plt.title('Scores for Encodings on %s Dataset' % (name, ))
    #plt.ylabel('Score (higher better)')
    #for tick in ax.get_xticklabels():
        #tick.set_rotation(90)
    #plt.grid()
    #plt.tight_layout()
    #plt.show()

    #return results, raw
    return results

if __name__ == '__main__':
   
    for dataSet in dataSetList:
        print(" ******************** " , dataSet , "******************")
        filePath = workDir+dataSet
        
        # Read the DataSet, Skipping the original header and passsing custom
        # Category-loaders library gives error if column name has (-) in the name, Explore more on this !
        df = pd.read_csv(filePath, names=columnHeader, skiprows=1)
        # Droping education columns (Duplicate)
        df.drop(duplicateColumn, axis=1, inplace = True)
        X = df.reindex(columns=[x for x in df.columns.values if x != targetLabel])
        #y = df.reindex(columns=['Class'])
        y = df[targetLabel]
        if "train" in dataSet:
            y.replace([' <=50K'], 0, inplace=True)
            y.replace([' >50K'], 1, inplace=True)
        elif "test" in dataSet:
            y.replace([' <=50K.'], 0, inplace=True)
            y.replace([' >50K.'], 1, inplace=True)
    
        #print(y.head(20))
        #main(dataSet, X, y)
    
        #out, raw = main(dataSet, X, y)
        out = main(dataSet, X, y)
        print(out.sort_values(by=['Dataset', 'Avg. Score']))
   