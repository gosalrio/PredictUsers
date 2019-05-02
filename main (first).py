#Import Modules
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor as RFR, RandomForestClassifier as RFC
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler, Binarizer, scale

#Import CSV
TSV_FILE = "pictures-train.tsv"

#MANIPULATE DATAFRAME TO BEST FIT
#Handle nans
trainDF = pd.read_csv(TSV_FILE, delimiter='\t')
trainDF.iloc[:,np.arange(-3,0)].astype('float64', inplace=True)
last3 = trainDF.iloc[:,np.arange(-3,0)]
last3[last3 < 0] = np.nan
trainDF = trainDF.drop(last3.columns, axis=1).join(last3).dropna()

##trainDF = (trainDF.join(etitleD)).join(regionD)
trainDF['votedon'] = pd.to_datetime(trainDF.votedon)
trainDF['takenon'] = pd.to_datetime(trainDF.takenon, errors="coerce")
trainDF = trainDF[-trainDF.takenon.isnull()]

#Converting to dummies
etitleD = pd.get_dummies(trainDF['etitle']).astype(int)
regionD = pd.get_dummies(trainDF['region']).astype(int)

#Select Necessary Features
## regionArr = (regionD.sum()/regionD.sum().max()).values
## (regionArr*100).astype('int64')[(regionArr*100).astype('int64') >= 10].size
##KBestEtitle = SelectKBest(chi2, k=9)
##KBestRegion = SelectKBest(chi2, k=58)
#### TrainDF Feature Select
##KBestTrain = SelectKBest(chi2, k=1) #5 Features max, 9 - 4 (etitle, region, votes, author_id), keeping 4 only

##new_trainDF = pd.DataFrame(KBestTrain.fit_transform(trainDF[['viewed', 'n_comments']], trainDF.votes),
##                           columns=trainDF.columns[KBestTrain.get_support(indices=True)])
##new_etitleD = pd.DataFrame(KBestEtitle.fit_transform(etitleD, trainDF.votes),
##                           columns=etitleD.columns[KBestEtitle.get_support(indices=True)])
##new_regionD = pd.DataFrame(KBestRegion.fit_transform(regionD, trainDF.votes),
##                           columns=regionD.columns[KBestRegion.get_support(indices=True)])
##mainDF = new_trainDF.join(new_etitleD).join(new_regionD)

#Split into Training and Test
X_train, X_test, y_train, y_test = tts(mainDF, trainDF.votes, test_size=0.30, random_state=42)
rfr = RFR()
rfr.fit(X_train, y_train)
print('Training Score:',rfr.score(X_train, y_train))
print('Testing Score:',rfr.score(X_test, y_test))
