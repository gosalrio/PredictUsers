#Import Modules
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestRegressor as RFR

#IMPORT CSV
TSV_FILE = "pictures-train.tsv"
trainDF = pd.read_csv(TSV_FILE, delimiter='\t')

#Handling NaNs
trainDF.iloc[:,np.arange(-3,0)].astype('float64', inplace=True)
last3 = trainDF.iloc[:,np.arange(-3,0)]
last3[last3 < 0] = np.nan
trainDF = trainDF.drop(last3.columns, axis=1).join(last3).dropna()

#Log Votes
trainDF['logVotes'] = np.log(trainDF.votes[trainDF.votes > 0])
trainDF.logVotes.fillna(0, inplace=True)

#Picture Age [Need Conversion to Int]
trainDF.takenon = pd.to_datetime(trainDF.takenon, errors='coerce')
trainDF = trainDF[-trainDF.takenon.isnull()]
trainDF['authorAge'] = (trainDF.takenon - trainDF.takenon.min()).astype('timedelta64[D]')

#Author Credibility
authorAge = trainDF.loc[:, ['author_id', 'takenon']].groupby('author_id').max() - trainDF.\
               loc[:, ['author_id', 'takenon']].groupby('author_id').min()
trainDF = trainDF.merge(authorAge, how='inner', left_on='author_id', right_index=True, suffixes=('_deltaTime','_inDays'))
trainDF.takenon_inDays = trainDF.takenon_inDays.astype('timedelta64[D]')

#Convert to DateTime64
trainDF.votedon = pd.to_datetime(trainDF.votedon)
trainDF['votedon'] = pd.to_datetime(trainDF.votedon)
trainDF['picAge'] = (trainDF.votedon - trainDF.votedon.min()).astype('timedelta64[D]')

#Create Dummy Columns
etitleD = pd.get_dummies(trainDF['etitle']).astype(bool)
regionD = pd.get_dummies(trainDF['region']).astype(bool)

#Split Dataset into Training and Testing
selFeatures = ['n_comments', 'viewed', 'picAge', 'takenon_inDays', 'author_id']
X_train, X_test, y_train, y_test = tts((trainDF.loc[:, selFeatures].join(etitleD)).join(regionD),
                                       trainDF.logVotes, test_size=0.30, random_state=42)
rfr = RFR()
rfr.fit(X_train, y_train)
print('Training Score:',rfr.score(X_train, y_train))
print('Testing Score:',rfr.score(X_test, y_test))
