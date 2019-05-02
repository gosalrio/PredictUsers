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

with plt.xkcd():
    pieRegion = trainDF.groupby('region').size()
    pieRegionLabels = trainDF.groupby('region')
    plt.pie(pieRegion)
    plt.legend(pieRegionLabels.groups.keys(),bbox_to_anchor=(0,1),title='Region')
    plt.title('Distribution of Regions')
    plt.annotate('Is this informative?',xy=(0,0),xytext=(0,-1.2))
    plt.savefig('Pie Regions 1.png',dpi=200)
    plt.close()

with plt.xkcd():
    pieETitle = trainDF.groupby('etitle').size()
    pieETitleLabels = trainDF.groupby('etitle').groups.keys()
    plt.pie(pieETitle)
    plt.legend(pieETitleLabels,bbox_to_anchor=(0,1),title='Category')
    plt.title('Post Category')
    plt.annotate('This looks good',xy=(0,0),xytext=(0,-1.2))
    plt.savefig('Pie Categories.png',dpi=200,bbox_inches='tight')
    plt.close()

with plt.xkcd():
    plt.hist(trainDF['votes'],bins=20)
    plt.xlabel('Upvotes')
    plt.ylabel('Number of Posts')
    plt.title('Distribution of Upvotes')
    plt.annotate('COOLEST TRAIN',xy=(trainDF['votes'].max(),360),xytext=(250,2500),
                 arrowprops=dict(arrowstyle='->'))
    plt.savefig('Histogram of Upvotes.png',dpi=200,bbox_inches='tight')
    plt.close()    

with plt.xkcd():
    plt.hist(trainDF['viewed'],bins=200)
    plt.xscale('log')
    plt.xlabel('Log Number of Views')
    plt.ylabel('Number of Posts')
    plt.title('Distribution of Views')
    plt.annotate('PRITTIEST TRAIN',xy=(trainDF['viewed'].max(),360),xytext=(2200,2500),
                 arrowprops=dict(arrowstyle='->'))
    plt.savefig('Histogram of Views.png',dpi=200,bbox_inches='tight')
    plt.close()

with plt.xkcd(scale=1):
    plt.hist(trainDF['n_comments'],bins=100)
    plt.xscale('log')
    plt.xlabel('Number of Comments')
    plt.ylabel('Log Number of Posts')
    plt.title('Distribution of Comments')
    plt.annotate('MOST INTRIGUING\n      TRAIN',xy=(trainDF['n_comments'].max(),360),
                 xytext=(15,10000),arrowprops=dict(arrowstyle='->'))
    plt.savefig('Histogram of Comments.png',dpi=200,bbox_inches='tight')
    plt.close()

trainDF['yearVoted'] = pd.DatetimeIndex(trainDF['votedon']).year
meanForYearsDF = trainDF.groupby('yearVoted').mean()
numPostPerYear = pd.DataFrame(trainDF.groupby(['yearVoted']).size())
numPostPerYear.columns = ['num']

with plt.xkcd():
    plt.yscale('log')
    plt.plot(meanForYearsDF.index,meanForYearsDF['votes'])
    plt.plot(meanForYearsDF.index,meanForYearsDF['viewed'])
    plt.plot(meanForYearsDF.index,meanForYearsDF['n_comments'])
    plt.plot(numPostPerYear.index,numPostPerYear['num'])
    plt.legend(['Votes','Views','Comments','Posts'],bbox_to_anchor=(1,1),title='Category')
    plt.title('Activity Over Time')
    plt.xlabel('Year')
    plt.ylabel('Number of Each Category (log)')
    plt.savefig('Combined Plot.png',dpi=200,bbox_inches='tight')
    plt.close()

newDF = pd.DataFrame({"Predicted Value":rfr.predict(X_test), "True Value":y_test})
with plt.xkcd():
    plt.hist(newDF['True Value']-newDF['Predicted Value'],bins=50)
    plt.title('Difference Between True and Predicted Votes')
    plt.xlabel('Difference (log votes)')
    plt.ylabel('Occurance')
    plt.annotate('OBSCURE OUTLIER\n    (please ignore)',xy=(min(newDF['True Value']-newDF['Predicted Value']),25),
                 xytext=(-4.2,1000),arrowprops=dict(arrowstyle='->'))
    plt.savefig('Diff Hist.png',dpi=200,bbox_inches='tight')
    plt.close()

