import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import itertools
import xgboost as xgb
import scipy
from sklearn.utils import resample
import statistics

stemmer = SnowballStemmer("english", ignore_stopwords=True)

def preprocess(tokens):
    tokens_nop = [stemmer.stem(t) for t in tokens if t not in string.punctuation]
    tokens_nop = [t.lower() for t in tokens_nop]
    return tokens_nop


#Open and preprocess training data
dftraindata = pd.read_csv('train_v2.csv')

y_train = dftraindata[['category']].copy()

dftraindata = dftraindata.values

dftrain = []

for i in range(len(dftraindata)):
    try:
        dftrain.append(dftraindata[i][1] + " " + dftraindata[i][3].replace(" ", ""))
    except:
        dftrain.append(dftraindata[i][1] + " ")

dftrain = pd.DataFrame(dftrain)

dftrain['title_tokens'] = dftrain[0].map(word_tokenize)
dftrain['title_processed'] = dftrain.title_tokens.apply(preprocess)
dftrain['title_processed_text'] = dftrain.title_processed.apply(lambda x: ' '.join(x))


#Open and preprocess test data
dftestfile = pd.read_csv('test_v2.csv')
dftestdata = dftestfile
dftestdata = dftestdata.values

dftest = []

for i in range(len(dftestdata)):
    try:
        dftest.append(dftestdata[i][1] + " " + dftestdata[i][3].replace(" ", ""))
    except:
        dftest.append(dftestdata[i][1] + " ")

dftest = pd.DataFrame(dftest)

dftest['title_tokens'] = dftest[0].map(word_tokenize)
dftest['title_processed'] = dftest.title_tokens.apply(preprocess)
dftest['title_processed_text'] = dftest.title_processed.apply(lambda x: ' '.join(x))


dftraintest = pd.concat([dftrain, dftest], ignore_index=True)

vec_tfidf = CountVectorizer(analyzer="word")
tfidf = vec_tfidf.fit_transform(dftraintest['title_processed_text'])

temp = tfidf[0:6027, :]
Xtest = tfidf[6027:, :]

#Train and publish

temp = temp.todense()
temp1 = np.concatenate((temp, y_train.values), axis=1)
Xtraintempdf = pd.DataFrame(temp1)
catcol = Xtest.shape[1]

df_cat0 = Xtraintempdf.loc[Xtraintempdf[catcol] == 0]
df_cat1 = Xtraintempdf.loc[Xtraintempdf[catcol] == 1]
df_cat2 = Xtraintempdf.loc[Xtraintempdf[catcol] == 2]
df_cat3 = Xtraintempdf.loc[Xtraintempdf[catcol] == 3]
df_cat4 = Xtraintempdf.loc[Xtraintempdf[catcol] == 4]

df_cat1 = resample(df_cat1, replace=True, n_samples=279 + 400, random_state=145)
df_cat3 = resample(df_cat3, replace=True, n_samples=362 + 300, random_state=145)

dfnormalized = pd.concat([df_cat0, df_cat1, df_cat2, df_cat3, df_cat4])
dfnormalized = dfnormalized.sample(frac=1).reset_index(drop=True)

y_train1 = dfnormalized[[catcol]].copy()
y_train1 = np.ravel(y_train1.values, order='C')

X_train1 = dfnormalized.drop([catcol], axis=1)
X_train1 = scipy.sparse.csr_matrix(X_train1.values)

clf = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, gamma=0.01, max_depth=3, base_score=2.5, n_jobs=3, random_state=815, reg_alpha=0.5).fit(X_train1, y_train1)

predicted = clf.predict(Xtest)
article_int = pd.to_numeric(dftestfile['article_id'])
output = np.stack((article_int, predicted), axis=-1)

with open('prediction.csv', 'w') as predictionfile:
    predictionfile.write("article_id,category\n")
    np.savetxt(predictionfile, output, delimiter=",", fmt='%s')
