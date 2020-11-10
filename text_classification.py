from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_extraction.text import TfidfVectorizer

# Corpus (X) and target (y) from a pandas df
corpus = df[['text']].values[:,0].astype('U').tolist()
ytrain = pd.Series(df['yvar'])

# TFIDF
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), lowercase = True, max_features = 20000)
tf_transformer = tf.fit(corpus)
xtrain = tf.fit_transform(corpus)
# Save transformer
pickle.dump(tf_transformer, open(mypath + "tfidf.pkl", "wb"))

clf = LogisticRegressionCV(n_jobs=2, penalty='l2', solver='liblinear', cv=10, scoring = 'accuracy', random_state=0)
clf.fit(xtrain, ytrain)

# Save classifier
with open(mypath + 'clf.pkl', 'wb') as f:
    pickle.dump(clf, f)

### New text
# Load classifier
with open(modelpath + "clf.pkl", 'rb') as f:
    clf = pickle.load(f)

# Load transformer and plug in new text
tf1 = pickle.load(open(modelpath + "tfidf.pkl", 'rb'))
tf1_new = TfidfVectorizer(analyzer='word', ngram_range=(1,2), lowercase = True, max_features = 20000, vocabulary = tf1.vocabulary_)

# Predict classifier on tf1_new...
xtest = tf1_new.fit_transform([mystring])
res = np.round(clf.predict_proba(xtest)[:,1],2)
