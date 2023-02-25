
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd


data = pd.read_csv("clean_data.csv")
Xfeatures = data["clean-text"]
Ylabels = data["sentiment"]
cv = CountVectorizer()
X =cv.fit_transform(Xfeatures.values.astype('U'))
X.toarray()
X_train,X_test,Y_train,Y_test = train_test_split(X,Ylabels,test_size=0.2,random_state=42)
nv_model = MultinomialNB()
nv_model.fit(X_train,Y_train)
# nv_model.score(X_test,Y_test)

def Model_1(text):
    vect= cv.transform([text]).toarray()
    value =nv_model.predict(vect)
    return value





  