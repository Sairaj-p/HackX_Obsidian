from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pandas as pd

def train_model(model, data, targets):

    # Create a Pipeline object with a TfidfVectorizer and the given model
    text_clf = Pipeline([('vect',TfidfVectorizer()),
                         ('clf', model)])
    # Fit the model on the data and targets
    text_clf.fit(data, targets)
    return text_clf



# #Logistic Regression model


# y_pred=log_reg.predict(X_test)
# log_reg_accuracy = accuracy_score(Y_test, y_pred)
# print('Accuracy logistic regression: ', log_reg_accuracy,'\n')

def analys(text):
    data = pd.read_csv("clean_data.csv")
    Xfeatures = data["clean-text"]
    Ylabels = data["sentiment"]
    cv = CountVectorizer()
    X =cv.fit_transform(Xfeatures.values.astype('U'))
    X.toarray()
    X_train,X_test,Y_train,Y_test = train_test_split(Xfeatures.values.astype('U'),Ylabels,test_size=0.2,random_state=42)
    log_reg = train_model(LogisticRegression(solver='liblinear',random_state = 42), X_train, Y_train)
    def final_predict(text):
        lr = list(log_reg.predict(text))
        return lr
    anlys =pd.DataFrame(final_predict(text))[0].value_counts()
    sum = anlys.sum()
    for i in range(len(anlys)):
        anlys[i] = anlys[i]/sum
    return anlys

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   analys()
   train_model()

