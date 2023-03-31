import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import pickle
import re
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline 
import sqlalchemy as sa
from sqlalchemy import inspect
import sys
from typing import Tuple

nltk.download(["punkt", "wordnet", "stopwords"])


def load_data(database_filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Load df from db
    """
    engine = sa.create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql_table("Message", engine)
    df = df.drop(["child_alone"],axis=1)
    df = df[df.related !=2]
    X = df.message
    Y = df[df.columns[5:]]
    cat_names = Y.columns

    return X, Y, cat_names


def tokenize(text: str) -> list:
    """
    Split text string into words and return list of words roots
    """
   
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    words = word_tokenize(text)
    stop_words = stopwords.words("english")
    words = [word for word in words if word not in stop_words]
    
    tokens = [WordNetLemmatizer().lemmatize(word) for word in words]
    return tokens


def build_model() -> GridSearchCV:
     """
     Build a model for classification of the disaster messages
     """
     pipeline = Pipeline([
         ('vect', CountVectorizer(tokenizer=tokenize)),
         ('tfidf', TfidfTransformer()),
         ('clf', MultiOutputClassifier(RandomForestClassifier()))
         ])
     
     parameters = {
         'clf__estimator__n_estimators': [10, 30],
         'clf__estimator__min_samples_split': [2, 3, 4]
         }   
     
     cv = GridSearchCV(pipeline, param_grid = parameters, scoring='f1_micro')

     return cv


def evaluate_model(model, X_test: pd.DataFrame, Y_test: pd.DataFrame, category_names: list) -> None:
    """
    Evaluate model and print the f1 score, precision and recall for each category in Y_test df
    """
    y_pred = model.predict(X_test)
    print(classification_report(Y_test.values, y_pred, target_names=category_names))
    return


def save_model(model: GridSearchCV, model_filepath: str) -> None:
    """
    Save model as pickle.dump
    """
    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)
    
    return


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()