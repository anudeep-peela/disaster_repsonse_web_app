import sys
from sqlalchemy import create_engine
import pandas as pd
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.pipeline import Pipeline , FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

def load_data(database_filepath):
    '''
    Loads data from database
    Input:
    database_filepath: (string) filepath to database
    Output:
    X : (pandas Series) messages series
    Y : (pandas DataFrame) target dataframe
    category_names : (list) labels of target varibales
    '''
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('df_clean', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = df.columns.to_list()[4:]
    return X, Y, category_names

def tokenize(text):
    '''
    Tokenizes the text data by 
    - lowering, removing punctuation
    - removing stopwords
    - lemmatizing
    Input: (strings) text data
    Output: (list) tokonized text

    '''
    # normalize case and remove punctuations
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize the text
    tokens = word_tokenize(text)
    # remove stop words and lemmatize the text
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return tokens


def build_model():
    '''
    Build a pipeline along with GridSearch
    Input: None
    Output: Trainable Model
    '''
    # Build pipeline 
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('multi_clf', MultiOutputClassifier(DecisionTreeClassifier()))])
    # set parameters for GridSearch
    parameters = {'multi_clf__estimator__max_depth': [5, 10]}
    # Set GridSearchCV model using pipline and parameters
    cv = GridSearchCV(pipeline, param_grid = parameters, n_jobs = -1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Displays model's performance on test set
    Input:
    model: Trained model
    X_test, Y_test : test data set
    category_names : labels of target variables
    '''
    # print classfication report for each column in Y
    y_pred = model.predict(X_test)
    for i in range(Y_test.shape[1]):
        print(classification_report(Y_test.values[:,i], y_predict[:, i]))


def save_model(model, model_filepath):
    '''
    Save the model as pickle 
    Input: 
    model: trained model
    model_filepath: path to save model
    '''
    # save the model
    pickle.dump(model, open(model_filepath, 'wb'))
    



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