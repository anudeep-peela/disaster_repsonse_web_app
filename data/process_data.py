import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Extract the data from csv files 
    INPUT:
    messages_filepath - (string) filepath of csv file containing messages 
    categories_failpath - (string) filepath of csv file containing response category
    OUTPUT:
    df - (pandas df) merged dataframe from messages and categories
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on = 'id')

    return df


def clean_data(df):
    '''
    Creates categories columns and drops duplicates
    INPUT:
    df - (pandas df) dataframe containg messages and categories
    OUTPUT:
    pandas df
    '''
    # convert categories column into a dataframe
    categories = df['categories'].str.split(';', expand = True)
    # set first row 
    row = categories.iloc[0,:]
    # use only the word part of the string
    categoty_colnames = row.apply(lambda x: x[:-2])
    categories.columns = categoty_colnames
    # use only the numeric part of the string
    for column in categories:
        categories[column] = categories[column].apply(lambda x: int(x[-1]))
    # drop the unclean category column
    df.drop('categories', axis = 1, inplace = True)
    # concat with new categories df
    df = pd.concat([df, categories], axis = 1)
    # drop duplicates
    df.drop_duplicates(inplace = True)

    return df

def save_data(df, database_filename):
    """
    Save the dataframe to desired database_filename
    Input:
    df : (pandas df) clean dataframe with messages and categories
    database_filename: (string) database filename
    Output:
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('df_clean', engine, index = False, if_exits = 'replace')  



def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()