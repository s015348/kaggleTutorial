# From: https://www.kaggle.com/mrisdal/titanic/exploring-survival-on-the-titanic

from utility import fill_data, fillna


def clean_data(data):
    data = fill_data(data, 'PassengerId', [62, 830], 'Embarked', 'C')
    data = fillna(data, 'Fare')    
    return data


def get_deck(data):
    # Create a Deck variable. Get passenger deck A - F:
    data['Cabin'] = data['Cabin'].fillna('-')
    deck = data['Cabin'].str.get(0).to_frame()
    deck.columns = ['Deck']
    #print deck.head(10)
    data = data.join(deck['Deck'])
    #print data.head(10)
    return data


def get_family_size(data):
    # Create a family size variable including the passenger themselves
    data['Family size'] = data['Parch'] + data['SibSp'] + 1
    #print data.head(10)


def get_person_title(data):
    # Grab surname_tf from passenger names
    surname_tf = data['Name'].str.split(',').str.get(1).\
                            str.split('.').str.get(0).\
                            to_frame()
    surname_tf.columns = ['Title']

    surname_tf = surname_tf.join(data['Sex'])
    #print surname_tf.head(10)
    return surname_tf


def get_person_surname(data):
    # Grab grab surname from passenger name
    surname_tf = data['Name'].str.split(',').str.get(1).\
                            str.split('.').str.get(1).\
                            to_frame()
    surname_tf.columns = ['Surname']
    #print surname_tf.head(10)
    return surname_tf


def merge_rare_title(data):
    # Titles with very low cell counts to be combined to 'rare' level
    data['Title'] = data['Title'].fillna('rare')
    title = data['Title'].str.replace('Mlle', 'Ms').\
                        str.replace('Miss', 'Ms').\
                        str.replace('Lady', 'Ms').\
                        str.replace('Mme', 'Mrs').\
                        str.replace('Dr', 'rare').\
                        str.replace('Col', 'rare').\
                        str.replace('Major', 'rare').\
                        str.replace('Jonkheer', 'rare').\
                        str.replace('Sir', 'rare').\
                        str.replace('Don', 'rare').\
                        str.replace('Capt', 'rare').\
                        str.replace('Rev', 'rare').\
                        to_frame()
    data = data.drop('Title', axis=1, inplace=False).join(title)
    return data