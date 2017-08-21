import pandas as pd

# All that's left to clean up is Cabin
# First simple key is that NaN will be 0
def convert_cabin_to_int(string_cabin):
    if string_cabin == 0:
        return 0
    interest = string_cabin[0]
    retval = None
    if interest == 'A': retval = 1
    elif interest == 'B': retval = 2
    elif interest == 'C': retval = 3
    elif interest == 'D': retval = 4
    elif interest == 'E': retval = 5
    elif interest == 'F': retval = 6
    elif interest == 'G': retval = 7
    # only one T
    elif interest == 'T': retval = 8
    else: print(interest)
    return retval

def get_x_values(dataframe, mean_age_train):
    dataframe = dataframe.set_index('PassengerId')
    dataframe['Sex'] = dataframe['Sex'].replace('female', value = 0)
    dataframe['Sex'] = dataframe['Sex'].replace('male', value = 1)
    dataframe['Age'] = dataframe['Age'].fillna(mean_age_train)
    # Key will be ->
    # NaN = 0, S = 1, C = 2, Q = 3
    dataframe['Embarked'].replace('S', value = 1, inplace = True)
    dataframe['Embarked'].replace('C', value = 2, inplace = True)
    dataframe['Embarked'].replace('Q', value = 3, inplace = True)
    dataframe['Embarked'].fillna(value = 0, inplace = True)
    dataframe['Embarked'].describe()
    dataframe['Cabin'] = dataframe['Cabin'].fillna(value = 0)
    dataframe['Cabin'] = dataframe['Cabin'].apply(convert_cabin_to_int)
    dataframe = dataframe.drop(['Name', 'Ticket'], axis = 1)
    return dataframe