import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
import clean_data as clean

base_path = 'C:/Users/Ayman/Desktop/kaggle/titanic/'

# (samples, features)
train_data = pd.DataFrame(data = pd.read_csv(base_path + 'train.csv'))
test_data = pd.DataFrame(data = pd.read_csv(base_path + 'test.csv'))
combined_data = [train_data, test_data]


train_data.columns.values

# What we've got then in the trained data is:
# PassengerID, Survived, PClass, Name, Sex, Age, SibSp, Parch,
# Ticket, Fare, Cabin, Embarked 

# Let's really quickly see what happens if we look at survival
survived = train_data[train_data['Survived'] == 1]
dead = train_data[train_data['Survived'] == 0]
survived.describe()
dead.describe()

# Seems that Fare, Parch, SibSp, Class are all important

# Clean up sex, make sure only male and female and no null
train_data['Sex'].unique()
train_clean = train_data.copy()
train_clean['Sex'] = train_clean['Sex'].replace('female', value = 0)
train_clean['Sex'] = train_clean['Sex'].replace('male', value = 1)

# Initial clean up for age is just going to be fill unknown ages with the
# mean value
# Not the greatest start but it shouldn't be unreasonable since the difference
# in age didn't seem to be a great predictor

train_clean['Age'] = train_data['Age'].fillna(train_data['Age'].mean())

# Clean up embarked by making it categorical
embarked_values = list(train_data['Embarked'].unique())
print(embarked_values)

# Key will be ->
# NaN = 0, S = 1, C = 2, Q = 3
train_clean['Embarked'].replace('S', value = 1, inplace = True)
train_clean['Embarked'].replace('C', value = 2, inplace = True)
train_clean['Embarked'].replace('Q', value = 3, inplace = True)
train_clean['Embarked'].fillna(value = 0, inplace = True)
train_clean['Embarked'].describe()

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

train_clean['Cabin'] = train_data['Cabin'].fillna(value = 0)
train_clean['Cabin'] = train_clean['Cabin'].apply(convert_cabin_to_int)

# Drop the name and ticket because they have relatively no information
# attached to them
train_clean = train_clean.drop(['Name', 'Ticket'], axis = 1)
train_clean.columns.values
train_clean.describe()

# See if there's an effect by cabin
survived = train_clean[train_clean['Survived'] == 1]
dead = train_clean[train_clean['Survived'] == 0]

survived
dead

# Seems promising
survived['Cabin'].describe()
dead['Cabin'].describe()

# Let's do the logistic regression

logis_x = train_clean.drop('Survived', axis = 1)
logis_x = logis_x.set_index('PassengerId')
logis_y = train_clean['Survived']
logis_y = np.ravel(logis_y)

model = LogisticRegression()
model.fit(logis_x, logis_y)
model.score(logis_x, logis_y)

logis_x.columns
check_coef = zip(logis_x.columns, np.transpose(model.coef_))
check_coef = pd.DataFrame(list(check_coef))
check_coef

cleaned_test = clean.get_x_values(test_data, train_data['Age'].mean())
cleaned_test['Fare'].fillna(value = train_data['Fare'].mean(), inplace = True)


predicted = model.predict(cleaned_test)
submission = pd.DataFrame(test_data['PassengerId'])
submission.insert(1, 'Survived', predicted)
submission.set_index('PassengerId')
submission.to_csv(base_path + 'submission.csv', index = False)