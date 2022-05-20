import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn import linear_model

df = pd.read_csv('tmdb_movies_data.csv')
df['net_profit'] = df['revenue_adj'] - df['budget_adj']
df.drop(['id', 'imdb_id', 'original_title', 'homepage', 'tagline', 'director', 'overview',
        'cast', 'production_companies', 'release_year', 'release_date', 'keywords', 'budget', 'revenue'], axis=1, inplace=True)


def cleanData():
    global df
    # remove null values
    df = df.dropna()
    # remove duplicates
    df = df.drop_duplicates()

    # Filter and clean the columns and rows
    df = df[df['vote_average'] > 0]
    df = df[df['vote_count'] > 0]
    df = df[df['popularity'] > 0]


def normalize():
    global df
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (
            df[feature_name] - min_value) / (max_value - min_value)
    df= result


def Feature_Encoder(X, cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X


df = Feature_Encoder(df, ['genres'])
cleanData()
normalize()

x = df.drop(['net_profit'],axis=1)
y = df['net_profit']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

scaler = StandardScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


y_train = np.array(y_train)
y_train = y_train.reshape(y_train.shape[0], 1)

y_test = np.array(y_test)
y_test = y_test.reshape(y_test.shape[0], 1)

x_train = np.vstack((np.ones((x_train.shape[0], )), x_train.T)).T
x_test = np.vstack((np.ones((x_test.shape[0], )), x_test.T)).T



def model(X, Y, learning_rate, iteration):

    m = Y.size
    theta = np.zeros((X.shape[1], 1))
    cost = 0
    for i in range(iteration):

        y_pred = np.dot(X, theta)
        cost = (1/(2*m))*np.sum(np.square(y_pred - Y))
        d_theta = (1/(m))*np.dot(X.T, y_pred - Y)
        theta = theta - learning_rate*d_theta
        
       
    return theta, cost

theta, cost = model(x_train, y_train, learning_rate=0.0008, iteration=1000)
y_pred = np.dot(x_test, theta)
error = (1/x_test.shape[0])*np.sum(np.abs(y_pred - y_test))
print("test error is :", error*100, "%")
print("test accuracy is :", (1 - error)*100, "%")

print(cost)
print(theta)




