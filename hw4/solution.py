import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

df = pd.read_csv('train.csv')

# print(df.isnull().sum())
'''ID       0
url      0
title    1
label    0'''

# print(df.describe())
'''                  ID          label
count  135309.000000  135309.000000
mean    67654.000000       0.123532
std     39060.488124       0.329048
min         0.000000       0.000000
25%     33827.000000       0.000000
50%     67654.000000       0.000000
75%    101481.000000       0.000000
max    135308.000000       1.000000'''
# Учитывая сколько у нас строк, 1 можно удалить за не надобностью
df = df.dropna(subset=['title'])

vec_url = TfidfVectorizer(max_features=10000)
vec_title = TfidfVectorizer(max_features=10000)
x_url = vec_url.fit_transform(df['url'])
x_title = vec_title.fit_transform(df['title'])
x = hstack([x_url, x_title])
y = df['label']


split = int(df['url'].count() * 0.9)
x_train, y_train = x[:split], y[:split]
x_test, y_test = x[split:], y[split:]


model = LogisticRegression()
model.fit(x_train, y_train)


# y_pred = model.predict(x_test)
# print(classification_report(y_test, y_pred))
'''              precision    recall  f1-score   support

           0       0.99      1.00      0.99     11804
           1       0.99      0.93      0.96      1727

    accuracy                           0.99     13531
   macro avg       0.99      0.96      0.98     13531
weighted avg       0.99      0.99      0.99     13531'''
# Теперь можно попробовать тестовые данные


test_df = pd.read_csv('test.csv')
# print(df.isnull().sum())
'''ID       0
url      0
title    0
label    0'''
# Можно не боятся пропущеных значений в test.csv


solution_df = pd.DataFrame()
solution_df['ID'] = test_df['ID']
x_url = vec_url.transform(test_df['url'])
x_title = vec_title.transform(test_df['title'])
x = hstack([x_url, x_title])
y_pred = model.predict(x)
solution_df['label'] = y_pred
solution_df.to_csv('submission.csv', index=False)
