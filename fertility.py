#!/usr/bin/env python
# coding: utf-8

# In[68]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# ### Data source: 
# https://archive.ics.uci.edu/ml/datasets/Fertility
# 
# ### Attribute Information:
# 
# - Season in which the analysis was performed. 1) winter, 2) spring, 3) Summer, 4) fall. (-1, -0.33, 0.33, 1)
# 
# - Age at the time of analysis. 18-36 (0, 1)
# 
# - Childish diseases (ie , chicken pox, measles, mumps, polio) 1) yes, 2) no. (0, 1)
# 
# - Accident or serious trauma 1) yes, 2) no. (0, 1)
# 
# - Surgical intervention 1) yes, 2) no. (0, 1)
# 
# - High fevers in the last year 1) less than three months ago, 2) more than three months ago, 3) no. (-1, 0, 1)
# 
# - Frequency of alcohol consumption 1) several times a day, 2) every day, 3) several times a week, 4) once a week, 5) hardly ever or never (0, 1)
# 
# - Smoking habit 1) never, 2) occasional 3) daily. (-1, 0, 1)
# 
# - Number of hours spent sitting per day ene-16 (0, 1)
# 
# - Output: Diagnosis normal (N), altered (O)

# In[69]:


headers = ['performed_seasons', 'age', 'childish_disease', 'trauma', 'surgical_intervention', 
           'fever_last_year', 'alcohol', 'smoking', 'hours_sitting', 'output']


# In[70]:


raw_data = pd.read_csv('raw_data/fertility_Diagnosis.txt', header = None, names = headers)
raw_data.info()


# In[71]:


raw_data['output'].value_counts()


# In[72]:


raw_data.head()


# In[73]:


# transform the data for interpretation
df = raw_data.copy()
seasons = {-1: 'winter', -0.33: 'spring', 0.33: 'summer', 1: 'fall'}
df['performed_seasons'] = df['performed_seasons'].apply(lambda x: seasons[x])

fevers = {-1:'less than 3M ago', 0: 'more than 3M ago', 1:'No'}
df['fever_last_year'] = df['fever_last_year'].apply(lambda x: fevers[x])

alcohols = {0.2:'several times a day', 0.4:'every day', 0.6:'several time a week', 0.8:'once a week', 1:'hardly ever or never'}
df['alcohol'] = df['alcohol'].apply(lambda x: alcohols[x])

smokings = {-1:'never', 0:'occasional', 1:'every day'}
df['smoking'] = df['smoking'].apply(lambda x: smokings[x])

df['age'] = ((df['age'] * 18) + 18).astype(int)
df['hours_sitting'] = df['hours_sitting'] * 16
df['output'] = df['output'].apply(lambda x: 'Normal' if x == 'N' else 'Altered')
df.head()


# A pretty small data set (100 data points). The label is 12% vs. 88% so I think it's fine to play around it and understand more.

# ### Exploratory Analysis

# **Performed Seasons**

# In[74]:


sns.countplot(data=df[['performed_seasons','output']],x='performed_seasons', hue='output')


# Less analyses were performed in the summer. 

# **Fevers in the last year**

# In[75]:


sns.countplot(data=df[['fever_last_year','output']],x='fever_last_year', hue='output')


# In[76]:


altered_by_fever = df[['fever_last_year', 'output']].groupby('fever_last_year').apply(lambda x: x[x['output'] == 'Altered']['output'].count()) 
ana_by_fever = df.groupby('fever_last_year')['output'].count()
(altered_by_fever / ana_by_fever).round(2)


# There is no significant difference between the timing of high fevers in the last year.

# **Alcohol Consumption**

# In[77]:


sns.countplot(data=df[['alcohol','output']],x='alcohol', hue='output')
plt.xticks(rotation=30);


# In[78]:


altered_by_alcohol = df[['alcohol', 'output']].groupby('alcohol').apply(lambda x: x[x['output'] == 'Altered']['output'].count()) 
ana_by_alcohol = df.groupby('alcohol')['output'].count()
(altered_by_alcohol / ana_by_alcohol).round(2)


# In terms of alcohol consumption, the groups of 'every day' and 'several times a day' have no altered data points due to the group sizes. The group of 'several time a week' has relatively altered rate than the rest.

# **Smoking Habits**

# In[79]:


sns.countplot(data=df[['smoking','output']],x='smoking', hue='output')


# The major group has no smoking behavior. As for the rest groups, there is no significant differences between them.

# **Age**

# In[80]:


age_max = df['age'].max()
age_min = df['age'].min() 

sns.histplot(data=df[['age','output']],bins=age_max-age_min,
             x='age', hue='output',multiple='stack')
ticks = [age_min + i for i in range(age_max-age_min)]
plt.xticks(ticks);


# In[81]:


altered_by_age = df[['age', 'output']].groupby('age').apply(lambda x: x[x['output'] == 'Altered']['output'].count()) 
ana_by_age = df.groupby('age')['output'].count()
(altered_by_age / ana_by_age).round(2)


# The smallest age 27 had the most analyses. As age increases, number of analyses reduces. Altered rate does not follow this trend.

# **Hours spent sitting per day**

# In[82]:


hours_max = df['hours_sitting'].max()
hours_min = df['hours_sitting'].min() 

sns.histplot(data=df[['hours_sitting','output']],x='hours_sitting', hue='output',multiple='stack')


# The majority spends less than 11 hours on sitting per day, within which altered appears in every sub-group.

# **Surgical intervention**

# In[83]:


sns.countplot(data=df[['surgical_intervention','output']], x='surgical_intervention', hue='output')


# In[84]:


df['surgical_intervention'].value_counts()


# There seems no significant differences whether there is surgical intervention or not.

# **Childish diseases**

# In[85]:


sns.countplot(data=df[['childish_disease','output']], x='childish_disease', hue='output')


# In[86]:


altered_by_disease = df[['childish_disease', 'output']].groupby('childish_disease').apply(lambda x: x[x['output'] == 'Altered']['output'].count()) 
ana_by_disease = df.groupby('childish_disease')['output'].count()
(altered_by_disease / ana_by_disease).round(2)


# There seems no significant differences whether childish diseases developed or not.

# **Accident or serious trauma**

# In[87]:


sns.countplot(data=df[['trauma','output']], x='trauma', hue='output')


# There seems no significant differences whether accident or serious trauma occurred or not.

# In[88]:


altered_group = df[df['output']=='Altered']
normal_group = df[df['output']!='Altered']


# In[89]:


print('Altered Group Stats\n',altered_group.describe()[1:])
print('\n')
print('Normal Group Stats\n',normal_group.describe()[1:])


# Observations:
# - Age: Altered is averagely older
# - Accident or trauma: more likely to happen in Altered
# - Surgical Intervention: more likely to happen in Normal
# - Hours spent sitting: Altered is averagely longer

# ### Maching Learning

# **KNN**

# In[90]:


#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.model_selection import train_test_split


# In[105]:


# Split the data

X = raw_data.iloc[:, :len(raw_data.columns)-1]
y = raw_data.iloc[:, len(raw_data.columns)-1].apply(lambda x: 1 if x == 'O' else 0)
print('X\'s shape:', X.shape)
print('Y\'s shape:', len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
print('\nX_train\'s shape:', X_train.shape)
print('\ny_train\'s shape:', len(y_train), '; it has {} altered'.format(sum(y_train)))
print('y_test\'s shape:', len(y_test), '; it has {} altered'.format(sum(y_test)))


# In[96]:


# Instantiate classifier
knn = KNeighborsClassifier(n_neighbors=2)


# In[97]:


# Fit the classifier to the training data
knn.fit(X_train, y_train)


# In[98]:


# Predict on the test data
y_pred = knn.predict(X_test)
y_pred


# In[99]:


# Find the mean accuracy on the given test data and labels.
knn.score(X_test, y_test)


# 3/20 = 0.15 => Failure. It couldn't identify any item

# Try with different combinations of predicted variables 
# 
# Because `performed_seasons` is not related to the patients themselves, we drop it from the X data set.

# In[151]:


def selectPredictors(n, all_cols, scores):
    import itertools
    for cols in itertools.combinations(all_cols, n):
        X = raw_data.loc[:, cols]
        scores.append((cols, runAlgorithm(X)))

def runAlgorithm(X):
    y = raw_data.iloc[:, len(raw_data.columns)-1].apply(lambda x: 1 if x == 'O' else 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)    
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train, y_train)
    return knn.score(X_test, y_test)


# In[162]:


scores = []
all_cols = raw_data.columns[1:-1]
for i in range(1, len(raw_data.columns)):
    selectPredictors(i, all_cols, scores)


# In[164]:


sort_scores = sorted(scores, key = lambda x: -x[1])
sort_scores


# Count the frequency of predictors that contribute in the combination with high score

# In[168]:


#from collections import defaultdict

predictors_ct = defaultdict(int)
for i in sort_scores:
    if i[1] >= 0.9:
        for j in i[0]:
            predictors_ct[j] += 1
predictors_ct


# Selecting the top 3 predictors of `age`, `trauma`, and `surgical_intervention` can slightly improve the accuracy.

# In[173]:


# confirm the prediction

X = raw_data.loc[:, ['age', 'trauma', 'surgical_intervention']]
y = raw_data.iloc[:, len(raw_data.columns)-1].apply(lambda x: 1 if x == 'O' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Instantiate classifier
knn = KNeighborsClassifier(n_neighbors=2)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict on the test data
y_pred = knn.predict(X_test)

print('Actual:', y_test)
print('\nPredicted:', y_pred)


# **Logistic Regression**

# In[100]:


# from sklearn.linear_model import LogisticRegression


# In[174]:


print('X_train\'s shape:', X_train.shape)
print('\ny_train\'s shape:', len(y_train), '; it has {} altered'.format(sum(y_train)))
print('y_test\'s shape:', len(y_test), '; it has {} altered'.format(sum(y_test)))


# In[175]:


logreg = LogisticRegression()


# In[176]:


logreg.fit(X_train, y_train)


# In[188]:


y_pred = logreg.predict(X_test)
y_pred_prob = logreg.predict_proba(X_test)[:,1].round(3)

print('Actual:', y_test)
print('\nPredicted 0/1:', y_pred)
print('\nPredicted Probability:', y_pred_prob)


# worse

# In[189]:


from sklearn.metrics import roc_curve


# In[190]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label = 'Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()


# **SVM**

# In[191]:


#from sklearn.svm import SVC


# In[202]:


clf = SVC(probability=True)


# In[203]:


clf.fit(X_train, y_train)


# In[204]:


y_pred = clf.predict(X_test)
y_pred


# In[205]:


clf.score(X_test, y_test)


# In[206]:


y_pred_prob = clf.predict_proba(X_test)[:,1].round(3)

print('Actual:', y_test)
print('\nPredicted 0/1:', y_pred)
print('\nPredicted Probability:', y_pred_prob)


# this is awful

# In[207]:


# Split the data

X = raw_data.iloc[:, :len(raw_data.columns)-1]
y = raw_data.iloc[:, len(raw_data.columns)-1].apply(lambda x: 1 if x == 'O' else 0)
print('X\'s shape:', X.shape)
print('Y\'s shape:', len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
print('\nX_train\'s shape:', X_train.shape)
print('\ny_train\'s shape:', len(y_train), '; it has {} altered'.format(sum(y_train)))
print('y_test\'s shape:', len(y_test), '; it has {} altered'.format(sum(y_test)))


# In[208]:


clf = SVC(probability=True)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)[:,1].round(3)

print('Actual:', y_test)
print('\nPredicted 0/1:', y_pred)
print('\nPredicted Probability:', y_pred_prob)


# After trying, I think the analysis couldn't give out a good information due to the extremely small size. (100)

# In[ ]:




