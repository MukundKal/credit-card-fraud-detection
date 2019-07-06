
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import keras

np.random.seed(2)


# In[2]:


data = pd.read_csv('creditcard.csv')


# ## Data exploration

# In[3]:


data.head()


# ## Pre-processing

# In[4]:


from sklearn.preprocessing import StandardScaler
data['normalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data = data.drop(['Amount'],axis=1)


# In[5]:


data.head()


# In[6]:


data = data.drop(['Time'],axis=1)
data.head()


# In[7]:


X = data.iloc[:, data.columns != 'Class']
y = data.iloc[:, data.columns == 'Class']


# In[8]:


y.head()


# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=0)


# In[10]:


X_train.shape


# In[11]:


X_test.shape


# ## Random Forest

# In[12]:


from sklearn.ensemble import RandomForestClassifier


# In[13]:


random_forest = RandomForestClassifier(n_estimators=100)


# In[14]:


random_forest.fit(X_train,y_train.values.ravel())


# In[15]:


y_pred = random_forest.predict(X_test)


# In[16]:


random_forest.score(X_test,y_test)


# In[17]:


import matplotlib.pyplot as plt
import itertools

from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[18]:


cnf_matrix = confusion_matrix(y_test,y_pred)


# In[19]:


plot_confusion_matrix(cnf_matrix,classes=[0,1])


# In[20]:


plt.show()


# In[22]:


y_pred = random_forest.predict(X)


# In[23]:


cnf_matrix = confusion_matrix(y,y_pred.round())


# In[24]:


plot_confusion_matrix(cnf_matrix,classes=[0,1])


# In[25]:


plt.show()

