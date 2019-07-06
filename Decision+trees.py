
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


# ## Decision trees

# In[12]:


from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()


# In[13]:


decision_tree.fit(X_train,y_train.values.ravel())


# In[14]:


y_pred = decision_tree.predict(X_test)


# In[15]:


decision_tree.score(X_test,y_test)


# In[16]:


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


# In[17]:


y_pred = decision_tree.predict(X)


# In[18]:


y_expected = pd.DataFrame(y)


# In[19]:


cnf_matrix = confusion_matrix(y_expected,y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])
plt.show()

