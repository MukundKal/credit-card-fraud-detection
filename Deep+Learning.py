
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

# In[5]:


from sklearn.preprocessing import StandardScaler
data['normalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data = data.drop(['Amount'],axis=1)


# In[6]:


data.head()


# In[7]:


data = data.drop(['Time'],axis=1)
data.head()


# In[8]:


X = data.iloc[:, data.columns != 'Class']
y = data.iloc[:, data.columns == 'Class']


# In[11]:


y.head()


# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=0)


# In[15]:


X_train.shape


# In[16]:


X_test.shape


# In[17]:


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# ## Deep neural network

# In[18]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# In[19]:


model = Sequential([
    Dense(units=16, input_dim = 29,activation='relu'),
    Dense(units=24,activation='relu'),
    Dropout(0.5),
    Dense(20,activation='relu'),
    Dense(24,activation='relu'),
    Dense(1,activation='sigmoid'),
])


# In[20]:


model.summary()


# ## Training

# In[21]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=15,epochs=5)


# In[22]:


score = model.evaluate(X_test, y_test)


# In[23]:


print(score)


# In[29]:


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


# In[26]:


y_pred = model.predict(X_test)
y_test = pd.DataFrame(y_test)


# In[27]:


cnf_matrix = confusion_matrix(y_test, y_pred.round())


# In[28]:


print(cnf_matrix)


# In[30]:


plot_confusion_matrix(cnf_matrix, classes=[0,1])


# In[31]:


plt.show()

