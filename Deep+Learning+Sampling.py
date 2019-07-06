
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


# In[12]:


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# ## Deep neural network

# In[13]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# In[14]:


model = Sequential([
    Dense(units=16, input_dim = 29,activation='relu'),
    Dense(units=24,activation='relu'),
    Dropout(0.5),
    Dense(20,activation='relu'),
    Dense(24,activation='relu'),
    Dense(1,activation='sigmoid'),
])


# In[15]:


model.summary()


# ## Training

# In[16]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=15,epochs=5)


# In[17]:


score = model.evaluate(X_test, y_test)


# In[18]:


print(score)


# In[19]:


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


# In[20]:


y_pred = model.predict(X_test)
y_test = pd.DataFrame(y_test)


# In[21]:


cnf_matrix = confusion_matrix(y_test, y_pred.round())


# In[28]:


print(cnf_matrix)


# In[30]:


plot_confusion_matrix(cnf_matrix, classes=[0,1])


# In[31]:


plt.show()


# In[22]:


y_pred = model.predict(X)
y_expected = pd.DataFrame(y)
cnf_matrix = confusion_matrix(y_expected, y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])
plt.show()


# ## Undersampling

# In[23]:


fraud_indices = np.array(data[data.Class == 1].index)
number_records_fraud = len(fraud_indices)
print(number_records_fraud)


# In[24]:


normal_indices = data[data.Class == 0].index


# In[25]:


random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
random_normal_indices = np.array(random_normal_indices)
print(len(random_normal_indices))


# In[26]:


under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])
print(len(under_sample_indices))


# In[27]:


under_sample_data = data.iloc[under_sample_indices,:]


# In[28]:


X_undersample = under_sample_data.iloc[:,under_sample_data.columns != 'Class']
y_undersample = under_sample_data.iloc[:,under_sample_data.columns == 'Class']


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X_undersample,y_undersample, test_size=0.3)


# In[33]:


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# In[34]:


model.summary()


# In[35]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=15,epochs=5)


# In[36]:


y_pred = model.predict(X_test)
y_expected = pd.DataFrame(y_test)
cnf_matrix = confusion_matrix(y_expected, y_pred.round())
plot_confusion_matrix(cnf_matrix, classes=[0,1])
plt.show()


# In[37]:


y_pred = model.predict(X)
y_expected = pd.DataFrame(y)
cnf_matrix = confusion_matrix(y_expected, y_pred.round())
plot_confusion_matrix(cnf_matrix, classes=[0,1])
plt.show()


# ## SMOTE

# In[38]:


get_ipython().run_cell_magic('bash', '', 'pip install -U imbalanced-learn')


# In[39]:


from imblearn.over_sampling import SMOTE


# In[40]:


X_resample, y_resample = SMOTE().fit_sample(X,y.values.ravel())


# In[41]:


y_resample = pd.DataFrame(y_resample)
X_resample = pd.DataFrame(X_resample)


# In[42]:


X_train, X_test, y_train, y_test = train_test_split(X_resample,y_resample,test_size=0.3)


# In[43]:


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# In[44]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=15,epochs=5)


# In[45]:


y_pred = model.predict(X_test)
y_expected = pd.DataFrame(y_test)
cnf_matrix = confusion_matrix(y_expected, y_pred.round())
plot_confusion_matrix(cnf_matrix, classes=[0,1])
plt.show()


# In[46]:


y_pred = model.predict(X)
y_expected = pd.DataFrame(y)
cnf_matrix = confusion_matrix(y_expected, y_pred.round())
plot_confusion_matrix(cnf_matrix, classes=[0,1])
plt.show()

