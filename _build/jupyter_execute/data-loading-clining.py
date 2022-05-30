#!/usr/bin/env python
# coding: utf-8

# # Data loading and clining
# ## Loading data
# We will use data stored in a CSV file. In order to process the data we will use Pandas framework.

# In[1]:


import pandas as pd
import numpy as np
from IPython.display import display

pd.set_option("display.precision", 2)
pd.options.display.max_columns = 50

CSV_FILENAME = './res/cleveland_data.csv'
names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']

raw_data = pd.read_csv(
    filepath_or_buffer = CSV_FILENAME,
    names = names,
    index_col = False
)


# ## Data cleaning
# ### Checking and removing incorrect data
# First of all, let's check if the dataset contains non-numeric data:

# In[2]:


dtypes = raw_data.dtypes
non_num_dtypes = dtypes[(dtypes != np.float64) & (dtypes != np.int64)]
print('non-numeric columns:\n', non_num_dtypes)


# After we have found that there are potentially incorrect data in columns ca and thal, let's check them

# In[3]:


print('unique values in "ca": ', raw_data['ca'].unique())
print('unique values in "thal": ', raw_data['thal'].unique())


# Now we can see that some of the fields contain the '?' symbol. Let's get rid of them:

# In[4]:


raw_data = raw_data.drop(raw_data[(raw_data['ca'] == '?') | (raw_data['thal'] == '?')].index)
raw_data['ca'] = pd.to_numeric(raw_data['ca']); raw_data['thal'] = pd.to_numeric(raw_data['thal'])
raw_data = raw_data.reset_index(drop = True)


# ### Splitting categorical fields
# We cannot directly use categorical parameters, because the numbers they contain do not quantify them, but only show the presence of a certain feature. Let's replace each of these parameters with a one-hot vector:

# In[5]:


cp_one_hot = pd.get_dummies(
    data = raw_data['cp'],
    dtype = np.float64
).set_axis(
    labels = ['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'],
    axis = 'columns'
)

thal_one_hot = pd.get_dummies(
    data = raw_data['thal'],
    dtype = np.float64
).set_axis(
    labels = ['thal norm', 'thal fixed def', 'thal reversable def'],
    axis = 'columns'
)

restecg_one_hot = pd.get_dummies(
    data = raw_data['restecg'],
    dtype = np.float64
).set_axis(
    labels = ['ecg norm', 'ecg ST-T abnormal', 'ecg hypertrophy'],
    axis = 'columns'
)

original_data = raw_data.drop(columns = ['cp', 'thal', 'restecg']).copy(deep = True)

original_data = pd.concat(
    objs = [original_data, cp_one_hot, thal_one_hot, restecg_one_hot],
    axis = 'columns',
    join = 'outer',
    ignore_index = False 
)


# Then we need to make some sence of the 'slope' values:

# In[6]:


# Update values: flat = 0.0, upsloping = 1.0, downsloping = -1.0
original_data.loc[original_data['slope'] == 1.0, 'slope']  = 1.0
original_data.loc[original_data['slope'] == 2.0, 'slope']  = 0.0
original_data.loc[original_data['slope'] == 3.0, 'slope']  = -1.0


# And finally, in the 'num' columns, assign a value of "0" to healthy patients, and a value of "1" to patients with heart disease (regardless the narrowing percentage):

# In[7]:


# Update values: 0.0 = no heart disease; 1.0 = heart disease
original_data.loc[original_data['num'] == 0.0, 'num']  = 0.0
original_data.loc[original_data['num'] > 0.0, 'num']  = 1.0

get_ipython().run_line_magic('store', 'original_data')

