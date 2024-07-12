#!/usr/bin/env python
# coding: utf-8

# # Importing Dependencies

# In[30]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


# # Loading the dataset

# In[31]:


# Loading the dataset to Pandas DataFrame
loan_data= pd.read_csv("loandata.csv")


# In[32]:


# Printing the first 5 rows of the dataset
loan_data.head()


# In[33]:


# Number of rows and colums
loan_data.shape


# In[34]:


# statistical measures
loan_data.describe()


# In[35]:


loan_data.info()


# # Missing value imputation
# let's list out the feature-wise count of missing values.

# In[36]:


loan_data.isnull().sum()


# In[37]:


# fill the missing values for numerical terms - mean
loan_data['LoanAmount'] = loan_data['LoanAmount'].fillna(loan_data['LoanAmount'].mean())
loan_data['Loan_Amount_Term'] = loan_data['Loan_Amount_Term'].fillna(loan_data['Loan_Amount_Term'].mean())
loan_data['Credit_History'] = loan_data['Credit_History'].fillna(loan_data['Credit_History'].mean())


# In[38]:


# fill the missing values for categorical terms - mode
loan_data['Gender'] = loan_data["Gender"].fillna(loan_data['Gender'].mode()[0])
loan_data['Married'] = loan_data["Married"].fillna(loan_data['Married'].mode()[0])
loan_data['Dependents'] = loan_data["Dependents"].fillna(loan_data['Dependents'].mode()[0])
loan_data['Self_Employed'] = loan_data["Self_Employed"].fillna(loan_data['Self_Employed'].mode()[0])


# In[39]:


# number of missing values in each column
loan_data.isnull().sum()


# # Exploratory Data Analysis

# In[40]:


# categorical attributes visualization
print(loan_data['Gender'].value_counts())
sns.countplot(x='Gender',data=loan_data,palette='Set1')


# In[41]:


print(loan_data['Married'].value_counts())
sns.countplot(x='Married',data=loan_data,palette='Set1')


# In[42]:


print(loan_data['Education'].value_counts())
sns.countplot(x='Education',data=loan_data,palette='Set1')


# In[43]:


# Label encoding : Converting categorical Loan Status column to numerical values.
loan_data.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)


# In[44]:


# printing the first 5 rows of the dataframe
loan_data.head()


# In[45]:


# Dependent column values
loan_data['Dependents'].value_counts()


# In[46]:


# 3+ is not a good data type so we cannot feed this 3+ value to our model.So we will replace 3+ to general value of 4.
loan_data = loan_data.replace(to_replace='3+', value=4)


# In[47]:


# dependent values
loan_data['Dependents'].value_counts()


# # Bivariate Analysis 

# Categorical Independent Variable vs Target Variable

# In[48]:


# education & Loan Status
sns.countplot(x='Education',hue='Loan_Status',data=loan_data)


# In[49]:


# marital status & Loan Status
sns.countplot(x='Married',hue='Loan_Status',data=loan_data)


# In[50]:


# Self Employed & Loan Status
sns.countplot(x='Self_Employed',hue='Loan_Status',data=loan_data)


# In[51]:


# education & Loan Status
sns.countplot(x='Dependents',hue='Loan_Status',data=loan_data)


# In[52]:


# education & Loan Status
sns.countplot(x='Gender',hue='Loan_Status',data=loan_data)


# # Independent Variable (Numerical)
# 

# Till now we have seen the categorical and ordinal variables and now lets visualize the numerical variables. Lets look at the distribution of Applicant income first.

# In[53]:


# Creating a distribution plot to visualize the distribution of applicant incomes in the 'loan_data' dataframe\n",
# using the seaborn library's distplot() function\n",
sns.distplot(loan_data['ApplicantIncome'])
plt.show()


#Creating a box plot to visualize the distribution of applicant incomes in the 'loan_data' dataframe
# using the pandas library's plot.box() function
# and setting the figure size to 16x5 using the figsize parameter
loan_data['ApplicantIncome'].plot.box(figsize=(16,5))
plt.show()


# It can be inferred that most of the data in the distribution of applicant income is towards left which means it is not normally distributed. We will try to make it normal in later sections as algorithms works better if the data is normally distributed.
# 
# The boxplot confirms the presence of a lot of outliers/extreme values. This can be attributed to the income disparity in the society. Part of this can be driven by the fact that we are looking at people with different eduation levels. Let us segregate them by Education.

# In[54]:


# Creating a box plot to visualize the distribution of applicant incomes in the 'loan_data' dataframe
# based on the 'Education' column\n",
# using the pandas library's boxplot() function\n",
loan_data.boxplot(column='ApplicantIncome', by='Education') 
plt.suptitle("")


# We can see that there are a higher number of graduates with very high incomes, which are appearing to be outliers.

# In[55]:


#Let's look at the Coapplicant income distribution.
sns.distplot(loan_data['CoapplicantIncome'])
plt.show()
loan_data['CoapplicantIncome'].plot.box(figsize=(16,5))
plt.show()


# We see a similar distribution as that of the applicant's income. The majority of co-applicants income ranges from 0 to 5000. We also see a lot of outliers in the applicant's income and it is not normally distributed.

# In[56]:


sns.distplot(loan_data['LoanAmount'])
plt.show()

loan_data['LoanAmount'].plot.box(figsize=(16,5))
plt.show()


# We see a lot of outliers in this variable and the distribution is fairly normal. We will treat the outliers in later sections.

#  Now let's look at the correlation between all numerical variables.We will use the heatmap to visualize the correlation. Heatmaps visualize data through variations in coloring. The variable with darker colors mean their correlation is more.

# In[57]:


matrix=loan_data.corr()
plt.subplots(figsize=(9,6))
sns.heatmap(matrix,vmax=.8,square=True,cmap='BuPu',annot=True)


#  We can see that the most corelated variables are (Applicant Income-Loan Amount and (Credit history-Loan Status).

# # Outlier Treatment

# In[59]:


loan_data['LoanAmount_log']=np.log(loan_data['LoanAmount'])
loan_data['LoanAmount_log'].hist(bins=20)


#  Now the distribution looks much closer to normal and effect of extreme values has been significantly subsided.

# In[ ]:




