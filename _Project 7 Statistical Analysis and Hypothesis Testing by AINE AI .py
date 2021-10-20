#!/usr/bin/env python
# coding: utf-8

# Statistical Analysis and Hypothesis Testing

# In[2]:


import numpy as np
import pandas as pd                               #packages setup
import seaborn as sns
import matplotlib.pyplot as plt
import os

from scipy.stats import shapiro
import scipy.stats as stats



#parameter settings
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[3]:


#Read data using pandas
user_df=pd.read_csv("file:///C:/Users/lenovo/Desktop/cookie_cats.csv")

user_df


# In[10]:


#Check data types of each column using "dtypes" function
print("Data types for the data set:")
user_df.dtypes


# In[11]:


#Check dimension of data i.e. # of rows and #column using pandas "shape" funtion
print("Shape of the data i.e. no. of rows and columns")
user_df.shape


# In[15]:


#display first 5 rows of the data using "head" function
print("First 5 rows of the raw data:")
user_df.head(10)


# In[16]:


user_df.tail(10) #bottom 10 rows shown


# In[35]:


#Check for any missing values in the data using isnull() function
user_df.isnull().sum()


# In[38]:


user_df.userid.nunique()/user_df.shape[0]


# In[39]:


#Check for outlier values in sum_gamerounds column #outliers are data points that are far from other data points ,simply they are unusal values of dataset
plt.title("Total gamerounds played")
plt.xlabel("Index")
plt.ylabel("sum_gamerounds")
plt.plot(user_df.sum_gamerounds)


# # above graph is due to outlier it get highest values of index ,outliers are unusal data or not in simlar range

# In[42]:


#Based on the plot, filter out the outlier from sum_gamerounds played; Use max() fucntion to find the index of the outlier

print("max value of the sum_gamerounds")
max_value = max(user_df.sum_gamerounds)
max_value
print("index of the max value")
index_value = user_df[user_df.sum_gamerounds.isin([max_value])].index.tolist()
index_value


# In[46]:


#remove the row by index 
user_df.drop(user_df.index[index_value],inplace=True)


# In[47]:


#Plot the graph for sum_gamerounds player after removing the outlie

plt.title("Total gamerounds played") 
plt.xlabel("Index")
plt.ylabel("sum_gamerounds")
plt.plot(user_df.sum_gamerounds)


# #By the above graph droping outliers we get the needed data then the graph is formed,outliers are drop by drophing max value in the index

# In[49]:


#Insert calculation for 7-day retention rate

retention_rate_7=round((user_df.retention_7.sum()/user_df.shape[0])*100,2)
print("Overal 7 days retention rate of the game for both versions is: " ,retention_rate_7,"%")


# In[50]:


# Find number of customers with sum_gamerounds is equal to zero
user_df[user_df.sum_gamerounds == 0].shape[0]


# In[51]:


#Group by sum_gamerounds and count the number of users for the first 200 gamerounds
#Use plot() function on the summarized stats to visualize the chart

new_data = user_df[["userid","sum_gamerounds"]].groupby("sum_gamerounds").count().reset_index().rename(columns={"userid":"count"})[0:200]
plt.title("count of players vs sum_gamerounds ") 
plt.xlabel("sum_gamerounds")
plt.ylabel("count of players ")
plt.plot(new_data["sum_gamerounds"],new_data["count"])


# #groupby of users to know how many players are playing the sum_gamrounds

# In[52]:


#Create cross tab for game version and retention_7 flag counting number of users for each possible categories

pd.crosstab(user_df.version, user_df.retention_7).apply(lambda r: r/r.sum(), axis=1)


# In[3]:


#use pandas group by to calculate average game rounds played summarized by different version


user_df[["version","sum_gamerounds"]].groupby("version").agg("mean")


# In[15]:


#Define A/B groups for hypothesis testing
#user_df["version"] = np.where(user_df.version == "gate_30", "A", "B")
user_df["version"] = user_df["version"].replace(["gate_30","gate_40"],["A","B"])
group_A=pd.DataFrame(user_df[user_df.version=="A"]['sum_gamerounds'])
group_B=pd.DataFrame(user_df[user_df.version=="B"]['sum_gamerounds'])


# In[27]:


#---------------------- Shapiro Test ----------------------
# NULL Hypothesis H0: Distribution is normal                 #normal distribution is mean & median is Equal
# ALTERNATE Hypothesis H1: Distribution is not normal    

shapiro(group_A) #test for group_A
shapiro(group_B) #test for group_B


# In[35]:


#---------------------- Leven's Test ----------------------
# NULL Hypothesis H0: Two groups have equal variances
# ALTERNATE Hypothesis H1: Two groups do not have equal variances

#perform levene's test and accept or reject the null hypothesis based on the results

import scipy.stats as st

st.levene(group_A.sum_gamerounds, group_B.sum_gamerounds)


# In[37]:


#---------------------- Two samples test ----------------------
# NULL Hypothesis H0: Two samples are equal
# ALTERNATE Hypothesis H1: Two samples are different

#Apply relevant two sample test to accept or reject the NULL hypothesis

stats.mannwhitneyu(group_A.sum_gamerounds, group_B.sum_gamerounds,alternative="greater")



# In[39]:


#Analyze the 1 day and 7 days retention rate for two different groups using group by function
user_df[["version","retention_7","retention_1"]].groupby("version").agg("mean")


# In[5]:


list_1d = []
list_2d =[]
for i in range(500):
    boot_mean1 = user_df.sample(frac=0.7,replace=True).groupby('version')['retention_1'].mean()
    list_1d.append(boot_mean1.values)
    
    boot_mean2 = user_df.sample(frac=0.7,replace=True).groupby('version')['retention_7'].mean()
    list_2d.append(boot_mean2.values)
    
    
#transforming the list to dataframe
list_1d = pd.DataFrame(list_1d,columns=['gate_30','gate_40'])
list_2d = pd.DataFrame(list_2d,columns=['gate_30','gate_40'])



list_1d.plot(kind="kde",title="retention_1")
plt.show()
list_2d.plot(kind="kde",title="retention_7")
plt.show()


# #above graph we see difference in the graph due to density in retention 1 is alomost equal where as retention 7 varies due to high denstity in retention _1

# In[ ]:




