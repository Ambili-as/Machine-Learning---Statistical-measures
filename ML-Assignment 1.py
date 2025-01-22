#!/usr/bin/env python
# coding: utf-8

# #### You are given house_price.csv which contains property prices in the city of Bangalore.
# 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# #### Load the dataset

# In[2]:


data = pd.read_csv(r"C:\Users\user\Downloads\house_price.csv")
data


# #### 1. Perform basic EDA 

# ##### Summary Statistics

# In[8]:


data.describe()


# In[9]:


data.info()


# ##### Checking null values

# In[7]:


data.isnull().sum()     


# ##### Checking Duplicate values

# In[8]:


data.duplicated().sum()


# ##### Removing duplicated values

# In[3]:


data=data.drop_duplicates()
data


# In[10]:


data.describe()


# #### Plotting Distributions and Relations

# ##### 1.  Distribution of price

# In[26]:


sns.histplot(data['price'],bins = 50, kde=True)
plt.title("Distribution of Price")
plt.show()


# ##### Here, most prices are concentrated on the lower end of the price scale which is 0-500. This indicates that most of the houses are inexpensive, except some outliers visible in the plot showing higher price

# #### 2. Distribution of Sqr feet

# In[27]:


sns.histplot(data['total_sqft'],bins = 50, kde=True)
plt.title("Distribution of Total Sqrft")
plt.show()


# ##### The histplot showing distribution of total square feet indicates that most of the houses have relatively smaller square feets but also outliers having higher values are present.

# ##### 3.Distribution of price per sqrft

# In[28]:


sns.histplot(data['price_per_sqft'],bins = 50, kde=True)
plt.title("Distribution of Price per Sqrft")
plt.show()


# #### The plot of price per sqrfeet indicates most houses have less price per sqrfeet

# #### 4.Price Vs total Sqrfeet

# In[31]:


sns.scatterplot(x='total_sqft',
                y='price',
                data = data)
plt.title("Price vs Total Sqr feet")
plt.show()


# ##### The scatter plot showing distribution of price vs square feet indicates a positive trend of higher price for larger houses excluding some outliers having larger square feet with very less price

# #### 5.Box plot for price_per_sqft

# In[32]:


sns.boxplot(x=data['price_per_sqft'])
plt.title("Boxplot for price per sqr feet")
plt.show()


# ##### The box plot shows several outliers and a high variability in price per sq.ft.

# #### Q2. Detect the outliers using following methods and remove it using methods like trimming / capping/ imputation using mean or median (Score: 4)  a) Mean and Standard deviation b)Percentile method c) IQR(Inter quartile range method) d) Z Score method

# In[2]:





# In[103]:


# Detecting outliers using various methods
price_per_sqft = data['price_per_sqft']

# a) Mean & Standard Deviation
mean = price_per_sqft.mean()
std_dev = price_per_sqft.std()
lower_bound = mean - 3 * std_dev
upper_bound = mean + 3 * std_dev
print("mean: ",mean)
print("standard deviation: ",std_dev)
print("lower bound: ",lower_bound)
print("upper bound: ",upper_bound)
#finding outliers:
data[(price_per_sqft>lower_bound) | (price_per_sqft<upper_bound)]
#removing outliers:
data_mean_std_filtered = data[(price_per_sqft>lower_bound) & (price_per_sqft<upper_bound)]
print("original data shape: ",data.shape)
print("filtered data shape: ",data_mean_std_filtered.shape)


# In[ ]:





# #### plotting the data after removing outliers

# In[105]:


plt.subplot(1,2,1)
sns.boxplot(x=data['price_per_sqft'])
plt.title("Original data")
plt.subplot(1,2,2)
sns.boxplot(x=data_mean_std_filtered['price_per_sqft'])
plt.title("Distribution of price per sqft after filtering")
plt.show()


# In[66]:


data['price_per_sqft'].skew()


# In[67]:


data_mean_std_filtered['price_per_sqft'].skew()


# In[6]:


# b) Percentile
price_per_sqft = data['price_per_sqft']

#defining the percentile thresholds:
lower_percentile = 0.05
upper_percentile = 0.95
 
#calculating the bounds:
lower_bound = price_per_sqft.quantile(lower_percentile)
upper_bound = price_per_sqft.quantile(upper_percentile)

print("Lower bound: ",lower_bound)
print("Upper bound: ",upper_bound)

#Finding ouliers:
data[(price_per_sqft>=lower_bound)| (price_per_sqft<=upper_bound)]
#removing outliers:
data_percentile_filtered = data[(price_per_sqft>=lower_bound) & (price_per_sqft<=upper_bound)]

print("original data shape: ",data.shape)
print("filtered data shape: ",data_percentile_filtered.shape)


# In[7]:


#Plotting data after outlier removal:
plt.subplot(1,2,1)
sns.boxplot(x=data['price_per_sqft'])
plt.title("Original data")
plt.subplot(1,2,2)
sns.boxplot(x=data_percentile_filtered['price_per_sqft'])
plt.title("Distribution of price per sqft after filtering")
plt.show()


# In[98]:


data['price_per_sqft'].skew()


# In[99]:


data_percentile_filtered['price_per_sqft'].skew()


# In[4]:


# c) IQR method
Q1 = data['price_per_sqft'].quantile(0.25)
Q3 = data['price_per_sqft'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1-1.5*IQR
upper_bound = Q3+1.5*IQR
print("Lower bound: ", lower_bound)
print("Upper bound: ",upper_bound)
#finding outliers:
data[(data['price_per_sqft']>= lower_bound) | (data['price_per_sqft']<= upper_bound)]
#removing outliers
data_iqr_cleaned = data[(data['price_per_sqft']>= lower_bound) & (data['price_per_sqft']<= upper_bound)]
print("Original datashape: ", data.shape)
print("Data Filtered applying IQR: ",data_iqr_cleaned.shape)


# In[5]:


#plotting the difference:
plt.subplot(1,2,1)
sns.boxplot(x = data['price_per_sqft'])
plt.title("Original data")
plt.subplot(1,2,2)
sns.boxplot(x = data_iqr_cleaned['price_per_sqft'])
plt.title("Data after applying IQR method")
plt.show()


# In[10]:


data['price_per_sqft'].skew()


# In[11]:


data_iqr_cleaned['price_per_sqft'].skew()


# In[12]:


# d) Z-Score method
#finding the limits
price_per_sqft = data['price_per_sqft']
upper_limit = price_per_sqft.mean() + 3 * price_per_sqft.std()
lower_limit = price_per_sqft.mean() - 3 * price_per_sqft.std()
print("Upper limit: ",upper_limit)
print("Lower limit: ",lower_limit)

#finding outliers:
data.loc[(price_per_sqft>=upper_limit) | (price_per_sqft<=lower_limit)]
#removing outliers:
cleaned_data = data.loc[(price_per_sqft<=upper_limit) & (price_per_sqft>=lower_limit)]
print("Original data: ",len(data))
print("Cleaned data: ",len(cleaned_data))


# In[13]:


#Plotting the difference

plt.subplot(1,2,1)
sns.boxplot(x=price_per_sqft)
plt.title("Original data")
plt.subplot(1,2,2)
sns.boxplot(x=cleaned_data['price_per_sqft'])
plt.title("Data after applying z-score")
plt.show()


# In[84]:





# In[14]:


cleaned_data['price_per_sqft'].skew()


# In[61]:


data['price_per_sqft'].skew()


# ##### Q4. Draw histplot to check the normality of the column(price per sqft column) and perform transformations if needed. Check the skewness and kurtosis before and after the transformation.

# In[17]:


# drawing histplot:
sns.histplot(x = data['price_per_sqft'], kde = True, bins = 50 )
plt.title("Price per sqft histplot")
plt.show()


# #### The histplot clearly shows high skewness in data with most of the values concentrated near the lower end and a few outliers in the higher end.So this distribution is clearly not a normal one.

# #### Applying transformation: its better to use outliers removed data for transformation. So let's use the data obtained after removing outliers using IQR method.
# 

# In[19]:


len(data_iqr_cleaned['price_per_sqft'])


# In[20]:


# checking skewness:
data_iqr_cleaned['price_per_sqft'].skew()


# In[12]:


#Applying log transformation:
data_iqr_cleaned['price_per_sqft_log'] = np.log(data_iqr_cleaned['price_per_sqft'] + 1)

sns.histplot(x = data_iqr_cleaned['price_per_sqft_log'], kde = True, bins = 50)
plt.title("Histplot after transformation")
plt.show()


# In[23]:


# skewness after applying transformation:
data_iqr_cleaned['price_per_sqft_log'].skew()


# In[11]:


# kurtosis checking before transformation:
data['price_per_sqft'].kurt()


# In[13]:


# kurtosis checking after transformation:
data_iqr_cleaned['price_per_sqft_log'].kurt()


# #### From the above plotted histplot and skewness obtained after applying transformation, it is clearly visible that price_per_sqft is almost normally distributed now.

# #### Q5. Check the correlation between all the numerical columns and plot heatmap

# In[17]:


#Checking the correlation:
correlation = data.corr()
correlation


# In[19]:


#Plotting the correlation:
sns.heatmap(correlation, annot = True)
plt.title("Correaltion heatmap of numerical values")
plt.show()


# #### Q6. Draw Scatter plot between the variables to check the correlation between them.

# In[31]:


plt.subplot(2,2,1)
sns.scatterplot(data = data, x='price_per_sqft',y='bhk')
plt.title("Price per Sqft Vs bhk")

plt.subplot(2,2,2)
sns.scatterplot(data = data, x='price_per_sqft',y='total_sqft')
plt.title("Price per Sqft Vs Total Sqft")

plt.subplot(2,2,3)
sns.scatterplot(data = data, x='price_per_sqft',y='price')
plt.title("Price per Sqft Vs Price")

plt.subplot(2,2,4)
sns.scatterplot(data = data, x='price_per_sqft',y='bath')
plt.title("Price per Sqft Vs bath")
plt.tight_layout()
plt.show()


# #### The Plots shows no clear linear correlation betweeen the variables.
