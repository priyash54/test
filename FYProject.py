#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sci

get_ipython().run_line_magic('matplotlib', 'inline')


# Current directory

# In[54]:


pwd


# Calling the csv file

# In[55]:


data = pd.read_csv('history_data.csv')


# In[56]:


data.dtypes


# In[57]:


data.head()


# In[58]:


x = data["Wind Chill"]
y = data["Temperature"]


# Scatter plot

# In[59]:


plt.scatter(x,y,c="r",s=10, marker = '*')
plt.xlabel('Wind Chill')
plt.ylabel('Temperature')


# Function to generate coefficient values

# In[60]:


def coeficient(x,y):
    
    n = np.size(x)
    mx,my = np.mean(x), np.mean(y)
    
    #sum of cross deviations of y and x
    CDxy = np.sum(x*y) - n*mx*my
    
    #sum of sqaured deviations of x
    CDxx = np.sum(x*x) - n*(mx*mx)
    
    #coefficient values
    b1 = CDxy/CDxx
    b0 = my - b1*mx
    
    return(b1,b0)


# Coefficient values

# In[61]:


b = coeficient(x,y)
b


# In[62]:


def plot_results(x,y,b):
    plt.scatter(x,y,c="r",s=10, marker = '*')
    #plt.xlabel(x)
    #plt.ylabel(y)
    
    ypred = b[1] + b[0]*x
    
    plt.plot(x,ypred,c='b', linewidth =2)
    plt.xlabel('Wind Chill')
    plt.ylabel('Temperature')
        
    plt.show()


# Plotting the regression line with data points

# In[63]:


plot_results(x,y,b)


# # Using Seaborn library

# In[64]:


import seaborn as sns
sns.set()
data.head()


# In[65]:


np.unique(data["Maximum Temperature"])


# In[66]:


data["Maximum Temperature"].plot(kind="hist")


# In[67]:


sns.distplot(data["Maximum Temperature"])


# In[68]:


df = data[["Wind Speed","Temperature","Cloud Cover","Relative Humidity"]]
sns.pairplot(df,height=5)

#scatterplot using seaborn
#sns.scatterplot(x="Maximum Temperature",y="Temperature", data=df)


# In[69]:


sns.jointplot(x="Wind Speed",y="Temperature",data=df)


# In[70]:


sns.lmplot(x="Wind Speed",y="Temperature",data=df)


# In[71]:


sns.set(rc={'figure.figsize':(10,5)})
ax=sns.boxplot(x='Wind Speed',y='Temperature', data=df)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

# ax=sns.swarmplot(x='Maximum Temperature',y='Temperature',data=df)
#ax=sns.countplot(x='Mfr Name', data=df)
#ax=sns.countplot(x='Mfr Name', data=df)
# ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
# df2 = df.pivot_table(index='Cylinders', columns='Eng Displ', values='CombMPG', aggfunc='mean')
# sns.heatmap(df2)


# # Different models

# In[72]:


df.head()


# In[73]:


#Linear regression plot for individual variables with temperature
# sns.lmplot(x="Temperature",y="Wind Chill",data=data)
# sns.lmplot(x="Temperature",y="Cloud Cover",data=data)
# sns.lmplot(x="Temperature",y="Relative Humidity",data=data)

x = data["Wind Chill"]
gamma = [8,3,4,5]
y1 = gamma[0] + gamma[1]*x +gamma[2]*x*x + gamma[3]*x*x*x + np.random.normal(0,1,len(x))
sns.scatterplot(x,y1)


# In[99]:


dat1 = pd.DataFrame(y1)
dat1.insert(1,"WC",data["Wind Chill"])
#sns.lmplot(x="WC",y="Wind Chill", data=dat1)
b=coeficient(data["Wind Chill"],y1)
plot_results(data["Wind Chill"],y1,b)


# # Multiple Linear Regression model

# In[75]:


data.head()


# In[76]:


import statsmodels.api as sm

y = data["Temperature"]
x = data[["Wind Chill","Wind Speed","Cloud Cover"]]
X = sm.add_constant(x)

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X[["Wind Chill","Wind Speed","Cloud Cover"]] = scale.fit_transform(X[["Wind Chill","Wind Speed","Cloud Cover"]].values)

mod = sm.OLS(y,X).fit()

mod.summary()


# R-squared = 0.829 which is very close to 1. \
# The p-value for 'Cloud Cover' is greater than 0.05, so it is less significant than the rest of the variables.

# In[77]:


X


# In[78]:


mod.params


# In[79]:


a=mod.params[0]
b=mod.params[1]
c=mod.params[2]
d=mod.params[3]
Y= a + b*X["Wind Chill"] + c*X["Wind Speed"] + d*X["Cloud Cover"]
Y


# In[80]:


np.corrcoef(y,Y)


# In[ ]:





# # Using curve fit

# model creation and coefficient calculation

# In[81]:


import scipy as sci
from scipy import optimize

def model(x,m,b):
    return m + x*b

initialval = [1,1]
fit = sci.optimize.curve_fit(model,data["Temperature"], data["Wind Chill"],p0 =initialval)


# In[82]:


fit


# In[83]:


ans,cov = fit


# In[84]:


ans


# In[85]:


cov


# In[86]:


plt.errorbar(data["Temperature"], data["Wind Chill"], fmt = 'b.', label = "linear relation")

t = np.linspace(35,53)
plt.plot(t,model(t,ans[0],ans[1]), label = "regression line")

plt.xlabel("Wind Chill")
plt.ylabel("Temperature")


# # LR using Stats package

# In[87]:


from scipy import stats
slope,intercept,rvalue,pvalue,stderr = stats.linregress(data["Maximum Temperature"],data["Temperature"])


# In[88]:


def predict(x):
    return slope*x + intercept

sns.scatterplot(data["Maximum Temperature"],data["Temperature"])
sns.lineplot(data["Maximum Temperature"], predict(data["Maximum Temperature"]),c='r')


# In[89]:


sns.scatterplot(data["Relative Humidity"],data["Temperature"])
fit = np.polyfit(data["Relative Humidity"],data["Temperature"],3)

xp = np.linspace(50,100,50)
yp = np.poly1d(fit)        #it is a convienient way of getting the predicted value
#https://numpy.org/doc/stable/reference/generated/numpy.poly1d.html
sns.lineplot(xp,yp(xp),c='r')


# In[90]:


yp


# # AHU data

# In[91]:


dat = pd.read_csv('all_in_one_finish_20130219_noblanks.csv')


# In[92]:


dat.head(2)
#dat.tail(5)


# In[93]:


dat.columns, dat.shape
#dat[['Date / Time','AHU9 CHW Offcoil Temperature (1 minute)','AHU9 CHW Valve Position (1 minute)']][:5]
#len(dat)


# In[94]:


dat.sort_values(['Date / Time'])


# In[95]:


dat.describe()


# In[96]:


countVP = dat['AHU9 CHW Valve Position (1 minute)'].value_counts()
countVP


# In[97]:


countVP.plot(kind='pie')


# In[ ]:




