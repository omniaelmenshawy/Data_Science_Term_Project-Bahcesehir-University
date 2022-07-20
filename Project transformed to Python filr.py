#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries:

# In[1]:


#pip install dataprep
#pip install folium
#pip install plotly
#pip install missingno
#!pip3 install catboost
#!pip3 install xgboost


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
#get_ipython().run_line_magic('matplotlib', 'inline')
import missingno as msno
import folium
from folium.plugins import HeatMap, MarkerCluster
from folium import Choropleth, Circle, Marker
import branca.colormap as cm
import plotly.express as px
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error
import catboost
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn import linear_model
from sklearn.linear_model import BayesianRidge


# In[3]:


df = pd.read_excel('//Users//omniaelmenshawy//Desktop///data-3.xlsx')
df.head(3)


# In[6]:


df.shape


# # EDA
# - Missing values
# - Duplicated values
# - Distribution
# - Skewness
# - Outliers

# In[4]:


df.describe().T.style.background_gradient(subset=['mean','std','50%','count'], cmap='PuBu')


# In[5]:


df.info()


# In[7]:


msno.matrix(df,sparkline=False)


# In[8]:


msno.bar(df)


# In[9]:


sns.heatmap(df.isnull(), cbar=False)
print("Total Missing: ", df.isna().sum().values.sum())


# In[10]:


df['Compund'].value_counts()


# In[11]:


df['Neighborhood'].value_counts()


# In[12]:


from dataprep.eda import create_report
report = create_report(df, title='My Report')


# In[13]:


report


# In[14]:


corr = df.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# In[15]:


df['Latitude'] = df['Latitude'].astype(float)


# In[16]:


ist_map = folium.Map(location=[41,29], zoom_start = 6, min_zoom=5)
df_map = df[['Latitude', 'Longitude']]
data = [[row['Latitude'],row['Longitude']] for index, row in df_map.iterrows()]
_ = HeatMap(data, radius=10).add_to(ist_map)
ist_map


# In[17]:


locations = list(zip(df.Latitude, df.Longitude))


# In[18]:


m = folium.Map(location=[41,29],width="%100",height="%100")
for i in range(len(locations)):
    folium.CircleMarker(location=locations[i],radius=1).add_to(m)
m


# In[19]:


istanbul =df[["Latitude","Longitude","Price"]]
min_price=df["Price"].min()
max_price=df["Price"].max()
min_price,max_price


# In[20]:


m = folium.Map(location=[41,29],width="%100",height="%100")
colormap = cm.LinearColormap(['green', 'yellow', 'red'],vmin=min_price, vmax=25000)
colormap = cm.StepColormap(colors=['green','yellow','orange','red'] ,index=[min_price,7000,10000,16500,max_price],vmin= min_price,vmax=max_price)
cm.LinearColormap.to_step

for loc, p in zip(zip(istanbul["Latitude"],istanbul["Longitude"]),istanbul["Price"]):
    folium.Circle(
        location=loc,
        radius=2,
        fill=True,
        color=colormap(p),
        #popup=
        fill_opacity=0.7
    ).add_to(m)
colormap.caption = 'Colormap Caption'

m.add_child(colormap)

m


# In[21]:


plt.figure(figsize=(15,6))
sns.barplot(x="Neighborhood",y="Price",data=df)
plt.xticks(rotation=90);
plt.grid();


# In[22]:


ax = sns.heatmap(corr, annot=True)


# In[23]:


df.columns


# # Data Preprocessing
# 
# - Missing
# - Outliers
# - Skewness
# - Encoding
# - Standadization
# - Dimentionality Reduction based on the Filtering Method

# ###  Handelling the Missing Vlaues and Dublicates

# In[24]:


def get_null_count(df):
    for i in df.columns:
        print(i,': ',len(df[df[i].isnull()][i]))


# In[25]:


get_null_count(df)


# In[26]:


df =df.dropna()


# In[27]:


df = df.drop_duplicates()


# ### Handelling Outliers

# In[28]:


fig = px.box(df, y="Price")
fig.show()


# In[29]:


df.drop(df[df['Price'] >= 26000].index, inplace = True)


# In[30]:


fig = px.box(df, y="Price")
fig.show()


# In[31]:


#df["Price"].describe()


# In[32]:


max_price=df["Price"].max()
m = folium.Map(location=[41,29],width="%100",height="%100")
colormap = cm.LinearColormap(['green', 'yellow', 'red'],vmin=min_price, vmax=25000)
colormap = cm.StepColormap(colors=['green','yellow','orange','red'] ,index=[min_price,7000,10000,16500,max_price],vmin= min_price,vmax=max_price)
cm.LinearColormap.to_step

for loc, p in zip(zip(istanbul["Latitude"],istanbul["Longitude"]),istanbul["Price"]):
    folium.Circle(
        location=loc,
        radius=2,
        fill=True,
        color=colormap(p),
        #popup=
        fill_opacity=0.7
    ).add_to(m)
colormap.caption = 'Colormap Caption'

m.add_child(colormap)

m


# ### Fixing the Skewness:

# In[33]:


df.hist(bins=50, figsize=(20,15))
plt.show();


# In[34]:


df.columns


# In[35]:


df[["Building Avg. Age", "Rooms + Salon","Bathrooms", "Net Area m^2","Price" ]].skew()


# In[36]:


df["Rooms + Salon"] = stats.boxcox(df["Rooms + Salon"])[0]
pd.Series(df["Rooms + Salon"]).skew()


# In[37]:


df["Bathrooms"] = stats.boxcox(df["Bathrooms"])[0]
pd.Series(df["Bathrooms"]).skew()


# In[38]:


df["Net Area m^2"] = stats.boxcox(df["Net Area m^2"])[0]
pd.Series(df["Net Area m^2"]).skew()


# In[39]:


fig = px.histogram(df, x="Price")
fig.show()


# In[40]:


df["Price"] = stats.boxcox(df["Price"])[0]
pd.Series(df["Price"]).skew()


# In[41]:


fig = px.histogram(df, x="Price")
fig.show()


# In[42]:


df.hist(bins=50, figsize=(20,15))
plt.show();


# ### Encoding categorical Values:

# In[43]:


df['Sea_View'] = df['Sea View']


# In[44]:


Sea = LabelEncoder()
df['Sea_View'] = Sea.fit_transform(df.Sea_View)
del df['Sea View']


# In[45]:


df['Neighborhood'] = Sea.fit_transform(df.Neighborhood)
df['Type'] = Sea.fit_transform(df.Type)
df['Furnished'] = Sea.fit_transform(df.Furnished)
df['Balcony'] = Sea.fit_transform(df.Balcony)
df['Compund'] = Sea.fit_transform(df.Compund)


# In[46]:


df.head(2)


# In[47]:


df.info()


# ### Scalling:

# In[48]:


transformer = RobustScaler().fit(df)
transformer.transform(df)


# ### Reduction:

# In[49]:


ax = sns.heatmap(corr, annot=True)


# In[50]:


df.columns


# In[51]:


df = df[['Neighborhood', 'Type', 'Building Avg. Age',
       'Rooms + Salon', 'Bathrooms', 'Furnished', 'Net Area m^2', 'Balcony',
       'Compund', 'Price', 'Sea_View']]


# In[52]:


df.head(1)


# # Data Modelling:
# 
# - Find Best regression models
# - try different training and testing sizes
# - evaluate the model 
# - Visualize the model
# 

# In[53]:


x = df.drop("Price",axis=1)
y = df["Price"]


# In[54]:


X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.10,random_state= 40)


# In[55]:


print("X_train shape {} and size {}".format(X_train.shape,X_train.size))
print("X_test shape {} and size {}".format(X_test.shape,X_test.size))
print("y_train shape {} and size {}".format(y_train.shape,y_train.size))
print("y_test shape {} and size {}".format(y_test.shape,y_test.size))


# ## Models:
# 

# In[56]:


lr = LinearRegression()

knn = KNeighborsRegressor(n_neighbors=7)

dt = DecisionTreeRegressor(max_depth = 7)

rf = RandomForestRegressor(max_depth = 7, n_estimators=700)
ada = AdaBoostRegressor( n_estimators=100, learning_rate =1)


gbr = GradientBoostingRegressor(max_depth=2, n_estimators=100, learning_rate =.2)

xgb = XGBRegressor(max_depth = 7, n_estimators=50, learning_rate =.2)

br = linear_model.BayesianRidge()

cat = CatBoostRegressor(iterations=10, learning_rate=1, verbose = 0)
         
regressors = [('Linear Regression', lr), ('K Nearest Neighbours', knn),
              ('Decision Tree', dt), ('Random Forest', rf), ('AdaBoost', ada),
              ('Gradient Boosting Regressor', gbr, ('CAT', cat), ('XGBoost', xgb), ('Bayesian Ridge', br))]


# # BayesianRidge

# In[57]:


br.fit(X_train,y_train)
predictions = br.predict(X_test)


# In[58]:


l = mean_absolute_percentage_error(y_test, predictions)
print('Mean Absolute Percentage Error is:' , l )
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print("Coefficient of determination score on training : ", br.score(X_train,y_train))
print("Coefficient of determination score on testing : ", br.score(X_test, y_test))


# In[59]:


test = pd.DataFrame({'Predicted':predictions,'Actual':y_test})
fig= plt.figure(figsize=(16,8))
test = test.reset_index()
test = test.drop(['index'],axis=1)
plt.plot(test[:50])
plt.legend(['Actual','Predicted'])
sns.jointplot(x='Actual',y='Predicted',data=test,kind="reg")


# # XGBRegressor

# In[60]:


xgb.fit(X_train,y_train)
predictions = xgb.predict(X_test)


# In[61]:


l = mean_absolute_percentage_error(y_test, predictions)
print('Mean Absolute Percentage Error is:' , l )
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print("Coefficient of determination score on training : ", xgb.score(X_train,y_train))
print("Coefficient of determination score on testing : ", xgb.score(X_test, y_test))


# In[62]:


test = pd.DataFrame({'Predicted':predictions,'Actual':y_test})
fig= plt.figure(figsize=(16,8))
test = test.reset_index()
test = test.drop(['index'],axis=1)
plt.plot(test[:50])
plt.legend(['Actual','Predicted'])
sns.jointplot(x='Actual',y='Predicted',data=test,kind="reg")


# # CatBoostRegressor

# In[63]:


cat.fit(X_train,y_train)
predictions = cat.predict(X_test)


# In[64]:


l = mean_absolute_percentage_error(y_test, predictions)
print('Mean Absolute Percentage Error is:' , l )
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print("Coefficient of determination score on training : ", cat.score(X_train,y_train))
print("Coefficient of determination score on testing : ", cat.score(X_test, y_test))


# In[65]:


test = pd.DataFrame({'Predicted':predictions,'Actual':y_test})
fig= plt.figure(figsize=(16,8))
test = test.reset_index()
test = test.drop(['index'],axis=1)
plt.plot(test[:50])
plt.legend(['Actual','Predicted'])
sns.jointplot(x='Actual',y='Predicted',data=test,kind="reg")


# ## ADA Boost Regressior

# In[66]:


ada.fit(X_train,y_train)
predictions = ada.predict(X_test)


# In[67]:


l = mean_absolute_percentage_error(y_test, predictions)
print('Mean Absolute Percentage Error is:' , l )
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print("Coefficient of determination score on training : ", ada.score(X_train,y_train))
print("Coefficient of determination score on testing : ", ada.score(X_test, y_test))


# In[68]:


test = pd.DataFrame({'Predicted':predictions,'Actual':y_test})
fig= plt.figure(figsize=(16,8))
test = test.reset_index()
test = test.drop(['index'],axis=1)
plt.plot(test[:50])
plt.legend(['Actual','Predicted'])
sns.jointplot(x='Actual',y='Predicted',data=test,kind="reg")


# # Gradient Boosting Regressor

# In[69]:


gbr.fit(X_train,y_train)
predictions = gbr.predict(X_test)


# In[70]:


l = mean_absolute_percentage_error(y_test, predictions)
print('Mean Absolute Percentage Error is:' , l )
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print("Coefficient of determination score on training : ", gbr.score(X_train,y_train))
print("Coefficient of determination score on testing : ", gbr.score(X_test, y_test))


# In[71]:


test = pd.DataFrame({'Predicted':predictions,'Actual':y_test})
fig= plt.figure(figsize=(16,8))
test = test.reset_index()
test = test.drop(['index'],axis=1)
plt.plot(test[:50])
plt.legend(['Actual','Predicted'])
sns.jointplot(x='Actual',y='Predicted',data=test,kind="reg")


# In[72]:


plt.scatter(y_test,predictions)


# # Linear Regression:

# In[74]:


lr.fit(X_train,y_train)
predictions = lr.predict(X_test)


# In[75]:


l = mean_absolute_percentage_error(y_test, predictions)
print('Mean Absolute Percentage Error is:' , l )
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print("Coefficient of determination score on training : ", lr.score(X_train,y_train))
print("Coefficient of determination score on testing : ", lr.score(X_test, y_test))


# In[76]:


test = pd.DataFrame({'Predicted':predictions,'Actual':y_test})
fig= plt.figure(figsize=(16,8))
test = test.reset_index()
test = test.drop(['index'],axis=1)
plt.plot(test[:50])
plt.legend(['Actual','Predicted'])
sns.jointplot(x='Actual',y='Predicted',data=test,kind="reg")


# # KNN

# In[77]:


knn.fit(X_train,y_train)
predictions = knn.predict(X_test)


# In[78]:


l = mean_absolute_percentage_error(y_test, predictions)
print('Mean Absolute Percentage Error is:' , l )
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print("Coefficient of determination score on training : ", knn.score(X_train,y_train))
print("Coefficient of determination score on testing : ", knn.score(X_test, y_test))


# In[79]:


test = pd.DataFrame({'Predicted':predictions,'Actual':y_test})
fig= plt.figure(figsize=(16,8))
test = test.reset_index()
test = test.drop(['index'],axis=1)
plt.plot(test[:50])
plt.legend(['Actual','Predicted'])
sns.jointplot(x='Actual',y='Predicted',data=test,kind="reg")


# In[80]:


plt.scatter(y_test,predictions)


# # DecisionTreeRegressor

# In[81]:


dt.fit(X_train,y_train)
predictions = dt.predict(X_test)


# In[82]:


l = mean_absolute_percentage_error(y_test, predictions)
print('Mean Absolute Percentage Error is:' , l )
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print("Coefficient of determination score on training : ", dt.score(X_train,y_train))
print("Coefficient of determination score on testing : ", dt.score(X_test, y_test))


# In[83]:


test = pd.DataFrame({'Predicted':predictions,'Actual':y_test})
fig= plt.figure(figsize=(16,8))
test = test.reset_index()
test = test.drop(['index'],axis=1)
plt.plot(test[:50])
plt.legend(['Actual','Predicted'])
sns.jointplot(x='Actual',y='Predicted',data=test,kind="reg")


# # Random Forest:

# In[84]:


rf.fit(X_train,y_train)
predictions = rf.predict(X_test)


# In[85]:


l = mean_absolute_percentage_error(y_test, predictions)
print('Mean Absolute Percentage Error is:' , l )
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print("Coefficient of determination score on training : ", rf.score(X_train,y_train))
print("Coefficient of determination score on testing : ", rf.score(X_test, y_test))


# In[86]:


test = pd.DataFrame({'Predicted':predictions,'Actual':y_test})
fig= plt.figure(figsize=(16,8))
test = test.reset_index()
test = test.drop(['index'],axis=1)
plt.plot(test[:50])
plt.legend(['Actual','Predicted'])
sns.jointplot(x='Actual',y='Predicted',data=test,kind="reg")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




