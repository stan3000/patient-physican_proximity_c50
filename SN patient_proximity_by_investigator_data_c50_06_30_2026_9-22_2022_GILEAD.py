#!/usr/bin/env python
# coding: utf-8

# #  patient_proximity_by_investigator_data_c50

# # 

# In[345]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')

import geopandas as gpd
from shapely.geometry import Point, Polygon  # shapely deal with geometry to enable lat and log
# from mpl_toolkits.basemap import Basemap
import plotly.express as px
import folium

plt.rcParams['figure.figsize']=(13, 8)
 


# In[346]:


df=pd.read_csv(r'F:\Gilead\GILEAD_COMPETITION\GCOMPITION_DATA\Citeline_Export\patient_proximity_by_investigator_data_c50.csv',encoding=('iso-8859-1'))
pd.set_option('display.max_columns', None)
df.head(5)


# In[347]:


# RENANE ï»¿investigatorId

df.rename(columns = {'ï»¿investigatorId':'investigatorId'}, inplace = True)


# # FILE LENGTH

# In[348]:


df.head(5)


# In[349]:


df.shape


# In[350]:


df.isnull().sum()


# In[351]:


df['patientCount'].value_counts()[:20].plot(kind='barh')


# In[352]:


df['icd10Terms'].value_counts()[:20].plot(kind='barh')


# In[353]:


df['investigatorNPI'].value_counts()[:20].plot(kind='barh')
# Top Five investigatorNPI
df['investigatorNPI'].value_counts()[:8]


# In[354]:


df['icd10Code'].value_counts()[:10].plot(kind='barh')


# In[355]:


# GEOPANDAS VISUALAZATION
# https://www.youtube.com/watch?v=wAIolYrOAAU

plt.scatter(x=df['longitude'], y=df['latitude'])
plt.show()


# plt.rcParams['figure.figsize']=(18, 16)


# # PLOT DATA IN MAP
# https://towardsdatascience.com/geopandas-101-plot-any-data-with-a-latitude-and-longitude-on-a-map-98e01944b972

# In[356]:


crs={'init':'epsg:4326'}
geometry =[Point(xy) for xy in zip(df['longitude'], df['latitude'])]
geometry[:3]


# In[357]:


geo_df=gpd.GeoDataFrame(df,crs=crs,geometry=geometry)
geo_df.head(3)


# # DROP NAN

# In[358]:


# df1 = df.dropna(axis='columns')
# df1


# In[359]:


df.isnull().sum()


# In[360]:


# df=folium.Map(location=[13.13393247666733, 16103938729508073],zoom_start=2)
# df


# # DATA CLEANING - DATA WRINGLING

# In[361]:


#  PART 1: SPLIT TEXTS ON physicianPatientCount

new = df["physicianPatientCount"].str.split(" ", n = 12, expand = True)
new.head(2)


# In[362]:


# # GET COLUMN 3 (Status_New) & 4 (Remarks)
# # PART 2: making separate first name column from new data frame (Recruitment_status)

df1=df["physician_Proximity_15m"]= new[1]
df1=df["patient_Proximity_15m"]=new[3]
df1=df["physician_Proximity_25m"]= new[5]
df1=df["patient_Proximity_25m"]=new[7]
df1=df["physician_Proximity_50m"]= new[9]
df1=df["patient_Proximity_50m"]=new[11]


df1.head(3)


# In[363]:


# Dropping old Name columns
# df2=df1.drop(columns =['physicianPatientCount','patientProximity15m','physicianProximity25m','patientProximity25m','physicianProximity50m','patientProximity50m'], inplace = False)
df1.head(3)


# # Dropping old Name columns
df1=df.drop(columns =['physicianPatientCount'], inplace = False)
df1.head(3)


# In[364]:


# REMOVE COMMAS IN ALL COLUMS AND CURRY BRACKET

df2= df1.replace(',','', regex=True)
df3 = df2.replace('}',']', regex=True)
df3


# In[365]:


df3.corr()


# In[366]:


# print(df3.corr())


# In[367]:


# plotting correlation heatmap
dataplot = sns.heatmap(df3.corr(), cmap="YlGnBu", annot=True)


# # SPLIT icd10TERMS

# In[368]:


# # #  PART 1: SPLIT TEXTS ON icd10Terms

new11a = df3["icd10Terms"].str.split(" ", n = 10, expand = True)
new11a.head(5)


# In[369]:


# Clean icd10Terms and add to column

# df3["Malignant"]= new11a[0]
df3["neoplasm"]= new11a[1]
df3["total_Patient_upper_lower"]= new11a[3]
df3["total_Patient_quadrant_area"]= new11a[4]
df3["patient_Count_part"]=new11a[6]
df3["patient_Count_breast_type"]= new11a[7]
# df3["patient_count_type"]=new11a[8]
# # # df2=df1["physician_Proximity_50m"]= new[9]
# # # df2=df1["patient_Proximity_50m"]=new[11]

#=====NOTE =====> Unable to seperate patient_count_Famale from the patientCount female  <====

# CONBINE Malignant, neoplasm, of,upper-outer, quadrant
# df3['icd10Terms_update'] = df3['Malignant'] + ' ' + df3['neoplasm'] + ' ' + df3['of'] + ' ' + df3['upper-outer'] + ' ' + df3['quadrant']


# CONBINE total_Patient_quadrant_site and patient_Count_part
df3['icd10Terms_malignant'] = df3['neoplasm'] + ' ' + df3['total_Patient_upper_lower'] + ' ' + df3['total_Patient_quadrant_area'] + ' ' + df3['patient_Count_part']

df3.head(3)


# In[370]:


#THEN DROP total_Patient_quadrant_site and patient_count_type

df4=df3.drop(columns =['icd10Terms','Unnamed: 10','neoplasm','total_Patient_upper_lower','total_Patient_quadrant_area','patient_Count_part'], inplace = False)

df4.head(3)


# In[371]:


# # REMOVE the remaining CHARACTERS

df5 = df4.replace(']','', regex=True)
df5


# In[372]:


# # TAKE COLUMN WITHOUT NA on patient_Count_breast_type

df6= df5[df5['patient_Count_breast_type'].notna()]
df6.head(5)


# In[373]:


df6['patient_Count_breast_type'].value_counts()[:10].plot(kind='barh')


# In[374]:


# df3[df3["patient_Count_breast_type"].str.contains("']")==False]


# In[375]:


df6['physician_Proximity_15m'].value_counts()[:10].plot(kind='bar')


# In[376]:


# # COMPARE patient_Count_area_part and patient_Count_group
import textwrap
sns.countplot(x='icd10Terms_malignant', hue='patient_Count_breast_type', data=df6,)

# # df3.plot.bar()
# plt.xticks(wrap = False)


# In[377]:


# sns.boxplot(x='patient_Proximity_15m', data=df6)


# In[378]:


# CONVERT STR TO INT

df6.patient_Proximity_15m = df6.patient_Proximity_15m.astype(int)
df6.physician_Proximity_15m = df6.physician_Proximity_15m.astype(int)


# In[ ]:





# In[379]:


sns.boxplot(x='patient_Proximity_15m', data=df6)


# In[380]:


# OUTLIER TREATMENT FOR PATIENT

print(df6['patient_Proximity_15m'].quantile(0.50))
print(df6['patient_Proximity_15m'].quantile(0.90))


# In[381]:


df6['patient_Proximity_15m']=np.where(df6['patient_Proximity_15m']>8042.0, 907.0, df6['patient_Proximity_15m'])


# In[382]:


sns.boxplot(x='patient_Proximity_15m', data=df6)


# In[ ]:





# In[383]:


# PHYSICIAN OUTLIER

sns.boxplot(x='physician_Proximity_15m', data=df6)


# In[384]:


# OUTLIER TREATMENT

print(df6['physician_Proximity_15m'].quantile(0.50))
print(df6['physician_Proximity_15m'].quantile(0.90))


# In[385]:


df6['physician_Proximity_15m']=np.where(df6['physician_Proximity_15m']>1847.0, 285.0, df6['physician_Proximity_15m'])


# In[386]:


sns.boxplot(x='physician_Proximity_15m', data=df6)


# # PATIENT COUNT SPLIT

# In[387]:


# # #  PART 1: SPLIT TEXTS ON physicianPatientCount

new1 = df6["patientCount"].str.split(" ", n = 5, expand = True)
new1.head(5)


# In[388]:


# # # GET COLUMN 3 (Status_New) & 4 (Remarks)
# # # PART 2: making separate first name column from new data frame (Recruitment_status)

df6["total_Patient_Count"]= new1[1]
df6["female"]=new1[3]
df6["adult"]= new1[5]

df6


# In[389]:


# # Dropping old Name columns
df7=df6.drop(columns =['patientCount'], inplace = False)
df7.head(3)


# REMOVE COMMAS IN ALL COLUMS AND CURRY BRACKET

# df4 = df4.replace('{','', regex=True)
# df4 = df4.replace('}','', regex=True)
# df4


# In[390]:


df7['icd10Code'].value_counts()[:8].plot(kind='barh')
df7['icd10Code'].value_counts()[:8]


# In[420]:


df7['icd10Terms_malignant'].value_counts()[:8].plot(kind='barh')
df7['icd10Terms_malignant'].value_counts()[:8]


# In[421]:


df7['total_Patient_Count'].value_counts()[:8].plot(kind='barh')
df7['total_Patient_Count'].value_counts()[:8]


# In[422]:


df7.shape


# In[423]:


# DROP ROWS WITH N/A
df8=df7.dropna(subset=['physician_Proximity_25m','patient_Proximity_25m',
                       'physician_Proximity_50m','patient_Proximity_50m',
                       'total_Patient_Count','female','adult'])

df8.shape


# In[415]:


# How many data dropped?
234163 - 230794


# In[416]:


df8.head(3)


# In[417]:


# df6.fillna(df6.mean(), inplace=True)


# In[418]:


df8.isnull().sum()


# # EXPORT DATA

# In[419]:


df8.to_csv("patient_appromity_2.csv", index=False, header=True)


# In[400]:


df8.plot.scatter(x="physician_Proximity_50m", y="patient_Proximity_50m", alpha=0.5, c='red')

plt.show()


# In[1474]:


# DROP NA

# df6 = df5.dropna(axis='columns')
df8


# In[93]:


plt.savefig('proximaty.png')


# In[403]:


import folium
folium.Map(location=[48.130518, 11.5364172], zoom_start=12)


# In[404]:


m = folium.Map(location=[48.218871184761596, 11.624819877497147], zoom_start=15)

tooltip = "Click Here For More Info"

marker = folium.Marker(
    location=[48.218871184761596, 11.924819877497147],
    popup="<stong>Allianz Arena</stong>",
    tooltip=tooltip)
marker.add_to(m)

m


# In[ ]:





# # GROUP DIFFICULTIES RECRUITING AND PHASE

# In[36]:


from wordcloud import WordCloud, STOPWORDS

wordcloud=WordCloud(width =1000, height=500).generate(''.join(df['WhyStopped']))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[ ]:




