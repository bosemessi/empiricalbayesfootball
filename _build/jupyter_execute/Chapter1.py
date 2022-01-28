#!/usr/bin/env python
# coding: utf-8

# # Collecting data from fbref
# 
# We will start by collecting the shots data from fbref 2017-18 onwards (since that has the advanced stats like xG) for the top 5 leagues. First let's import the necessary packages.

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from scipy.stats import beta
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd 
import numpy as np
import matplotlib.cm as cm


# We will combine the player name and birth year to create unique combinations, drop goalkeepers

# In[2]:


urls = ['https://fbref.com/en/comps/Big5/shooting/players/Big-5-European-Leagues-Stats',
       'https://fbref.com/en/comps/Big5/2020-2021/shooting/players/2020-2021-Big-5-European-Leagues-Stats',
       'https://fbref.com/en/comps/Big5/2019-2020/shooting/players/2019-2020-Big-5-European-Leagues-Stats',
       'https://fbref.com/en/comps/Big5/2018-2019/shooting/players/2018-2019-Big-5-European-Leagues-Stats',
       'https://fbref.com/en/comps/Big5/2017-2018/shooting/players/2017-2018-Big-5-European-Leagues-Stats']

dfs = []
for url in urls:
    df = pd.read_html(url)[0]
    df.columns = [c[1] if 'Unnamed' in c[0] else c[0]+'_'+c[1] for c in df.columns]
    df = df[['Player','Pos','Born','90s','Standard_Gls','Standard_Sh','Standard_PK','Expected_npxG']]
    df = df[(df.Player != "Player") & (df.Pos.notna())]
    df = df[~df.Pos.str.contains('GK')].reset_index(drop=True)
    df['Player'] = df['Player'] + ' (' + df['Born'] + ')'
    for cols in ['90s','Standard_Gls','Standard_Sh','Standard_PK','Expected_npxG']:
        df[cols] = df[cols].astype(float)
    df.fillna(value=0.0, inplace=True)
    dfs.append(df)
    
df = pd.concat(dfs, ignore_index=True)


# In[3]:


df.head()


# We will use the unique combinations (hopefully) of name + birth year to groupby and get total shots, total non-penalty goals, total non-penalty xG, Age, total 90s played etc.

# In[4]:


gdf = df.groupby('Player').sum().reset_index()
gdf.columns = ['Player','90s','Goals','Shots','PKs','npxG'] 
gdf['npG'] = gdf['Goals'] - gdf['PKs']
gdf = gdf[gdf.Player!=0.0].reset_index(drop=True)
gdf['Born'] = [float(gdf['Player'][i].split('(')[1].split(')')[0]) for i in range(len(gdf))]
gdf['Age'] = 2021 - gdf['Born']
gdf.to_csv('fbrefshootingdata.csv', index=False, encoding='utf-8-sig')


# In[5]:


gdf

