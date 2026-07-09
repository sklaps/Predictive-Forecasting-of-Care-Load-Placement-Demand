# Predictive-Forecasting-of-Care-Load-Placement-Demand
This project focuses on predicting future care load and placement demand for the Unaccompanied Alien Children (UAC) Program managed by the U.S. Department of Health and Human Services (HHS).
## 🎯 Project Objectives
Primary Objectives
Forecast future Children in HHS Care
Predict future discharge (placement) demand
Analyze imbalance between incoming transfers and discharges
Secondary Objectives
Compare statistical and machine learning forecasting models
Quantify forecast uncertainty
Provide operational insights for healthcare planners
Develop an interactive Streamlit dashboard

---

## 📚 Import Required Libraries




```python
import pandas as pd
```


```python

```


```python
df = pd.read_csv('Predictive Forecasting of Care Load & Placement Demand.csv')
```


```python
df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Children apprehended and placed in CBP custody*</th>
      <th>Children in CBP custody</th>
      <th>Children transferred out of CBP custody</th>
      <th>Children in HHS Care</th>
      <th>Children discharged from HHS Care</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>December 21, 2025</td>
      <td>6</td>
      <td>18</td>
      <td>11</td>
      <td>2,484</td>
      <td>14</td>
    </tr>
    <tr>
      <th>1</th>
      <td>December 18, 2025</td>
      <td>11</td>
      <td>50</td>
      <td>6</td>
      <td>2,472</td>
      <td>16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>December 17, 2025</td>
      <td>7</td>
      <td>31</td>
      <td>11</td>
      <td>2,481</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>December 16, 2025</td>
      <td>8</td>
      <td>54</td>
      <td>15</td>
      <td>2,468</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>December 15, 2025</td>
      <td>11</td>
      <td>42</td>
      <td>9</td>
      <td>2,470</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 720 entries, 0 to 719
    Data columns (total 6 columns):
     #   Column                                           Non-Null Count  Dtype 
    ---  ------                                           --------------  ----- 
     0   Date                                             720 non-null    object
     1   Children apprehended and placed in CBP custody*  720 non-null    int64 
     2   Children in CBP custody                          720 non-null    int64 
     3   Children transferred out of CBP custody          720 non-null    int64 
     4   Children in HHS Care                             720 non-null    object
     5   Children discharged from HHS Care                720 non-null    int64 
    dtypes: int64(4), object(2)
    memory usage: 33.9+ KB
    


```python

df.dtypes
```




    Date                                               object
    Children apprehended and placed in CBP custody*     int64
    Children in CBP custody                             int64
    Children transferred out of CBP custody             int64
    Children in HHS Care                               object
    Children discharged from HHS Care                   int64
    dtype: object




```python
df["Children in HHS Care"].head(10)
```




    0    2,484
    1    2,472
    2    2,481
    3    2,468
    4    2,470
    5    2,462
    6    2,437
    7    2,439
    8    2,443
    9    2,440
    Name: Children in HHS Care, dtype: object




```python

# ==========================
# STEP 2: DATA CLEANING
# ==========================

# Make a copy of the original dataset
df = df.copy()

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Remove commas from Children in HHS Care
df["Children in HHS Care"] = (
    df["Children in HHS Care"]
    .str.replace(",", "", regex=False)
    .astype(int)
)

# Sort the dataset by date (Oldest --> Newest)
df = df.sort_values("Date")

# Reset index
df.reset_index(drop=True, inplace=True)

# Set Date as the index
df.set_index("Date", inplace=True)

print("Data cleaned successfully!")
```

    Data cleaned successfully!
    


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 720 entries, 2023-01-12 to 2025-12-21
    Data columns (total 5 columns):
     #   Column                                           Non-Null Count  Dtype
    ---  ------                                           --------------  -----
     0   Children apprehended and placed in CBP custody*  720 non-null    int64
     1   Children in CBP custody                          720 non-null    int64
     2   Children transferred out of CBP custody          720 non-null    int64
     3   Children in HHS Care                             720 non-null    int64
     4   Children discharged from HHS Care                720 non-null    int64
    dtypes: int64(5)
    memory usage: 33.8 KB
    


```python
 df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Children apprehended and placed in CBP custody*</th>
      <th>Children in CBP custody</th>
      <th>Children transferred out of CBP custody</th>
      <th>Children in HHS Care</th>
      <th>Children discharged from HHS Care</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2023-01-12</th>
      <td>33</td>
      <td>53</td>
      <td>34</td>
      <td>6566</td>
      <td>436</td>
    </tr>
    <tr>
      <th>2023-01-22</th>
      <td>32</td>
      <td>49</td>
      <td>39</td>
      <td>7122</td>
      <td>227</td>
    </tr>
    <tr>
      <th>2023-01-23</th>
      <td>32</td>
      <td>50</td>
      <td>39</td>
      <td>7280</td>
      <td>181</td>
    </tr>
    <tr>
      <th>2023-01-24</th>
      <td>47</td>
      <td>42</td>
      <td>47</td>
      <td>7433</td>
      <td>175</td>
    </tr>
    <tr>
      <th>2023-01-25</th>
      <td>20</td>
      <td>22</td>
      <td>41</td>
      <td>7538</td>
      <td>180</td>
    </tr>
  </tbody>
</table>
</div>




```python

import matplotlib.pyplot as plt

plt.figure(figsize=(15,6))

plt.plot(df.index,
         df["Children in HHS Care"],
         linewidth=2)

plt.title("Children in HHS Care Over Time")
plt.xlabel("Date")
plt.ylabel("Children")

plt.grid(True)

plt.show()
```


    
![png](output_10_0.png)
    



```python

plt.figure(figsize=(15,6))

plt.plot(df.index,
         df["Children transferred out of CBP custody"],
         color="green")

plt.title("Children Transferred Out of CBP Custody")

plt.xlabel("Date")
plt.ylabel("Children")

plt.grid(True)

plt.show()
```


    
![png](output_11_0.png)
    



```python

plt.figure(figsize=(15,6))

plt.plot(df.index,
         df["Children discharged from HHS Care"],
         color="red")

plt.title("Children Discharged from HHS Care")

plt.xlabel("Date")

plt.ylabel("Children")

plt.grid(True)

plt.show()
```


    
![png](output_12_0.png)
    



```python

plt.figure(figsize=(15,6))

plt.plot(df.index,
         df["Children apprehended and placed in CBP custody*"],
         color="orange")

plt.title("Children Apprehended and Placed in CBP Custody")

plt.xlabel("Date")

plt.ylabel("Children")

plt.grid(True)

plt.show()
```


    
![png](output_13_0.png)
    



```python

print("Number of missing dates:", len(missing_dates))
print(missing_dates[:20])
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[54], line 1
    ----> 1 print("Number of missing dates:", len(missing_dates))
          2 print(missing_dates[:20])
    

    NameError: name 'missing_dates' is not defined



```python

# ==========================================
# Check for Missing Dates
# ==========================================

import pandas as pd

# Create a complete daily date range
full_range = pd.date_range(
    start=df.index.min(),
    end=df.index.max(),
    freq='D'
)

# Find missing dates
missing_dates = full_range.difference(df.index)

print("Total missing dates:", len(missing_dates))

if len(missing_dates) > 0:
    print("\nFirst 20 missing dates:")
    print(missing_dates[:20])
else:
    print("\nNo missing dates found. The dataset has continuous daily observations.")
```


```python

# Difference between consecutive dates
date_diff = df.index.to_series().diff()

print(date_diff.value_counts().sort_index())
```


```python

date_diff = df.index.to_series().diff()
print(date_diff.value_counts().sort_index())
```


```python

# ==========================================
# 3. Exploratory Data Analysis (EDA)
# ==========================================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Plot settings
plt.style.use("ggplot")
sns.set(font_scale=1.1)

# Display all columns if needed
pd.set_option('display.max_columns', None)
```


```python

!pip install seaborn
```


```python


!conda install seaborn -y

```


```python

```


```python

# ==========================================
# 3. Exploratory Data Analysis (EDA)
# ==========================================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Graph settings
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# Figure size
plt.rcParams["figure.figsize"] = (15,6)
```


```python

print("="*70)
print("DATASET SUMMARY")
print("="*70)

print("Shape:", df.shape)

print("\nData Types")
print(df.dtypes)

print("\nSummary Statistics")
display(df.describe())
```

    ======================================================================
    DATASET SUMMARY
    ======================================================================
    Shape: (720, 5)
    
    Data Types
    Children apprehended and placed in CBP custody*    int64
    Children in CBP custody                            int64
    Children transferred out of CBP custody            int64
    Children in HHS Care                               int64
    Children discharged from HHS Care                  int64
    dtype: object
    
    Summary Statistics
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Children apprehended and placed in CBP custody*</th>
      <th>Children in CBP custody</th>
      <th>Children transferred out of CBP custody</th>
      <th>Children in HHS Care</th>
      <th>Children discharged from HHS Care</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>720.000000</td>
      <td>720.000000</td>
      <td>720.000000</td>
      <td>720.000000</td>
      <td>720.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>93.523611</td>
      <td>171.494444</td>
      <td>128.668056</td>
      <td>6061.275000</td>
      <td>173.406944</td>
    </tr>
    <tr>
      <th>std</th>
      <td>72.646625</td>
      <td>126.354965</td>
      <td>97.322012</td>
      <td>2833.070109</td>
      <td>125.702841</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>7.000000</td>
      <td>0.000000</td>
      <td>1972.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>12.000000</td>
      <td>36.000000</td>
      <td>14.000000</td>
      <td>2467.750000</td>
      <td>19.750000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>99.000000</td>
      <td>193.000000</td>
      <td>157.000000</td>
      <td>6406.500000</td>
      <td>181.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>147.250000</td>
      <td>263.250000</td>
      <td>199.250000</td>
      <td>8010.250000</td>
      <td>267.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>333.000000</td>
      <td>531.000000</td>
      <td>440.000000</td>
      <td>11516.000000</td>
      <td>505.000000</td>
    </tr>
  </tbody>
</table>
</div>



```python

print(df.isnull().sum())
```

    Children apprehended and placed in CBP custody*    0
    Children in CBP custody                            0
    Children transferred out of CBP custody            0
    Children in HHS Care                               0
    Children discharged from HHS Care                  0
    dtype: int64
    


```python
fig, ax = plt.subplots(5,1, figsize=(18,20), sharex=True)

columns = df.columns

for i,col in enumerate(columns):
    
    ax[i].plot(df.index,
               df[col],
               linewidth=2)

    ax[i].set_title(col,
                    fontsize=13)

    ax[i].grid(True)

plt.tight_layout()

plt.show()
```


    
![png](output_25_0.png)
    



```python

df.hist(
    bins=25,
    figsize=(15,10),
    edgecolor="black"
)

plt.suptitle("Distribution of Variables",
             fontsize=18)

plt.show()
```


    
![png](output_26_0.png)
    



```python

plt.figure(figsize=(16,7))

sns.boxplot(data=df)

plt.xticks(rotation=30)

plt.title("Boxplot of Numerical Variables")

plt.show()
```


    
![png](output_27_0.png)
    



```python

plt.figure(figsize=(10,8))

corr = df.corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    linewidths=0.5,
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()
```


    
![png](output_28_0.png)
    



```python

plt.figure(figsize=(16,6))

plt.plot(df.index,
         df["Children in HHS Care"],
         label="Original")

plt.plot(df.index,
         df["Children in HHS Care"].rolling(7).mean(),
         linewidth=3,
         label="7 Observation Rolling Mean")

plt.legend()

plt.title("Children in HHS Care")

plt.show()
```


    
![png](output_29_0.png)
    



```python
plt.figure(figsize=(16,6))

plt.plot(df.index,
         df["Children discharged from HHS Care"],
         label="Original")

plt.plot(df.index,
         df["Children discharged from HHS Care"].rolling(7).mean(),
         linewidth=3,
         label="7 Observation Rolling Mean")

plt.legend()

plt.title("Children Discharged from HHS Care")

plt.show()
```


    
![png](output_30_0.png)
    



```python
sns.pairplot(df)

plt.show()
```


    
![png](output_31_0.png)
    



```python
corr = df.corr()

corr["Children in HHS Care"].sort_values(ascending=False)
```




    Children in HHS Care                               1.000000
    Children discharged from HHS Care                  0.920881
    Children transferred out of CBP custody            0.713899
    Children apprehended and placed in CBP custody*    0.691312
    Children in CBP custody                            0.663662
    Name: Children in HHS Care, dtype: float64




```python

# ==========================================
# 4. Time-Series Decomposition
# ==========================================

from statsmodels.tsa.seasonal import seasonal_decompose
```


```python
import statsmodels
print(statsmodels.__version__)
```

    0.14.6
    


```python
import statsmodels
print(statsmodels.__version__)
```

    0.14.6
    


```python

# ==========================================
# 4. Time-Series Decomposition
# ==========================================

from statsmodels.tsa.seasonal import seasonal_decompose
```


```python

# Perform additive decomposition
decomposition = seasonal_decompose(
    df["Children in HHS Care"],
    model="additive",
    period=7
)
```


```python

fig = decomposition.plot()

fig.set_size_inches(16,10)

plt.show()
```


    
![png](output_38_0.png)
    



```python
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

display(trend.head())
display(seasonal.head())
display(residual.head())
```


    Date
    2023-01-12            NaN
    2023-01-22            NaN
    2023-01-23            NaN
    2023-01-24    7307.714286
    2023-01-25    7484.428571
    Name: trend, dtype: float64



    Date
    2023-01-12    11.611244
    2023-01-22    -7.156263
    2023-01-23    -3.915366
    2023-01-24    -8.709484
    2023-01-25    -4.304722
    Name: seasonal, dtype: float64



    Date
    2023-01-12           NaN
    2023-01-22           NaN
    2023-01-23           NaN
    2023-01-24    133.995198
    2023-01-25     57.876150
    Name: resid, dtype: float64



```python
from statsmodels.tsa.stattools import adfuller
```


```python




```


```python
p-value < 0.05
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[74], line 1
    ----> 1 p-value < 0.05
    

    NameError: name 'p' is not defined



```python

p-value > 0.05
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[75], line 1
    ----> 1 p-value > 0.05
    

    NameError: name 'p' is not defined



```python

from statsmodels.tsa.stattools import adfuller

# Perform ADF Test
result = adfuller(df["Children in HHS Care"])

print("ADF Statistic :", result[0])
print("p-value       :", result[1])
print("Lags Used     :", result[2])
print("Observations  :", result[3])

print("\nCritical Values:")
for key, value in result[4].items():
    print(f"{key}: {value}")
```

    ADF Statistic : -1.0271046053535111
    p-value       : 0.7432506157229029
    Lags Used     : 18
    Observations  : 701
    
    Critical Values:
    1%: -3.4397129207385357
    5%: -2.8656718422599923
    10%: -2.568970295481694
    


```python

# ==========================================
# First Order Differencing
# ==========================================

df["HHS_Difference"] = df["Children in HHS Care"].diff()

# Remove the first NaN created by differencing
df_diff = df.dropna()

print(df_diff.head())
```

                Children apprehended and placed in CBP custody*  \
    Date                                                          
    2023-01-22                                               32   
    2023-01-23                                               32   
    2023-01-24                                               47   
    2023-01-25                                               20   
    2023-01-29                                               23   
    
                Children in CBP custody  Children transferred out of CBP custody  \
    Date                                                                           
    2023-01-22                       49                                       39   
    2023-01-23                       50                                       39   
    2023-01-24                       42                                       47   
    2023-01-25                       22                                       41   
    2023-01-29                       45                                       11   
    
                Children in HHS Care  Children discharged from HHS Care  \
    Date                                                                  
    2023-01-22                  7122                                227   
    2023-01-23                  7280                                181   
    2023-01-24                  7433                                175   
    2023-01-25                  7538                                180   
    2023-01-29                  7472                                303   
    
                HHS_Difference  
    Date                        
    2023-01-22           556.0  
    2023-01-23           158.0  
    2023-01-24           153.0  
    2023-01-25           105.0  
    2023-01-29           -66.0  
    


```python
plt.figure(figsize=(15,6))

plt.plot(df_diff.index,
         df_diff["HHS_Difference"])

plt.title("First Order Differenced Series")

plt.xlabel("Date")
plt.ylabel("Difference")

plt.grid(True)

plt.show()
```


    
![png](output_46_0.png)
    



```python
result = adfuller(df_diff["HHS_Difference"])

print("ADF Statistic :", result[0])
print("p-value       :", result[1])

print("\nCritical Values:")

for key, value in result[4].items():
    print(f"{key}: {value}")
```

    ADF Statistic : -5.89330289858055
    p-value       : 2.890297178984451e-07
    
    Critical Values:
    1%: -3.439753311961436
    5%: -2.8656896390914217
    10%: -2.568979777013325
    


```python
# ==========================================
# Feature Engineering
# ==========================================

feature_df = df.copy()

feature_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Children apprehended and placed in CBP custody*</th>
      <th>Children in CBP custody</th>
      <th>Children transferred out of CBP custody</th>
      <th>Children in HHS Care</th>
      <th>Children discharged from HHS Care</th>
      <th>HHS_Difference</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2023-01-12</th>
      <td>33</td>
      <td>53</td>
      <td>34</td>
      <td>6566</td>
      <td>436</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2023-01-22</th>
      <td>32</td>
      <td>49</td>
      <td>39</td>
      <td>7122</td>
      <td>227</td>
      <td>556.0</td>
    </tr>
    <tr>
      <th>2023-01-23</th>
      <td>32</td>
      <td>50</td>
      <td>39</td>
      <td>7280</td>
      <td>181</td>
      <td>158.0</td>
    </tr>
    <tr>
      <th>2023-01-24</th>
      <td>47</td>
      <td>42</td>
      <td>47</td>
      <td>7433</td>
      <td>175</td>
      <td>153.0</td>
    </tr>
    <tr>
      <th>2023-01-25</th>
      <td>20</td>
      <td>22</td>
      <td>41</td>
      <td>7538</td>
      <td>180</td>
      <td>105.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Lag Features

feature_df["Lag_1"] = feature_df["Children in HHS Care"].shift(1)

feature_df["Lag_7"] = feature_df["Children in HHS Care"].shift(7)

feature_df["Lag_14"] = feature_df["Children in HHS Care"].shift(14)
```


```python
feature_df["Rolling_Mean_7"] = (
    feature_df["Children in HHS Care"]
    .rolling(7)
    .mean()
)

feature_df["Rolling_Mean_14"] = (
    feature_df["Children in HHS Care"]
    .rolling(14)
    .mean()
)
```


```python
feature_df["Rolling_STD_7"] = (
    feature_df["Children in HHS Care"]
    .rolling(7)
    .std()
)

feature_df["Rolling_STD_14"] = (
    feature_df["Children in HHS Care"]
    .rolling(14)
    .std()
)
```


```python
feature_df["Net_Pressure"] = (
    feature_df["Children transferred out of CBP custody"]
    -
    feature_df["Children discharged from HHS Care"]
)
```


```python
feature_df["Day_of_Week"] = feature_df.index.dayofweek

feature_df["Month"] = feature_df.index.month

feature_df["Quarter"] = feature_df.index.quarter

feature_df["Year"] = feature_df.index.year
```


```python

feature_df.isnull().sum()
```




    Children apprehended and placed in CBP custody*     0
    Children in CBP custody                             0
    Children transferred out of CBP custody             0
    Children in HHS Care                                0
    Children discharged from HHS Care                   0
    HHS_Difference                                      1
    Lag_1                                               1
    Lag_7                                               7
    Lag_14                                             14
    Rolling_Mean_7                                      6
    Rolling_Mean_14                                    13
    Rolling_STD_7                                       6
    Rolling_STD_14                                     13
    Net_Pressure                                        0
    Day_of_Week                                         0
    Month                                               0
    Quarter                                             0
    Year                                                0
    dtype: int64




```python
feature_df = feature_df.dropna()

feature_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Children apprehended and placed in CBP custody*</th>
      <th>Children in CBP custody</th>
      <th>Children transferred out of CBP custody</th>
      <th>Children in HHS Care</th>
      <th>Children discharged from HHS Care</th>
      <th>HHS_Difference</th>
      <th>Lag_1</th>
      <th>Lag_7</th>
      <th>Lag_14</th>
      <th>Rolling_Mean_7</th>
      <th>Rolling_Mean_14</th>
      <th>Rolling_STD_7</th>
      <th>Rolling_STD_14</th>
      <th>Net_Pressure</th>
      <th>Day_of_Week</th>
      <th>Month</th>
      <th>Quarter</th>
      <th>Year</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2023-02-09</th>
      <td>124</td>
      <td>234</td>
      <td>161</td>
      <td>7908</td>
      <td>353</td>
      <td>-7.0</td>
      <td>7915.0</td>
      <td>7803.0</td>
      <td>6566.0</td>
      <td>7823.714286</td>
      <td>7654.071429</td>
      <td>124.514677</td>
      <td>254.662209</td>
      <td>-192</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2023</td>
    </tr>
    <tr>
      <th>2023-02-12</th>
      <td>92</td>
      <td>203</td>
      <td>173</td>
      <td>7434</td>
      <td>317</td>
      <td>-474.0</td>
      <td>7908.0</td>
      <td>7903.0</td>
      <td>7122.0</td>
      <td>7756.714286</td>
      <td>7676.357143</td>
      <td>185.827621</td>
      <td>215.096618</td>
      <td>-144</td>
      <td>6</td>
      <td>2</td>
      <td>1</td>
      <td>2023</td>
    </tr>
    <tr>
      <th>2023-02-13</th>
      <td>186</td>
      <td>259</td>
      <td>172</td>
      <td>7483</td>
      <td>244</td>
      <td>49.0</td>
      <td>7434.0</td>
      <td>7879.0</td>
      <td>7280.0</td>
      <td>7700.142857</td>
      <td>7690.857143</td>
      <td>201.971474</td>
      <td>191.915389</td>
      <td>-72</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2023</td>
    </tr>
    <tr>
      <th>2023-02-14</th>
      <td>154</td>
      <td>225</td>
      <td>220</td>
      <td>7794</td>
      <td>223</td>
      <td>311.0</td>
      <td>7483.0</td>
      <td>7586.0</td>
      <td>7433.0</td>
      <td>7729.857143</td>
      <td>7716.642857</td>
      <td>197.633861</td>
      <td>178.379288</td>
      <td>-3</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2023</td>
    </tr>
    <tr>
      <th>2023-02-15</th>
      <td>91</td>
      <td>199</td>
      <td>172</td>
      <td>7869</td>
      <td>290</td>
      <td>75.0</td>
      <td>7794.0</td>
      <td>7720.0</td>
      <td>7538.0</td>
      <td>7751.142857</td>
      <td>7740.285714</td>
      <td>204.306492</td>
      <td>174.779610</td>
      <td>-118</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2023</td>
    </tr>
  </tbody>
</table>
</div>




```python
feature_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 706 entries, 2023-02-09 to 2025-12-21
    Data columns (total 18 columns):
     #   Column                                           Non-Null Count  Dtype  
    ---  ------                                           --------------  -----  
     0   Children apprehended and placed in CBP custody*  706 non-null    int64  
     1   Children in CBP custody                          706 non-null    int64  
     2   Children transferred out of CBP custody          706 non-null    int64  
     3   Children in HHS Care                             706 non-null    int64  
     4   Children discharged from HHS Care                706 non-null    int64  
     5   HHS_Difference                                   706 non-null    float64
     6   Lag_1                                            706 non-null    float64
     7   Lag_7                                            706 non-null    float64
     8   Lag_14                                           706 non-null    float64
     9   Rolling_Mean_7                                   706 non-null    float64
     10  Rolling_Mean_14                                  706 non-null    float64
     11  Rolling_STD_7                                    706 non-null    float64
     12  Rolling_STD_14                                   706 non-null    float64
     13  Net_Pressure                                     706 non-null    int64  
     14  Day_of_Week                                      706 non-null    int32  
     15  Month                                            706 non-null    int32  
     16  Quarter                                          706 non-null    int32  
     17  Year                                             706 non-null    int32  
    dtypes: float64(8), int32(4), int64(6)
    memory usage: 93.8 KB
    


```python
# ==========================================
# Train-Test Split
# ==========================================

target = "Children in HHS Care"

X = feature_df.drop(columns=[target])

y = feature_df[target]

print("Feature Shape :", X.shape)
print("Target Shape :", y.shape)
```

    Feature Shape : (706, 17)
    Target Shape : (706,)
    


```python
split_index = int(len(feature_df)*0.8)

X_train = X.iloc[:split_index]

X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]

y_test = y.iloc[split_index:]
```


```python
print("Training Samples :", len(X_train))

print("Testing Samples :", len(X_test))
```

    Training Samples : 564
    Testing Samples : 142
    


```python
# ==========================================
# Naïve Forecast
# ==========================================

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Predict each test value using the previous actual value
naive_predictions = y_test.shift(1)

# Remove the first NaN
naive_predictions = naive_predictions.dropna()
naive_actual = y_test.iloc[1:]

# Evaluation
naive_mae = mean_absolute_error(naive_actual, naive_predictions)
naive_rmse = np.sqrt(mean_squared_error(naive_actual, naive_predictions))
naive_mape = np.mean(np.abs((naive_actual - naive_predictions) / naive_actual)) * 100

print("Naïve Forecast Results")
print("-" * 40)
print(f"MAE  : {naive_mae:.2f}")
print(f"RMSE : {naive_rmse:.2f}")
print(f"MAPE : {naive_mape:.2f}%")
```

    Naïve Forecast Results
    ----------------------------------------
    MAE  : 10.30
    RMSE : 14.08
    MAPE : 0.46%
    


```python
!pip install scikit-learn
```

    Requirement already satisfied: scikit-learn in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (1.9.0)
    Requirement already satisfied: numpy>=1.24.1 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from scikit-learn) (2.3.4)
    Requirement already satisfied: scipy>=1.10.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from scikit-learn) (1.18.0)
    Requirement already satisfied: joblib>=1.4.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from scikit-learn) (1.5.3)
    Requirement already satisfied: narwhals>=2.0.1 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from scikit-learn) (2.23.0)
    Requirement already satisfied: threadpoolctl>=3.5.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from scikit-learn) (3.6.0)
    

    
    [notice] A new release of pip is available: 25.1.1 -> 26.1.2
    [notice] To update, run: python.exe -m pip install --upgrade pip
    


```python
import sklearn
print(sklearn.__version__)
```

    1.9.0
    


```python
!pip install scikit-learn statsmodels xgboost streamlit plotly openpyxl
```

    Requirement already satisfied: scikit-learn in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (1.9.0)
    Requirement already satisfied: statsmodels in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (0.14.6)
    Requirement already satisfied: xgboost in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (3.3.0)
    Requirement already satisfied: streamlit in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (1.59.1)
    Requirement already satisfied: plotly in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (6.8.0)
    Requirement already satisfied: openpyxl in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (3.1.5)
    Requirement already satisfied: numpy>=1.24.1 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from scikit-learn) (2.3.4)
    Requirement already satisfied: scipy>=1.10.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from scikit-learn) (1.18.0)
    Requirement already satisfied: joblib>=1.4.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from scikit-learn) (1.5.3)
    Requirement already satisfied: narwhals>=2.0.1 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from scikit-learn) (2.23.0)
    Requirement already satisfied: threadpoolctl>=3.5.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from scikit-learn) (3.6.0)
    Requirement already satisfied: pandas!=2.1.0,>=1.4 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from statsmodels) (2.3.3)
    Requirement already satisfied: patsy>=0.5.6 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from statsmodels) (1.0.2)
    Requirement already satisfied: packaging>=21.3 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from statsmodels) (25.0)
    Requirement already satisfied: altair!=5.4.0,!=5.4.1,<7,>=4.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (6.2.2)
    Requirement already satisfied: blinker<2,>=1.5.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (1.9.0)
    Requirement already satisfied: cachetools<8,>=5.5 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (7.1.4)
    Requirement already satisfied: click<9,>=7.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (8.4.2)
    Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (3.1.50)
    Requirement already satisfied: pillow<13,>=7.1.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (12.0.0)
    Requirement already satisfied: pydeck<1,>=0.8.0b4 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (0.9.3)
    Requirement already satisfied: protobuf<8,>=3.20 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (7.35.1)
    Requirement already satisfied: pyarrow>=7.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (24.0.0)
    Requirement already satisfied: requests<3,>=2.27 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (2.32.3)
    Requirement already satisfied: tenacity<10,>=8.1.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (9.1.4)
    Requirement already satisfied: toml<2,>=0.10.1 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (0.10.2)
    Requirement already satisfied: typing-extensions<5,>=4.10.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (4.13.2)
    Requirement already satisfied: starlette>=0.40.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (1.3.1)
    Requirement already satisfied: uvicorn>=0.30.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (0.51.0)
    Requirement already satisfied: httptools>=0.6.3 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (0.8.0)
    Requirement already satisfied: anyio>=4.0.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (4.9.0)
    Requirement already satisfied: python-multipart>=0.0.10 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (0.0.32)
    Requirement already satisfied: websockets>=12.0.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (16.0)
    Requirement already satisfied: itsdangerous>=2.1.2 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (2.2.0)
    Requirement already satisfied: watchdog<7,>=2.1.5 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (6.0.0)
    Requirement already satisfied: jinja2 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (3.1.6)
    Requirement already satisfied: jsonschema>=3.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (4.24.0)
    Requirement already satisfied: colorama in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from click<9,>=7.0->streamlit) (0.4.6)
    Requirement already satisfied: gitdb<5,>=4.0.1 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.12)
    Requirement already satisfied: smmap<6,>=3.0.1 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.3)
    Requirement already satisfied: python-dateutil>=2.8.2 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from pandas!=2.1.0,>=1.4->statsmodels) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from pandas!=2.1.0,>=1.4->statsmodels) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from pandas!=2.1.0,>=1.4->statsmodels) (2025.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from requests<3,>=2.27->streamlit) (3.4.2)
    Requirement already satisfied: idna<4,>=2.5 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from requests<3,>=2.27->streamlit) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from requests<3,>=2.27->streamlit) (2.4.0)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from requests<3,>=2.27->streamlit) (2025.4.26)
    Requirement already satisfied: et-xmlfile in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from openpyxl) (2.0.0)
    Requirement already satisfied: sniffio>=1.1 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from anyio>=4.0.0->streamlit) (1.3.1)
    Requirement already satisfied: MarkupSafe>=2.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from jinja2->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (3.0.2)
    Requirement already satisfied: attrs>=22.2.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (25.3.0)
    Requirement already satisfied: jsonschema-specifications>=2023.03.6 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (2025.4.1)
    Requirement already satisfied: referencing>=0.28.4 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (0.36.2)
    Requirement already satisfied: rpds-py>=0.7.1 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (0.25.1)
    Requirement already satisfied: six>=1.5 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from python-dateutil>=2.8.2->pandas!=2.1.0,>=1.4->statsmodels) (1.17.0)
    Requirement already satisfied: h11>=0.8 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from uvicorn>=0.30.0->streamlit) (0.16.0)
    

    
    [notice] A new release of pip is available: 25.1.1 -> 26.1.2
    [notice] To update, run: python.exe -m pip install --upgrade pip
    


```python
import sklearn
print(sklearn.__version__)
```

    1.9.0
    


```python
X = feature_df.drop(columns=[target])
```


```python
# ==========================================
# Prepare Features and Target
# ==========================================

features = [
    "Children apprehended and placed in CBP custody*",
    "Children in CBP custody",
    "Children transferred out of CBP custody",
    "Children discharged from HHS Care",
    "Lag_1",
    "Lag_7",
    "Lag_14",
    "Rolling_Mean_7",
    "Rolling_Mean_14",
    "Rolling_STD_7",
    "Rolling_STD_14",
    "Net_Pressure",
    "Day_of_Week",
    "Month",
    "Quarter",
    "Year"
]

target = "Children in HHS Care"

X = feature_df[features]
y = feature_df[target]

print("Feature Shape:", X.shape)
print("Target Shape :", y.shape)
```

    Feature Shape: (706, 16)
    Target Shape : (706,)
    


```python
split_index = int(len(feature_df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training Samples:", len(X_train))
print("Testing Samples :", len(X_test))
```

    Training Samples: 564
    Testing Samples : 142
    


```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

naive_predictions = y_test.shift(1)

naive_predictions = naive_predictions.dropna()
naive_actual = y_test.iloc[1:]

naive_mae = mean_absolute_error(naive_actual, naive_predictions)
naive_rmse = np.sqrt(mean_squared_error(naive_actual, naive_predictions))
naive_mape = np.mean(np.abs((naive_actual - naive_predictions) / naive_actual)) * 100

print("Naïve Forecast")
print(f"MAE  : {naive_mae:.2f}")
print(f"RMSE : {naive_rmse:.2f}")
print(f"MAPE : {naive_mape:.2f}%")
```

    Naïve Forecast
    MAE  : 10.30
    RMSE : 14.08
    MAPE : 0.46%
    


```python
moving_predictions = []

history = list(y_train)

for t in range(len(y_test)):
    prediction = np.mean(history[-7:])
    moving_predictions.append(prediction)
    history.append(y_test.iloc[t])

moving_mae = mean_absolute_error(y_test, moving_predictions)
moving_rmse = np.sqrt(mean_squared_error(y_test, moving_predictions))
moving_mape = np.mean(np.abs((y_test - moving_predictions) / y_test)) * 100

print("Moving Average Forecast")
print(f"MAE  : {moving_mae:.2f}")
print(f"RMSE : {moving_rmse:.2f}")
print(f"MAPE : {moving_mape:.2f}%")
```

    Moving Average Forecast
    MAE  : 33.23
    RMSE : 40.68
    MAPE : 1.48%
    


```python

```


```python

from statsmodels.tsa.holtwinters import ExponentialSmoothing
```


```python
# ==========================================
# Exponential Smoothing Model
# ==========================================

exp_model = ExponentialSmoothing(
    y_train,
    trend="add",
    seasonal=None
)

exp_fit = exp_model.fit()
```

    C:\Users\pccli\AppData\Local\Programs\Python\Python313\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    


```python

exp_predictions = exp_fit.forecast(len(y_test))
```

    C:\Users\pccli\AppData\Local\Programs\Python\Python313\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
      return get_prediction_index(
    C:\Users\pccli\AppData\Local\Programs\Python\Python313\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: FutureWarning: No supported index is available. In the next version, calling this method in a model without a supported index will result in an exception.
      return get_prediction_index(
    


```python

from statsmodels.tsa.arima.model import ARIMA
```


```python

arima_model = ARIMA(
    y_train,
    order=(1,1,1)
)

arima_fit = arima_model.fit()
```

    C:\Users\pccli\AppData\Local\Programs\Python\Python313\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    C:\Users\pccli\AppData\Local\Programs\Python\Python313\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    C:\Users\pccli\AppData\Local\Programs\Python\Python313\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    


```python
arima_predictions = arima_fit.forecast(steps=len(y_test))
```

    C:\Users\pccli\AppData\Local\Programs\Python\Python313\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
      return get_prediction_index(
    C:\Users\pccli\AppData\Local\Programs\Python\Python313\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: FutureWarning: No supported index is available. In the next version, calling this method in a model without a supported index will result in an exception.
      return get_prediction_index(
    


```python

arima_predictions = arima_fit.forecast(steps=len(y_test))
```

    C:\Users\pccli\AppData\Local\Programs\Python\Python313\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
      return get_prediction_index(
    


```python

# ==========================================
# Exponential Smoothing Model
# ==========================================

exp_model = ExponentialSmoothing(
    y_train,
    trend="add",
    seasonal=None
)

exp_fit = exp_model.fit()
```

    C:\Users\pccli\AppData\Local\Programs\Python\Python313\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    


```python

exp_predictions = exp_fit.forecast(len(y_test))
```

    C:\Users\pccli\AppData\Local\Programs\Python\Python313\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
      return get_prediction_index(
    C:\Users\pccli\AppData\Local\Programs\Python\Python313\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: FutureWarning: No supported index is available. In the next version, calling this method in a model without a supported index will result in an exception.
      return get_prediction_index(
    


```python


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

exp_mae = mean_absolute_error(y_test, exp_predictions)
exp_rmse = np.sqrt(mean_squared_error(y_test, exp_predictions))
exp_mape = np.mean(np.abs((y_test - exp_predictions) / y_test)) * 100

print("Exponential Smoothing")
print("----------------------------")
print(f"MAE  : {exp_mae:.2f}")
print(f"RMSE : {exp_rmse:.2f}")
print(f"MAPE : {exp_mape:.2f}%")
```

    Exponential Smoothing
    ----------------------------
    MAE  : 732.50
    RMSE : 813.98
    MAPE : nan%
    

    C:\Users\pccli\AppData\Local\Temp\ipykernel_9568\3262653604.py:6: RuntimeWarning: '<' not supported between instances of 'int' and 'Timestamp', sort order is undefined for incomparable objects.
      exp_mape = np.mean(np.abs((y_test - exp_predictions) / y_test)) * 100
    


```python

plt.figure(figsize=(15,6))

plt.plot(y_test.index, y_test, label="Actual")
plt.plot(y_test.index, exp_predictions, label="Exponential Smoothing")

plt.title("Exponential Smoothing Forecast")
plt.legend()
plt.grid(True)

plt.show()
```


    
![png](output_81_0.png)
    



```python
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Fit ARIMA model
model = ARIMA(y_train, order=(1,1,1))
model_fit = model.fit()

# Forecast
arima_predictions = model_fit.forecast(steps=len(y_test))

# Metrics
arima_mae = mean_absolute_error(y_test, arima_predictions)
arima_rmse = np.sqrt(mean_squared_error(y_test, arima_predictions))
arima_mape = np.mean(np.abs((y_test - arima_predictions) / y_test)) * 100

print("ARIMA Forecast")
print(f"MAE  : {arima_mae:.2f}")
print(f"RMSE : {arima_rmse:.2f}")
print(f"MAPE : {arima_mape:.2f}%")
```

    C:\Users\pccli\AppData\Local\Programs\Python\Python313\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    C:\Users\pccli\AppData\Local\Programs\Python\Python313\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    C:\Users\pccli\AppData\Local\Programs\Python\Python313\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    

    ARIMA Forecast
    MAE  : 197.56
    RMSE : 241.43
    MAPE : nan%
    

    C:\Users\pccli\AppData\Local\Programs\Python\Python313\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
      return get_prediction_index(
    C:\Users\pccli\AppData\Local\Programs\Python\Python313\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: FutureWarning: No supported index is available. In the next version, calling this method in a model without a supported index will result in an exception.
      return get_prediction_index(
    C:\Users\pccli\AppData\Local\Temp\ipykernel_9568\2065594013.py:15: RuntimeWarning: '<' not supported between instances of 'int' and 'Timestamp', sort order is undefined for incomparable objects.
      arima_mape = np.mean(np.abs((y_test - arima_predictions) / y_test)) * 100
    


```python
print(arima_mae, arima_rmse, arima_mape)
```

    197.55578308612536 241.43059648592765 nan
    


```python
arima_predictions = arima_fit.forecast(steps=len(y_test))
```

    C:\Users\pccli\AppData\Local\Programs\Python\Python313\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
      return get_prediction_index(
    C:\Users\pccli\AppData\Local\Programs\Python\Python313\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: FutureWarning: No supported index is available. In the next version, calling this method in a model without a supported index will result in an exception.
      return get_prediction_index(
    


```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

arima_mae = mean_absolute_error(y_test, arima_predictions)
arima_rmse = np.sqrt(mean_squared_error(y_test, arima_predictions))
arima_mape = np.mean(np.abs((y_test - arima_predictions) / y_test)) * 100
```

    C:\Users\pccli\AppData\Local\Temp\ipykernel_9568\3151370831.py:6: RuntimeWarning: '<' not supported between instances of 'int' and 'Timestamp', sort order is undefined for incomparable objects.
      arima_mape = np.mean(np.abs((y_test - arima_predictions) / y_test)) * 100
    


```python
print("ARIMA Forecast")
print(f"MAE  : {arima_mae:.2f}")
print(f"RMSE : {arima_rmse:.2f}")
print(f"MAPE : {arima_mape:.2f}%")
```

    ARIMA Forecast
    MAE  : 197.56
    RMSE : 241.43
    MAPE : nan%
    


```python
print("Naïve Forecast")
print(f"MAE  : {naive_mae:.2f}")
print(f"RMSE : {naive_rmse:.2f}")
print(f"MAPE : {naive_mape:.2f}%")

print("\nMoving Average Forecast")
print(f"MAE  : {moving_mae:.2f}")
print(f"RMSE : {moving_rmse:.2f}")
print(f"MAPE : {moving_mape:.2f}%")

print("\nARIMA Forecast")
print(f"MAE  : {arima_mae:.2f}")
print(f"RMSE : {arima_rmse:.2f}")
print(f"MAPE : {arima_mape:.2f}%")
```

    Naïve Forecast
    MAE  : 10.30
    RMSE : 14.08
    MAPE : 0.46%
    
    Moving Average Forecast
    MAE  : 33.23
    RMSE : 40.68
    MAPE : 1.48%
    
    ARIMA Forecast
    MAE  : 197.56
    RMSE : 241.43
    MAPE : nan%
    


```python
print(exp_mae, exp_rmse, exp_mape)
```

    732.4989439378518 813.9784929797727 nan
    


```python

print(arima_mae, arima_rmse, arima_mape)
```

    197.55578308612536 241.43059648592765 nan
    


```python
from sklearn.ensemble import RandomForestRegressor
```


```python
# ==========================================
# Random Forest Model
# ==========================================

rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)
```




<style>.sk-global {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;
}

.sk-global.light {
  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: black;
  --sklearn-color-background: white;
  --sklearn-color-border-box: black;
  --sklearn-color-icon: #696969;
}

.sk-global.dark {
  --sklearn-color-text-on-default-background: white;
  --sklearn-color-background: #111;
  --sklearn-color-border-box: white;
  --sklearn-color-icon: #878787;
}

.sk-global {
  color: var(--sklearn-color-text);
}

.sk-global pre {
  padding: 0;
}

.sk-global input.sk-hidden--visually {
  border: 0;
  clip-path: inset(100%);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

.sk-global div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

.sk-global div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

.sk-global div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

.sk-global div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

.sk-global div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

.sk-global div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

.sk-global div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

.sk-global div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

.sk-global div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

.sk-global div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

.sk-global div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
.sk-global label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: center;
  justify-content: center;
  gap: 0.5em;
}

.sk-global label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

.sk-global label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

.sk-global label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

.sk-global div.sk-toggleable__content {
  display: none;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

.sk-global div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

.sk-global div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

.sk-global div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

.sk-global input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  display: block;
  width: 100%;
  overflow: visible;
}

.sk-global input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

.sk-global div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

.sk-global div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
.sk-global div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

.sk-global div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

.sk-global div.sk-label label.sk-toggleable__label,
.sk-global div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
.sk-global div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
.sk-global div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

.sk-global div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  line-height: 1.2em;
}

.sk-global div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
.sk-global div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

.sk-global div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
.sk-global div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

.sk-global div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-unfitted-level-0);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-3) 1pt solid;
  color: var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3) 1pt solid;
  color: var(--sklearn-color-fitted-level-3);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  border: var(--sklearn-color-fitted-level-0) 1pt solid;
  color: var(--sklearn-color-unfitted-level-0);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  border: var(--sklearn-color-fitted-level-0) 1pt solid;
  color: var(--sklearn-color-fitted-level-0);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

.sk-global a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-unfitted-level-0);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

.sk-global a.estimator_doc_link.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
.sk-global a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

.sk-global a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}

.sk-top-container.sk-global {
  /* pydata-sphinx-theme hides overflow, so scrolling is disabled.
   We need to set it to !important and add tabindex="0" in the HTML
   to allow keyboard-only users to navigate the display. */
  overflow-x: scroll !important;
  max-width: 100%;
}

.estimator-table {
    font-family: monospace;
}

.estimator-table summary {
    padding: .5rem;
    cursor: pointer;
}

.estimator-table summary::marker {
    font-size: 0.7rem;
}

.estimator-table details[open] {
    padding-left: 0.1rem;
    padding-right: 0.1rem;
    padding-bottom: 0.3rem;
}

.estimator-table .parameters-table {
    margin-left: auto !important;
    margin-right: auto !important;
    margin-top: 0;
}

.estimator-table .parameters-table tr:nth-child(odd) {
    background-color: #fff;
}

.estimator-table .parameters-table tr:nth-child(even) {
    background-color: #f6f6f6;
}

.estimator-table .parameters-table tr:hover td {
    background-color: #e0e0e0;
}

.estimator-table table :is(td, th) {
    border: 1px solid rgba(106, 105, 104, 0.232);
}

/*
    `table td`is set in notebook with right text-align.
    We need to overwrite it.
*/
.estimator-table table td.param {
    text-align: left;
    position: relative;
    padding: 0;
}

.user-set td {
    color:rgb(255, 94, 0);
    text-align: left !important;
}

.user-set td.value {
    color:rgb(255, 94, 0);
    background-color: transparent;
}

.default td, .estimator-table th {
    color: black;
    text-align: left !important;
}

.user-set td i,
.default td i {
    color: black;
}

td.fitted-att-type {
    white-space: preserve nowrap;
}

/*
    Styles for parameter documentation links
    We need styling for visited so jupyter doesn't overwrite it
*/
a.param-doc-link,
a.param-doc-link:link,
a.param-doc-link:visited {
    text-decoration: underline dashed;
    text-underline-offset: .3em;
    color: inherit;
    display: block;
    padding: .5em;
}

@supports(anchor-name: --doc-link) {
    a.param-doc-link,
    a.param-doc-link:link,
    a.param-doc-link:visited {
    anchor-name: --doc-link;
    }
}

/* "hack" to make the entire area of the cell containing the link clickable */
a.param-doc-link::before {
    position: absolute;
    content: "";
    inset: 0;
}

.param-doc-description {
    display: none;
    position: absolute;
    z-index: 9999;
    left: 0;
    padding: .5ex;
    margin-left: 1.5em;
    color: var(--sklearn-color-text);
    box-shadow: .3em .3em .4em #999;
    width: max-content;
    text-align: left;
    max-height: 10em;
    overflow-y: auto;

    /* unfitted */
    background: var(--sklearn-color-unfitted-level-0);
    border: thin solid var(--sklearn-color-unfitted-level-3);
}

@supports(position-area: center right) {
    .param-doc-description {
    position-area: center right;
    position: fixed;
    margin-left: 0;
    }
}

/* Fitted state for parameter tooltips */
.fitted .param-doc-description {
    /* fitted */
    background: var(--sklearn-color-fitted-level-0);
    border: thin solid var(--sklearn-color-fitted-level-3);
}

.param-doc-link:hover .param-doc-description {
    display: block;
}

.copy-paste-icon {
    background-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NDggNTEyIj48IS0tIUZvbnQgQXdlc29tZSBGcmVlIDYuNy4yIGJ5IEBmb250YXdlc29tZSAtIGh0dHBzOi8vZm9udGF3ZXNvbWUuY29tIExpY2Vuc2UgLSBodHRwczovL2ZvbnRhd2Vzb21lLmNvbS9saWNlbnNlL2ZyZWUgQ29weXJpZ2h0IDIwMjUgRm9udGljb25zLCBJbmMuLS0+PHBhdGggZD0iTTIwOCAwTDMzMi4xIDBjMTIuNyAwIDI0LjkgNS4xIDMzLjkgMTQuMWw2Ny45IDY3LjljOSA5IDE0LjEgMjEuMiAxNC4xIDMzLjlMNDQ4IDMzNmMwIDI2LjUtMjEuNSA0OC00OCA0OGwtMTkyIDBjLTI2LjUgMC00OC0yMS41LTQ4LTQ4bDAtMjg4YzAtMjYuNSAyMS41LTQ4IDQ4LTQ4ek00OCAxMjhsODAgMCAwIDY0LTY0IDAgMCAyNTYgMTkyIDAgMC0zMiA2NCAwIDAgNDhjMCAyNi41LTIxLjUgNDgtNDggNDhMNDggNTEyYy0yNi41IDAtNDgtMjEuNS00OC00OEwwIDE3NmMwLTI2LjUgMjEuNS00OCA0OC00OHoiLz48L3N2Zz4=);
    background-repeat: no-repeat;
    background-size: 14px 14px;
    background-position: 0;
    display: inline-block;
    width: 14px;
    height: 14px;
    cursor: pointer;
}

.features {
  font-family: monospace;
  cursor: pointer;
  background-color: var(--sklearn-color-unfitted-level-0);
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: .20em;
  margin-bottom: 0.5em;
  font-size: inherit; /* Needed for jupyter */
}

.features.fitted {
  background-color: var(--sklearn-color-fitted-level-0);
}

.features summary {
  cursor: pointer;
  display: flex;
  margin-bottom: 0;
  text-align: center;
  align-items: center;
  justify-content: center;
  gap: 0.5em;
  padding: .25em;
}

.features details[open] > summary {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
  border-radius: .20em 0 0 0;
}

.features.fitted details[open] > summary {
  background-color: var(--sklearn-color-fitted-level-2);
  border-radius: .20em 0 0 0;
}

.features details > summary .arrow::before {
  content: "▸";
  color: grey;
}

.features details[open] > summary .arrow::before {
  content: "▾";
}

.features details:hover > summary {
  margin: 0;
  background-color: var(--sklearn-color-unfitted-level-2);
}

.features.fitted details:hover > summary {
  margin: 0;
  background-color: var(--sklearn-color-fitted-level-2);
}

.features .features-container {
  max-width: 15em;
  max-height: 10em;
  overflow: auto;
  scrollbar-width: thin;
  padding: .25em 0.1rem;
  background-color: var(--sklearn-color-unfitted-level-0);
  border-radius: 0 0 .5em .5em;
}

.features.fitted .features-container {
  background-color: var(--sklearn-color-fitted-level-0);
}

.features .image-container {
  block-size: 1em;
  inline-size: 1em;
  padding: 0;
  margin: 0%;
  display: flex;
  justify-content: center;
  align-items: center;
}

.features .copy-paste-icon {
  background-size: 1em 1em;
  width: 1em;
  height: 1em;
  filter: grayscale(100%) opacity(60%);
}

.features .features-container table {
  width: 100%;
  margin: 0.01em;
}

.features .features-container table tr:nth-child(odd) {
  background-color: #fff;
}

.features .features-container table tr:nth-child(even) {
  background-color: #f6f6f6;
}

.features .features-container table tr:hover {
  background-color: #e0e0e0;
}

.features .features-container table {
  table-layout: inherit;
}

.features .features-container table td {
  text-align: left;
  padding: 0 0.5em;
  border: 1px solid rgba(106, 105, 104, 0.232);
  white-space: nowrap;
  color: var(--sklearn-color-text);
}

.total_features {
  display: flex;
  justify-content: center;
  margin-top: 0.5em;
}
</style><body><div id="sk-container-id-1" tabindex="0" class="sk-top-container sk-global"><div class="sk-text-repr-fallback"><pre>RandomForestRegressor(n_estimators=200, random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually sk-global" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>RandomForestRegressor</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html">?<span>Documentation for RandomForestRegressor</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted" data-param-prefix="">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('n_estimators',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-n_estimators;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=n_estimators,-int%2C%20default%3D100">
            n_estimators
            <span class="param-doc-description"
            style="position-anchor: --doc-link-n_estimators;">
            n_estimators: int, default=100<br><br>The number of trees in the forest.<br><br>.. versionchanged:: 0.22<br>   The default value of ``n_estimators`` changed from 10 to 100<br>   in 0.22.</span>
        </a>
    </td>
            <td class="value">200</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('random_state',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-random_state;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=random_state,-int%2C%20RandomState%20instance%20or%20None%2C%20default%3DNone">
            random_state
            <span class="param-doc-description"
            style="position-anchor: --doc-link-random_state;">
            random_state: int, RandomState instance or None, default=None<br><br>Controls both the randomness of the bootstrapping of the samples used<br>when building trees (if ``bootstrap=True``) and the sampling of the<br>features to consider when looking for the best split at each node<br>(if ``max_features &lt; n_features``).<br>See :term:`Glossary &lt;random_state&gt;` for details.</span>
        </a>
    </td>
            <td class="value">42</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('criterion',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-criterion;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=criterion,-%7B%22squared_error%22%2C%20%22absolute_error%22%2C%20%22poisson%22%7D%2C%20default%3D%22squared_error%22">
            criterion
            <span class="param-doc-description"
            style="position-anchor: --doc-link-criterion;">
            criterion: {&quot;squared_error&quot;, &quot;absolute_error&quot;, &quot;poisson&quot;}, default=&quot;squared_error&quot;<br><br>The function to measure the quality of a split. Supported criteria<br>are &quot;squared_error&quot; for the mean squared error, which is equal to<br>variance reduction as feature selection criterion and minimizes the L2<br>loss using the mean of each terminal node, &quot;absolute_error&quot; for the mean<br>absolute error, which minimizes the L1 loss using the median of each terminal<br>node, and &quot;poisson&quot; which uses reduction in Poisson deviance to find splits,<br>also using the mean of each terminal node.<br><br>.. versionadded:: 0.18<br>   Mean Absolute Error (MAE) criterion.<br><br>.. versionadded:: 1.0<br>   Poisson criterion.<br><br>.. versionchanged:: 1.9<br>    Criterion `&quot;friedman_mse&quot;` was deprecated.</span>
        </a>
    </td>
            <td class="value">&#x27;squared_error&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_depth',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-max_depth;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=max_depth,-int%2C%20default%3DNone">
            max_depth
            <span class="param-doc-description"
            style="position-anchor: --doc-link-max_depth;">
            max_depth: int, default=None<br><br>The maximum depth of the tree. If None, then nodes are expanded until<br>all leaves are pure or until all leaves contain less than<br>min_samples_split samples.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_samples_split',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-min_samples_split;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=min_samples_split,-int%20or%20float%2C%20default%3D2">
            min_samples_split
            <span class="param-doc-description"
            style="position-anchor: --doc-link-min_samples_split;">
            min_samples_split: int or float, default=2<br><br>The minimum number of samples required to split an internal node:<br><br>- If int, then consider `min_samples_split` as the minimum number.<br>- If float, then `min_samples_split` is a fraction and<br>  `ceil(min_samples_split * n_samples)` are the minimum<br>  number of samples for each split.<br><br>.. versionchanged:: 0.18<br>   Added float values for fractions.</span>
        </a>
    </td>
            <td class="value">2</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_samples_leaf',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-min_samples_leaf;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=min_samples_leaf,-int%20or%20float%2C%20default%3D1">
            min_samples_leaf
            <span class="param-doc-description"
            style="position-anchor: --doc-link-min_samples_leaf;">
            min_samples_leaf: int or float, default=1<br><br>The minimum number of samples required to be at a leaf node.<br>A split point at any depth will only be considered if it leaves at<br>least ``min_samples_leaf`` training samples in each of the left and<br>right branches.  This may have the effect of smoothing the model,<br>especially in regression.<br><br>- If int, then consider `min_samples_leaf` as the minimum number.<br>- If float, then `min_samples_leaf` is a fraction and<br>  `ceil(min_samples_leaf * n_samples)` are the minimum<br>  number of samples for each node.<br><br>.. versionchanged:: 0.18<br>   Added float values for fractions.</span>
        </a>
    </td>
            <td class="value">1</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_weight_fraction_leaf',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-min_weight_fraction_leaf;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=min_weight_fraction_leaf,-float%2C%20default%3D0.0">
            min_weight_fraction_leaf
            <span class="param-doc-description"
            style="position-anchor: --doc-link-min_weight_fraction_leaf;">
            min_weight_fraction_leaf: float, default=0.0<br><br>The minimum weighted fraction of the sum total of weights (of all<br>the input samples) required to be at a leaf node. Samples have<br>equal weight when sample_weight is not provided.</span>
        </a>
    </td>
            <td class="value">0.0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_features',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-max_features;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=max_features,-%7B%22sqrt%22%2C%20%22log2%22%2C%20None%7D%2C%20int%20or%20float%2C%20default%3D1.0">
            max_features
            <span class="param-doc-description"
            style="position-anchor: --doc-link-max_features;">
            max_features: {&quot;sqrt&quot;, &quot;log2&quot;, None}, int or float, default=1.0<br><br>The number of features to consider when looking for the best split:<br><br>- If int, then consider `max_features` features at each split.<br>- If float, then `max_features` is a fraction and<br>  `max(1, int(max_features * n_features_in_))` features are considered at each<br>  split.<br>- If &quot;sqrt&quot;, then `max_features=sqrt(n_features)`.<br>- If &quot;log2&quot;, then `max_features=log2(n_features)`.<br>- If None or 1.0, then `max_features=n_features`.<br><br>.. note::<br>    The default of 1.0 is equivalent to bagged trees and more<br>    randomness can be achieved by setting smaller values, e.g. 0.3.<br><br>.. versionchanged:: 1.1<br>    The default of `max_features` changed from `&quot;auto&quot;` to 1.0.<br><br>Note: the search for a split does not stop until at least one<br>valid partition of the node samples is found, even if it requires to<br>effectively inspect more than ``max_features`` features.</span>
        </a>
    </td>
            <td class="value">1.0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_leaf_nodes',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-max_leaf_nodes;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=max_leaf_nodes,-int%2C%20default%3DNone">
            max_leaf_nodes
            <span class="param-doc-description"
            style="position-anchor: --doc-link-max_leaf_nodes;">
            max_leaf_nodes: int, default=None<br><br>Grow trees with ``max_leaf_nodes`` in best-first fashion.<br>Best nodes are defined as relative reduction in impurity.<br>If None then unlimited number of leaf nodes.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_impurity_decrease',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-min_impurity_decrease;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=min_impurity_decrease,-float%2C%20default%3D0.0">
            min_impurity_decrease
            <span class="param-doc-description"
            style="position-anchor: --doc-link-min_impurity_decrease;">
            min_impurity_decrease: float, default=0.0<br><br>A node will be split if this split induces a decrease of the impurity<br>greater than or equal to this value.<br><br>The weighted impurity decrease equation is the following::<br><br>    N_t / N * (impurity - N_t_R / N_t * right_impurity<br>                        - N_t_L / N_t * left_impurity)<br><br>where ``N`` is the total number of samples, ``N_t`` is the number of<br>samples at the current node, ``N_t_L`` is the number of samples in the<br>left child, and ``N_t_R`` is the number of samples in the right child.<br><br>``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,<br>if ``sample_weight`` is passed.<br><br>.. versionadded:: 0.19</span>
        </a>
    </td>
            <td class="value">0.0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('bootstrap',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-bootstrap;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=bootstrap,-bool%2C%20default%3DTrue">
            bootstrap
            <span class="param-doc-description"
            style="position-anchor: --doc-link-bootstrap;">
            bootstrap: bool, default=True<br><br>Whether bootstrap samples are used when building trees. If False, the<br>whole dataset is used to build each tree.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('oob_score',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-oob_score;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=oob_score,-bool%20or%20callable%2C%20default%3DFalse">
            oob_score
            <span class="param-doc-description"
            style="position-anchor: --doc-link-oob_score;">
            oob_score: bool or callable, default=False<br><br>Whether to use out-of-bag samples to estimate the generalization score.<br>By default, :func:`~sklearn.metrics.r2_score` is used.<br>Provide a callable with signature `metric(y_true, y_pred)` to use a<br>custom metric. Only available if `bootstrap=True`.<br><br>For an illustration of out-of-bag (OOB) error estimation, see the example<br>:ref:`sphx_glr_auto_examples_ensemble_plot_ensemble_oob.py`.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('n_jobs',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-n_jobs;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=n_jobs,-int%2C%20default%3DNone">
            n_jobs
            <span class="param-doc-description"
            style="position-anchor: --doc-link-n_jobs;">
            n_jobs: int, default=None<br><br>The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,<br>:meth:`decision_path` and :meth:`apply` are all parallelized over the<br>trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`<br>context. ``-1`` means using all processors. See :term:`Glossary<br>&lt;n_jobs&gt;` for more details.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('verbose',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-verbose;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=verbose,-int%2C%20default%3D0">
            verbose
            <span class="param-doc-description"
            style="position-anchor: --doc-link-verbose;">
            verbose: int, default=0<br><br>Controls the verbosity when fitting and predicting.</span>
        </a>
    </td>
            <td class="value">0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('warm_start',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-warm_start;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=warm_start,-bool%2C%20default%3DFalse">
            warm_start
            <span class="param-doc-description"
            style="position-anchor: --doc-link-warm_start;">
            warm_start: bool, default=False<br><br>When set to ``True``, reuse the solution of the previous call to fit<br>and add more estimators to the ensemble, otherwise, just fit a whole<br>new forest. See :term:`Glossary &lt;warm_start&gt;` and<br>:ref:`tree_ensemble_warm_start` for details.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('ccp_alpha',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-ccp_alpha;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=ccp_alpha,-non-negative%20float%2C%20default%3D0.0">
            ccp_alpha
            <span class="param-doc-description"
            style="position-anchor: --doc-link-ccp_alpha;">
            ccp_alpha: non-negative float, default=0.0<br><br>Complexity parameter used for Minimal Cost-Complexity Pruning. The<br>subtree with the largest cost complexity that is smaller than<br>``ccp_alpha`` will be chosen. By default, no pruning is performed. See<br>:ref:`minimal_cost_complexity_pruning` for details. See<br>:ref:`sphx_glr_auto_examples_tree_plot_cost_complexity_pruning.py`<br>for an example of such pruning.<br><br>.. versionadded:: 0.22</span>
        </a>
    </td>
            <td class="value">0.0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_samples',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-max_samples;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=max_samples,-int%20or%20float%2C%20default%3DNone">
            max_samples
            <span class="param-doc-description"
            style="position-anchor: --doc-link-max_samples;">
            max_samples: int or float, default=None<br><br>If bootstrap is True, the number of samples to draw from X<br>to train each base estimator.<br><br>- If None (default), then draw `X.shape[0]` samples irrespective of<br>  `sample_weight`.<br>- If int, then draw `max_samples` samples.<br>- If float, then draw `max_samples * X.shape[0]` unweighted samples<br>  or `max_samples * sample_weight.sum()` weighted samples.<br><br>.. versionadded:: 0.22<br><br>.. versionchanged:: 1.9<br>    Float `max_samples` is relative to `sample_weight.sum()` instead of<br>    `X.shape[0]` for weighted samples.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('monotonic_cst',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-monotonic_cst;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=monotonic_cst,-array-like%20of%20int%20of%20shape%20%28n_features%29%2C%20default%3DNone">
            monotonic_cst
            <span class="param-doc-description"
            style="position-anchor: --doc-link-monotonic_cst;">
            monotonic_cst: array-like of int of shape (n_features), default=None<br><br>Indicates the monotonicity constraint to enforce on each feature.<br>  - 1: monotonically increasing<br>  - 0: no constraint<br>  - -1: monotonically decreasing<br><br>If monotonic_cst is None, no constraints are applied.<br><br>Monotonicity constraints are not supported for:<br>  - multioutput regressions (i.e. when `n_outputs_ &gt; 1`).<br><br>Read more in the :ref:`User Guide &lt;monotonic_cst_gbdt&gt;`.<br><br>.. versionadded:: 1.4</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>

        <div class="estimator-table">
            <details>
                <summary>Fitted attributes</summary>
                <table class="parameters-table">
                    <tbody>
                        <tr>
                        <th>Name</th>
                        <th>Type</th>
                        <th>Value</th>
                        </tr>

       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-estimator_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=estimator_,-%3Aclass%3A~sklearn.tree.DecisionTreeRegressor">
            estimator_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-estimator_;">
            estimator_: :class:`~sklearn.tree.DecisionTreeRegressor`<br><br>The child estimator template used to create the collection of fitted<br>sub-estimators.<br><br>.. versionadded:: 1.2<br>   `base_estimator_` was renamed to `estimator_`.</span>
        </a>
    </td>
           <td class="fitted-att-type">DecisionTreeRegressor</td>
           <td>DecisionTreeRegressor()</td>


       </tr>


       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-estimators_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=estimators_,-list%20of%20DecisionTreeRegressor">
            estimators_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-estimators_;">
            estimators_: list of DecisionTreeRegressor<br><br>The collection of fitted sub-estimators.</span>
        </a>
    </td>
           <td class="fitted-att-type">list</td>
           <td>[DecisionTreeR...te=1608637542), DecisionTreeR...te=1273642419), DecisionTreeR...te=1935803228), DecisionTreeR...ate=787846414), ...]</td>


       </tr>


       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-estimators_samples_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=estimators_samples_,-list%20of%20arrays">
            estimators_samples_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-estimators_samples_;">
            estimators_samples_: list of arrays<br><br>The subset of drawn samples (i.e., the in-bag samples) for each base<br>estimator. Each subset is defined by an array of the indices selected.<br><br>.. versionadded:: 1.4</span>
        </a>
    </td>
           <td class="fitted-att-type">list</td>
           <td>[array([475,  ..., dtype=int32), array([543, 5..., dtype=int32), array([342, 4..., dtype=int32), array([ 63, 2..., dtype=int32), ...]</td>


       </tr>


       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-feature_importances_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=feature_importances_,-ndarray%20of%20shape%20%28n_features%2C%29">
            feature_importances_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-feature_importances_;">
            feature_importances_: ndarray of shape (n_features,)<br><br>The impurity-based feature importances.<br>The higher, the more important the feature.<br>The importance of a feature is computed as the (normalized)<br>total reduction of the criterion brought by that feature.  It is also<br>known as the Gini importance.<br><br>Warning: impurity-based feature importances can be misleading for<br>high cardinality features (many unique values). See<br>:func:`sklearn.inspection.permutation_importance` as an alternative.</span>
        </a>
    </td>
           <td class="fitted-att-type">ndarray[float64](16,)</td>
           <td>[0.,0.,0.,...,0.,0.,0.]</td>


       </tr>


       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-feature_names_in_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=feature_names_in_,-ndarray%20of%20shape%20%28n_features_in_%2C%29">
            feature_names_in_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-feature_names_in_;">
            feature_names_in_: ndarray of shape (`n_features_in_`,)<br><br>Names of features seen during :term:`fit`. Defined only when `X`<br>has feature names that are all strings.<br><br>.. versionadded:: 1.0</span>
        </a>
    </td>
           <td class="fitted-att-type">ndarray[object](16,)</td>
           <td>[&#x27;Children apprehended and placed in CBP custody*&#x27;,
 &#x27;Children in CBP custody&#x27;,&#x27;Children transferred out of CBP custody&#x27;,...,
 &#x27;Month&#x27;,&#x27;Quarter&#x27;,&#x27;Year&#x27;]</td>


       </tr>


       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-n_features_in_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=n_features_in_,-int">
            n_features_in_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-n_features_in_;">
            n_features_in_: int<br><br>Number of features seen during :term:`fit`.<br><br>.. versionadded:: 0.24</span>
        </a>
    </td>
           <td class="fitted-att-type">int</td>
           <td>16</td>


       </tr>


       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-n_outputs_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=n_outputs_,-int">
            n_outputs_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-n_outputs_;">
            n_outputs_: int<br><br>The number of outputs when ``fit`` is performed.</span>
        </a>
    </td>
           <td class="fitted-att-type">int</td>
           <td>1</td>


       </tr>

                    </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div><script>/*  Authors: The scikit-learn developers
 SPDX-License-Identifier: BSD-3-Clause
*/

function copyToClipboard(text, element) {
    // Get the parameter prefix from the closest toggleable content
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const fullParamName = paramPrefix ? `${paramPrefix}${text}` : text;

    const originalStyle = element.style;
    const computedStyle = window.getComputedStyle(element);
    const originalWidth = computedStyle.width;
    const originalHTML = element.innerHTML.replace('Copied!', '');

    navigator.clipboard.writeText(fullParamName)
        .then(() => {
            element.style.width = originalWidth;
            element.style.color = 'green';
            element.innerHTML = "Copied!";

            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        })
        .catch(err => {
            console.error('Failed to copy:', err);
            element.style.color = 'red';
            element.innerHTML = "Failed!";
            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        });
    return false;
}

document.querySelectorAll('.copy-paste-icon').forEach(function(element) {
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';

    const parent = element.parentElement;
    if (!parent || !parent.nextElementSibling) {
        console.warn('Expected copy-paste icon is missing from the DOM structure');
        return;
    }

    const paramName = element.parentElement.nextElementSibling
        .textContent.trim().split(' ')[0];
    const fullParamName = paramPrefix ? `${paramPrefix}${paramName}` : paramName;

    element.setAttribute('title', fullParamName);
});

/**
 * Copy the list of feature names formatted as a Python list.
 *
 * @param {HTMLElement} element - The copy button inside a `.features` block; its siblings
 *   contain a `details` element and a table containing feature named.
 * @returns {boolean} Always returns `false` so callers can prevent the default click behavior.
 */
function copyFeatureNamesToClipboard(element) {
    var detailsElem = element.closest('.features').querySelector('details');
    var wasOpen = detailsElem.open;
    detailsElem.open = true;
    var content = element.closest('.features').querySelector('tbody')
                  .innerText.trim();
    if (!wasOpen) detailsElem.open = false;
    const rows = content.split('\n').map(row => `    "${row}"`);
    const formattedText = `[\n${rows.join(',\n')},\n]`;
    const originalHTML = element.innerHTML.replace('âœ”', '');
    const originalStyle = element.style;
    const copyMark = document.createElement('span');
    copyMark.innerHTML = 'âœ”';
    copyMark.style.color = 'blue';
    copyMark.style.fontSize = '1em';

    navigator.clipboard.writeText(formattedText)
        .then(() => {
            element.style.display = 'none';
            element.parentElement.appendChild(copyMark);

            setTimeout(() => {
                copyMark.remove();
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 1000);
        })
        .catch(err => {
            console.error('Failed to copy:', err);
            element.style.color = 'orange';
            element.innerHTML = "Failed!";
            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 1000);
        });
    return false;
}
/**
 * Adapted from Skrub
 * https://github.com/skrub-data/skrub/blob/403466d1d5d4dc76a7ef569b3f8228db59a31dc3/skrub/_reporting/_data/templates/report.js#L789
 * @returns "light" or "dark"
 */
function detectTheme(element) {
    const body = document.querySelector('body');

    // Check VSCode theme
    const themeKindAttr = body.getAttribute('data-vscode-theme-kind');
    const themeNameAttr = body.getAttribute('data-vscode-theme-name');

    if (themeKindAttr && themeNameAttr) {
        const themeKind = themeKindAttr.toLowerCase();
        const themeName = themeNameAttr.toLowerCase();

        if (themeKind.includes("dark") || themeName.includes("dark")) {
            return "dark";
        }
        if (themeKind.includes("light") || themeName.includes("light")) {
            return "light";
        }
    }

    // Check Jupyter theme
    if (body.getAttribute('data-jp-theme-light') === 'false') {
        return 'dark';
    } else if (body.getAttribute('data-jp-theme-light') === 'true') {
        return 'light';
    }

    // Guess based on a parent element's color
    const color = window.getComputedStyle(element.parentNode, null).getPropertyValue('color');
    const match = color.match(/^rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*$/i);
    if (match) {
        const [r, g, b] = [
            parseFloat(match[1]),
            parseFloat(match[2]),
            parseFloat(match[3])
        ];

        // https://en.wikipedia.org/wiki/HSL_and_HSV#Lightness
        const luma = 0.299 * r + 0.587 * g + 0.114 * b;

        if (luma > 180) {
            // If the text is very bright we have a dark theme
            return 'dark';
        }
        if (luma < 75) {
            // If the text is very dark we have a light theme
            return 'light';
        }
        // Otherwise fall back to the next heuristic.
    }

    // Fallback to system preference
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}


function forceTheme(elementId) {
    const estimatorElement = document.querySelector(`#${elementId}`);
    if (estimatorElement === null) {
        console.error(`Element with id ${elementId} not found.`);
    } else {
        const theme = detectTheme(estimatorElement);
        estimatorElement.classList.add(theme);
    }
}

forceTheme('sk-container-id-1');</script></body>




```python
rf_predictions = rf_model.predict(X_test)
```


```python
rf_mae = mean_absolute_error(y_test, rf_predictions)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))

rf_mape = np.mean(
    np.abs((y_test - rf_predictions) / y_test)
) * 100

print("Random Forest")
print("------------------------")
print(f"MAE  : {rf_mae:.2f}")
print(f"RMSE : {rf_rmse:.2f}")
print(f"MAPE : {rf_mape:.2f}%")
```

    Random Forest
    ------------------------
    MAE  : 66.44
    RMSE : 88.80
    MAPE : 3.08%
    


```python
plt.figure(figsize=(15,6))

plt.plot(y_test.index, y_test, label="Actual")

plt.plot(y_test.index, rf_predictions, label="Random Forest")

plt.title("Random Forest Forecast")

plt.legend()

plt.grid(True)

plt.show()
```


    
![png](output_94_0.png)
    



```python
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance)
```

                                                Feature  Importance
    4                                             Lag_1    0.718893
    7                                    Rolling_Mean_7    0.209808
    5                                             Lag_7    0.028152
    8                                   Rolling_Mean_14    0.027378
    6                                            Lag_14    0.006027
    1                           Children in CBP custody    0.004780
    2           Children transferred out of CBP custody    0.001406
    0   Children apprehended and placed in CBP custody*    0.000676
    13                                            Month    0.000662
    3                 Children discharged from HHS Care    0.000517
    9                                     Rolling_STD_7    0.000414
    11                                     Net_Pressure    0.000407
    10                                   Rolling_STD_14    0.000404
    12                                      Day_of_Week    0.000391
    15                                             Year    0.000057
    14                                          Quarter    0.000028
    


```python
plt.figure(figsize=(10,6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()

plt.show()
```


    
![png](output_96_0.png)
    



```python
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)
```




<style>.sk-global {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;
}

.sk-global.light {
  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: black;
  --sklearn-color-background: white;
  --sklearn-color-border-box: black;
  --sklearn-color-icon: #696969;
}

.sk-global.dark {
  --sklearn-color-text-on-default-background: white;
  --sklearn-color-background: #111;
  --sklearn-color-border-box: white;
  --sklearn-color-icon: #878787;
}

.sk-global {
  color: var(--sklearn-color-text);
}

.sk-global pre {
  padding: 0;
}

.sk-global input.sk-hidden--visually {
  border: 0;
  clip-path: inset(100%);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

.sk-global div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

.sk-global div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

.sk-global div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

.sk-global div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

.sk-global div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

.sk-global div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

.sk-global div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

.sk-global div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

.sk-global div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

.sk-global div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

.sk-global div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
.sk-global label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: center;
  justify-content: center;
  gap: 0.5em;
}

.sk-global label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

.sk-global label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

.sk-global label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

.sk-global div.sk-toggleable__content {
  display: none;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

.sk-global div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

.sk-global div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

.sk-global div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

.sk-global input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  display: block;
  width: 100%;
  overflow: visible;
}

.sk-global input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

.sk-global div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

.sk-global div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
.sk-global div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

.sk-global div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

.sk-global div.sk-label label.sk-toggleable__label,
.sk-global div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
.sk-global div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
.sk-global div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

.sk-global div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  line-height: 1.2em;
}

.sk-global div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
.sk-global div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

.sk-global div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
.sk-global div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

.sk-global div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-unfitted-level-0);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-3) 1pt solid;
  color: var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3) 1pt solid;
  color: var(--sklearn-color-fitted-level-3);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  border: var(--sklearn-color-fitted-level-0) 1pt solid;
  color: var(--sklearn-color-unfitted-level-0);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  border: var(--sklearn-color-fitted-level-0) 1pt solid;
  color: var(--sklearn-color-fitted-level-0);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

.sk-global a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-unfitted-level-0);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

.sk-global a.estimator_doc_link.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
.sk-global a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

.sk-global a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}

.sk-top-container.sk-global {
  /* pydata-sphinx-theme hides overflow, so scrolling is disabled.
   We need to set it to !important and add tabindex="0" in the HTML
   to allow keyboard-only users to navigate the display. */
  overflow-x: scroll !important;
  max-width: 100%;
}

.estimator-table {
    font-family: monospace;
}

.estimator-table summary {
    padding: .5rem;
    cursor: pointer;
}

.estimator-table summary::marker {
    font-size: 0.7rem;
}

.estimator-table details[open] {
    padding-left: 0.1rem;
    padding-right: 0.1rem;
    padding-bottom: 0.3rem;
}

.estimator-table .parameters-table {
    margin-left: auto !important;
    margin-right: auto !important;
    margin-top: 0;
}

.estimator-table .parameters-table tr:nth-child(odd) {
    background-color: #fff;
}

.estimator-table .parameters-table tr:nth-child(even) {
    background-color: #f6f6f6;
}

.estimator-table .parameters-table tr:hover td {
    background-color: #e0e0e0;
}

.estimator-table table :is(td, th) {
    border: 1px solid rgba(106, 105, 104, 0.232);
}

/*
    `table td`is set in notebook with right text-align.
    We need to overwrite it.
*/
.estimator-table table td.param {
    text-align: left;
    position: relative;
    padding: 0;
}

.user-set td {
    color:rgb(255, 94, 0);
    text-align: left !important;
}

.user-set td.value {
    color:rgb(255, 94, 0);
    background-color: transparent;
}

.default td, .estimator-table th {
    color: black;
    text-align: left !important;
}

.user-set td i,
.default td i {
    color: black;
}

td.fitted-att-type {
    white-space: preserve nowrap;
}

/*
    Styles for parameter documentation links
    We need styling for visited so jupyter doesn't overwrite it
*/
a.param-doc-link,
a.param-doc-link:link,
a.param-doc-link:visited {
    text-decoration: underline dashed;
    text-underline-offset: .3em;
    color: inherit;
    display: block;
    padding: .5em;
}

@supports(anchor-name: --doc-link) {
    a.param-doc-link,
    a.param-doc-link:link,
    a.param-doc-link:visited {
    anchor-name: --doc-link;
    }
}

/* "hack" to make the entire area of the cell containing the link clickable */
a.param-doc-link::before {
    position: absolute;
    content: "";
    inset: 0;
}

.param-doc-description {
    display: none;
    position: absolute;
    z-index: 9999;
    left: 0;
    padding: .5ex;
    margin-left: 1.5em;
    color: var(--sklearn-color-text);
    box-shadow: .3em .3em .4em #999;
    width: max-content;
    text-align: left;
    max-height: 10em;
    overflow-y: auto;

    /* unfitted */
    background: var(--sklearn-color-unfitted-level-0);
    border: thin solid var(--sklearn-color-unfitted-level-3);
}

@supports(position-area: center right) {
    .param-doc-description {
    position-area: center right;
    position: fixed;
    margin-left: 0;
    }
}

/* Fitted state for parameter tooltips */
.fitted .param-doc-description {
    /* fitted */
    background: var(--sklearn-color-fitted-level-0);
    border: thin solid var(--sklearn-color-fitted-level-3);
}

.param-doc-link:hover .param-doc-description {
    display: block;
}

.copy-paste-icon {
    background-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NDggNTEyIj48IS0tIUZvbnQgQXdlc29tZSBGcmVlIDYuNy4yIGJ5IEBmb250YXdlc29tZSAtIGh0dHBzOi8vZm9udGF3ZXNvbWUuY29tIExpY2Vuc2UgLSBodHRwczovL2ZvbnRhd2Vzb21lLmNvbS9saWNlbnNlL2ZyZWUgQ29weXJpZ2h0IDIwMjUgRm9udGljb25zLCBJbmMuLS0+PHBhdGggZD0iTTIwOCAwTDMzMi4xIDBjMTIuNyAwIDI0LjkgNS4xIDMzLjkgMTQuMWw2Ny45IDY3LjljOSA5IDE0LjEgMjEuMiAxNC4xIDMzLjlMNDQ4IDMzNmMwIDI2LjUtMjEuNSA0OC00OCA0OGwtMTkyIDBjLTI2LjUgMC00OC0yMS41LTQ4LTQ4bDAtMjg4YzAtMjYuNSAyMS41LTQ4IDQ4LTQ4ek00OCAxMjhsODAgMCAwIDY0LTY0IDAgMCAyNTYgMTkyIDAgMC0zMiA2NCAwIDAgNDhjMCAyNi41LTIxLjUgNDgtNDggNDhMNDggNTEyYy0yNi41IDAtNDgtMjEuNS00OC00OEwwIDE3NmMwLTI2LjUgMjEuNS00OCA0OC00OHoiLz48L3N2Zz4=);
    background-repeat: no-repeat;
    background-size: 14px 14px;
    background-position: 0;
    display: inline-block;
    width: 14px;
    height: 14px;
    cursor: pointer;
}

.features {
  font-family: monospace;
  cursor: pointer;
  background-color: var(--sklearn-color-unfitted-level-0);
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: .20em;
  margin-bottom: 0.5em;
  font-size: inherit; /* Needed for jupyter */
}

.features.fitted {
  background-color: var(--sklearn-color-fitted-level-0);
}

.features summary {
  cursor: pointer;
  display: flex;
  margin-bottom: 0;
  text-align: center;
  align-items: center;
  justify-content: center;
  gap: 0.5em;
  padding: .25em;
}

.features details[open] > summary {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
  border-radius: .20em 0 0 0;
}

.features.fitted details[open] > summary {
  background-color: var(--sklearn-color-fitted-level-2);
  border-radius: .20em 0 0 0;
}

.features details > summary .arrow::before {
  content: "▸";
  color: grey;
}

.features details[open] > summary .arrow::before {
  content: "▾";
}

.features details:hover > summary {
  margin: 0;
  background-color: var(--sklearn-color-unfitted-level-2);
}

.features.fitted details:hover > summary {
  margin: 0;
  background-color: var(--sklearn-color-fitted-level-2);
}

.features .features-container {
  max-width: 15em;
  max-height: 10em;
  overflow: auto;
  scrollbar-width: thin;
  padding: .25em 0.1rem;
  background-color: var(--sklearn-color-unfitted-level-0);
  border-radius: 0 0 .5em .5em;
}

.features.fitted .features-container {
  background-color: var(--sklearn-color-fitted-level-0);
}

.features .image-container {
  block-size: 1em;
  inline-size: 1em;
  padding: 0;
  margin: 0%;
  display: flex;
  justify-content: center;
  align-items: center;
}

.features .copy-paste-icon {
  background-size: 1em 1em;
  width: 1em;
  height: 1em;
  filter: grayscale(100%) opacity(60%);
}

.features .features-container table {
  width: 100%;
  margin: 0.01em;
}

.features .features-container table tr:nth-child(odd) {
  background-color: #fff;
}

.features .features-container table tr:nth-child(even) {
  background-color: #f6f6f6;
}

.features .features-container table tr:hover {
  background-color: #e0e0e0;
}

.features .features-container table {
  table-layout: inherit;
}

.features .features-container table td {
  text-align: left;
  padding: 0 0.5em;
  border: 1px solid rgba(106, 105, 104, 0.232);
  white-space: nowrap;
  color: var(--sklearn-color-text);
}

.total_features {
  display: flex;
  justify-content: center;
  margin-top: 0.5em;
}
</style><body><div id="sk-container-id-2" tabindex="0" class="sk-top-container sk-global"><div class="sk-text-repr-fallback"><pre>RandomForestRegressor(n_estimators=200, random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually sk-global" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>RandomForestRegressor</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html">?<span>Documentation for RandomForestRegressor</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted" data-param-prefix="">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('n_estimators',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-n_estimators;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=n_estimators,-int%2C%20default%3D100">
            n_estimators
            <span class="param-doc-description"
            style="position-anchor: --doc-link-n_estimators;">
            n_estimators: int, default=100<br><br>The number of trees in the forest.<br><br>.. versionchanged:: 0.22<br>   The default value of ``n_estimators`` changed from 10 to 100<br>   in 0.22.</span>
        </a>
    </td>
            <td class="value">200</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('random_state',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-random_state;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=random_state,-int%2C%20RandomState%20instance%20or%20None%2C%20default%3DNone">
            random_state
            <span class="param-doc-description"
            style="position-anchor: --doc-link-random_state;">
            random_state: int, RandomState instance or None, default=None<br><br>Controls both the randomness of the bootstrapping of the samples used<br>when building trees (if ``bootstrap=True``) and the sampling of the<br>features to consider when looking for the best split at each node<br>(if ``max_features &lt; n_features``).<br>See :term:`Glossary &lt;random_state&gt;` for details.</span>
        </a>
    </td>
            <td class="value">42</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('criterion',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-criterion;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=criterion,-%7B%22squared_error%22%2C%20%22absolute_error%22%2C%20%22poisson%22%7D%2C%20default%3D%22squared_error%22">
            criterion
            <span class="param-doc-description"
            style="position-anchor: --doc-link-criterion;">
            criterion: {&quot;squared_error&quot;, &quot;absolute_error&quot;, &quot;poisson&quot;}, default=&quot;squared_error&quot;<br><br>The function to measure the quality of a split. Supported criteria<br>are &quot;squared_error&quot; for the mean squared error, which is equal to<br>variance reduction as feature selection criterion and minimizes the L2<br>loss using the mean of each terminal node, &quot;absolute_error&quot; for the mean<br>absolute error, which minimizes the L1 loss using the median of each terminal<br>node, and &quot;poisson&quot; which uses reduction in Poisson deviance to find splits,<br>also using the mean of each terminal node.<br><br>.. versionadded:: 0.18<br>   Mean Absolute Error (MAE) criterion.<br><br>.. versionadded:: 1.0<br>   Poisson criterion.<br><br>.. versionchanged:: 1.9<br>    Criterion `&quot;friedman_mse&quot;` was deprecated.</span>
        </a>
    </td>
            <td class="value">&#x27;squared_error&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_depth',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-max_depth;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=max_depth,-int%2C%20default%3DNone">
            max_depth
            <span class="param-doc-description"
            style="position-anchor: --doc-link-max_depth;">
            max_depth: int, default=None<br><br>The maximum depth of the tree. If None, then nodes are expanded until<br>all leaves are pure or until all leaves contain less than<br>min_samples_split samples.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_samples_split',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-min_samples_split;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=min_samples_split,-int%20or%20float%2C%20default%3D2">
            min_samples_split
            <span class="param-doc-description"
            style="position-anchor: --doc-link-min_samples_split;">
            min_samples_split: int or float, default=2<br><br>The minimum number of samples required to split an internal node:<br><br>- If int, then consider `min_samples_split` as the minimum number.<br>- If float, then `min_samples_split` is a fraction and<br>  `ceil(min_samples_split * n_samples)` are the minimum<br>  number of samples for each split.<br><br>.. versionchanged:: 0.18<br>   Added float values for fractions.</span>
        </a>
    </td>
            <td class="value">2</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_samples_leaf',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-min_samples_leaf;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=min_samples_leaf,-int%20or%20float%2C%20default%3D1">
            min_samples_leaf
            <span class="param-doc-description"
            style="position-anchor: --doc-link-min_samples_leaf;">
            min_samples_leaf: int or float, default=1<br><br>The minimum number of samples required to be at a leaf node.<br>A split point at any depth will only be considered if it leaves at<br>least ``min_samples_leaf`` training samples in each of the left and<br>right branches.  This may have the effect of smoothing the model,<br>especially in regression.<br><br>- If int, then consider `min_samples_leaf` as the minimum number.<br>- If float, then `min_samples_leaf` is a fraction and<br>  `ceil(min_samples_leaf * n_samples)` are the minimum<br>  number of samples for each node.<br><br>.. versionchanged:: 0.18<br>   Added float values for fractions.</span>
        </a>
    </td>
            <td class="value">1</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_weight_fraction_leaf',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-min_weight_fraction_leaf;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=min_weight_fraction_leaf,-float%2C%20default%3D0.0">
            min_weight_fraction_leaf
            <span class="param-doc-description"
            style="position-anchor: --doc-link-min_weight_fraction_leaf;">
            min_weight_fraction_leaf: float, default=0.0<br><br>The minimum weighted fraction of the sum total of weights (of all<br>the input samples) required to be at a leaf node. Samples have<br>equal weight when sample_weight is not provided.</span>
        </a>
    </td>
            <td class="value">0.0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_features',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-max_features;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=max_features,-%7B%22sqrt%22%2C%20%22log2%22%2C%20None%7D%2C%20int%20or%20float%2C%20default%3D1.0">
            max_features
            <span class="param-doc-description"
            style="position-anchor: --doc-link-max_features;">
            max_features: {&quot;sqrt&quot;, &quot;log2&quot;, None}, int or float, default=1.0<br><br>The number of features to consider when looking for the best split:<br><br>- If int, then consider `max_features` features at each split.<br>- If float, then `max_features` is a fraction and<br>  `max(1, int(max_features * n_features_in_))` features are considered at each<br>  split.<br>- If &quot;sqrt&quot;, then `max_features=sqrt(n_features)`.<br>- If &quot;log2&quot;, then `max_features=log2(n_features)`.<br>- If None or 1.0, then `max_features=n_features`.<br><br>.. note::<br>    The default of 1.0 is equivalent to bagged trees and more<br>    randomness can be achieved by setting smaller values, e.g. 0.3.<br><br>.. versionchanged:: 1.1<br>    The default of `max_features` changed from `&quot;auto&quot;` to 1.0.<br><br>Note: the search for a split does not stop until at least one<br>valid partition of the node samples is found, even if it requires to<br>effectively inspect more than ``max_features`` features.</span>
        </a>
    </td>
            <td class="value">1.0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_leaf_nodes',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-max_leaf_nodes;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=max_leaf_nodes,-int%2C%20default%3DNone">
            max_leaf_nodes
            <span class="param-doc-description"
            style="position-anchor: --doc-link-max_leaf_nodes;">
            max_leaf_nodes: int, default=None<br><br>Grow trees with ``max_leaf_nodes`` in best-first fashion.<br>Best nodes are defined as relative reduction in impurity.<br>If None then unlimited number of leaf nodes.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_impurity_decrease',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-min_impurity_decrease;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=min_impurity_decrease,-float%2C%20default%3D0.0">
            min_impurity_decrease
            <span class="param-doc-description"
            style="position-anchor: --doc-link-min_impurity_decrease;">
            min_impurity_decrease: float, default=0.0<br><br>A node will be split if this split induces a decrease of the impurity<br>greater than or equal to this value.<br><br>The weighted impurity decrease equation is the following::<br><br>    N_t / N * (impurity - N_t_R / N_t * right_impurity<br>                        - N_t_L / N_t * left_impurity)<br><br>where ``N`` is the total number of samples, ``N_t`` is the number of<br>samples at the current node, ``N_t_L`` is the number of samples in the<br>left child, and ``N_t_R`` is the number of samples in the right child.<br><br>``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,<br>if ``sample_weight`` is passed.<br><br>.. versionadded:: 0.19</span>
        </a>
    </td>
            <td class="value">0.0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('bootstrap',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-bootstrap;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=bootstrap,-bool%2C%20default%3DTrue">
            bootstrap
            <span class="param-doc-description"
            style="position-anchor: --doc-link-bootstrap;">
            bootstrap: bool, default=True<br><br>Whether bootstrap samples are used when building trees. If False, the<br>whole dataset is used to build each tree.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('oob_score',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-oob_score;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=oob_score,-bool%20or%20callable%2C%20default%3DFalse">
            oob_score
            <span class="param-doc-description"
            style="position-anchor: --doc-link-oob_score;">
            oob_score: bool or callable, default=False<br><br>Whether to use out-of-bag samples to estimate the generalization score.<br>By default, :func:`~sklearn.metrics.r2_score` is used.<br>Provide a callable with signature `metric(y_true, y_pred)` to use a<br>custom metric. Only available if `bootstrap=True`.<br><br>For an illustration of out-of-bag (OOB) error estimation, see the example<br>:ref:`sphx_glr_auto_examples_ensemble_plot_ensemble_oob.py`.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('n_jobs',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-n_jobs;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=n_jobs,-int%2C%20default%3DNone">
            n_jobs
            <span class="param-doc-description"
            style="position-anchor: --doc-link-n_jobs;">
            n_jobs: int, default=None<br><br>The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,<br>:meth:`decision_path` and :meth:`apply` are all parallelized over the<br>trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`<br>context. ``-1`` means using all processors. See :term:`Glossary<br>&lt;n_jobs&gt;` for more details.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('verbose',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-verbose;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=verbose,-int%2C%20default%3D0">
            verbose
            <span class="param-doc-description"
            style="position-anchor: --doc-link-verbose;">
            verbose: int, default=0<br><br>Controls the verbosity when fitting and predicting.</span>
        </a>
    </td>
            <td class="value">0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('warm_start',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-warm_start;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=warm_start,-bool%2C%20default%3DFalse">
            warm_start
            <span class="param-doc-description"
            style="position-anchor: --doc-link-warm_start;">
            warm_start: bool, default=False<br><br>When set to ``True``, reuse the solution of the previous call to fit<br>and add more estimators to the ensemble, otherwise, just fit a whole<br>new forest. See :term:`Glossary &lt;warm_start&gt;` and<br>:ref:`tree_ensemble_warm_start` for details.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('ccp_alpha',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-ccp_alpha;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=ccp_alpha,-non-negative%20float%2C%20default%3D0.0">
            ccp_alpha
            <span class="param-doc-description"
            style="position-anchor: --doc-link-ccp_alpha;">
            ccp_alpha: non-negative float, default=0.0<br><br>Complexity parameter used for Minimal Cost-Complexity Pruning. The<br>subtree with the largest cost complexity that is smaller than<br>``ccp_alpha`` will be chosen. By default, no pruning is performed. See<br>:ref:`minimal_cost_complexity_pruning` for details. See<br>:ref:`sphx_glr_auto_examples_tree_plot_cost_complexity_pruning.py`<br>for an example of such pruning.<br><br>.. versionadded:: 0.22</span>
        </a>
    </td>
            <td class="value">0.0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_samples',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-max_samples;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=max_samples,-int%20or%20float%2C%20default%3DNone">
            max_samples
            <span class="param-doc-description"
            style="position-anchor: --doc-link-max_samples;">
            max_samples: int or float, default=None<br><br>If bootstrap is True, the number of samples to draw from X<br>to train each base estimator.<br><br>- If None (default), then draw `X.shape[0]` samples irrespective of<br>  `sample_weight`.<br>- If int, then draw `max_samples` samples.<br>- If float, then draw `max_samples * X.shape[0]` unweighted samples<br>  or `max_samples * sample_weight.sum()` weighted samples.<br><br>.. versionadded:: 0.22<br><br>.. versionchanged:: 1.9<br>    Float `max_samples` is relative to `sample_weight.sum()` instead of<br>    `X.shape[0]` for weighted samples.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('monotonic_cst',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-monotonic_cst;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=monotonic_cst,-array-like%20of%20int%20of%20shape%20%28n_features%29%2C%20default%3DNone">
            monotonic_cst
            <span class="param-doc-description"
            style="position-anchor: --doc-link-monotonic_cst;">
            monotonic_cst: array-like of int of shape (n_features), default=None<br><br>Indicates the monotonicity constraint to enforce on each feature.<br>  - 1: monotonically increasing<br>  - 0: no constraint<br>  - -1: monotonically decreasing<br><br>If monotonic_cst is None, no constraints are applied.<br><br>Monotonicity constraints are not supported for:<br>  - multioutput regressions (i.e. when `n_outputs_ &gt; 1`).<br><br>Read more in the :ref:`User Guide &lt;monotonic_cst_gbdt&gt;`.<br><br>.. versionadded:: 1.4</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>

        <div class="estimator-table">
            <details>
                <summary>Fitted attributes</summary>
                <table class="parameters-table">
                    <tbody>
                        <tr>
                        <th>Name</th>
                        <th>Type</th>
                        <th>Value</th>
                        </tr>

       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-estimator_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=estimator_,-%3Aclass%3A~sklearn.tree.DecisionTreeRegressor">
            estimator_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-estimator_;">
            estimator_: :class:`~sklearn.tree.DecisionTreeRegressor`<br><br>The child estimator template used to create the collection of fitted<br>sub-estimators.<br><br>.. versionadded:: 1.2<br>   `base_estimator_` was renamed to `estimator_`.</span>
        </a>
    </td>
           <td class="fitted-att-type">DecisionTreeRegressor</td>
           <td>DecisionTreeRegressor()</td>


       </tr>


       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-estimators_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=estimators_,-list%20of%20DecisionTreeRegressor">
            estimators_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-estimators_;">
            estimators_: list of DecisionTreeRegressor<br><br>The collection of fitted sub-estimators.</span>
        </a>
    </td>
           <td class="fitted-att-type">list</td>
           <td>[DecisionTreeR...te=1608637542), DecisionTreeR...te=1273642419), DecisionTreeR...te=1935803228), DecisionTreeR...ate=787846414), ...]</td>


       </tr>


       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-estimators_samples_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=estimators_samples_,-list%20of%20arrays">
            estimators_samples_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-estimators_samples_;">
            estimators_samples_: list of arrays<br><br>The subset of drawn samples (i.e., the in-bag samples) for each base<br>estimator. Each subset is defined by an array of the indices selected.<br><br>.. versionadded:: 1.4</span>
        </a>
    </td>
           <td class="fitted-att-type">list</td>
           <td>[array([475,  ..., dtype=int32), array([543, 5..., dtype=int32), array([342, 4..., dtype=int32), array([ 63, 2..., dtype=int32), ...]</td>


       </tr>


       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-feature_importances_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=feature_importances_,-ndarray%20of%20shape%20%28n_features%2C%29">
            feature_importances_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-feature_importances_;">
            feature_importances_: ndarray of shape (n_features,)<br><br>The impurity-based feature importances.<br>The higher, the more important the feature.<br>The importance of a feature is computed as the (normalized)<br>total reduction of the criterion brought by that feature.  It is also<br>known as the Gini importance.<br><br>Warning: impurity-based feature importances can be misleading for<br>high cardinality features (many unique values). See<br>:func:`sklearn.inspection.permutation_importance` as an alternative.</span>
        </a>
    </td>
           <td class="fitted-att-type">ndarray[float64](16,)</td>
           <td>[0.,0.,0.,...,0.,0.,0.]</td>


       </tr>


       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-feature_names_in_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=feature_names_in_,-ndarray%20of%20shape%20%28n_features_in_%2C%29">
            feature_names_in_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-feature_names_in_;">
            feature_names_in_: ndarray of shape (`n_features_in_`,)<br><br>Names of features seen during :term:`fit`. Defined only when `X`<br>has feature names that are all strings.<br><br>.. versionadded:: 1.0</span>
        </a>
    </td>
           <td class="fitted-att-type">ndarray[object](16,)</td>
           <td>[&#x27;Children apprehended and placed in CBP custody*&#x27;,
 &#x27;Children in CBP custody&#x27;,&#x27;Children transferred out of CBP custody&#x27;,...,
 &#x27;Month&#x27;,&#x27;Quarter&#x27;,&#x27;Year&#x27;]</td>


       </tr>


       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-n_features_in_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=n_features_in_,-int">
            n_features_in_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-n_features_in_;">
            n_features_in_: int<br><br>Number of features seen during :term:`fit`.<br><br>.. versionadded:: 0.24</span>
        </a>
    </td>
           <td class="fitted-att-type">int</td>
           <td>16</td>


       </tr>


       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-n_outputs_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html#:~:text=n_outputs_,-int">
            n_outputs_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-n_outputs_;">
            n_outputs_: int<br><br>The number of outputs when ``fit`` is performed.</span>
        </a>
    </td>
           <td class="fitted-att-type">int</td>
           <td>1</td>


       </tr>

                    </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div><script>/*  Authors: The scikit-learn developers
 SPDX-License-Identifier: BSD-3-Clause
*/

function copyToClipboard(text, element) {
    // Get the parameter prefix from the closest toggleable content
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const fullParamName = paramPrefix ? `${paramPrefix}${text}` : text;

    const originalStyle = element.style;
    const computedStyle = window.getComputedStyle(element);
    const originalWidth = computedStyle.width;
    const originalHTML = element.innerHTML.replace('Copied!', '');

    navigator.clipboard.writeText(fullParamName)
        .then(() => {
            element.style.width = originalWidth;
            element.style.color = 'green';
            element.innerHTML = "Copied!";

            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        })
        .catch(err => {
            console.error('Failed to copy:', err);
            element.style.color = 'red';
            element.innerHTML = "Failed!";
            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        });
    return false;
}

document.querySelectorAll('.copy-paste-icon').forEach(function(element) {
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';

    const parent = element.parentElement;
    if (!parent || !parent.nextElementSibling) {
        console.warn('Expected copy-paste icon is missing from the DOM structure');
        return;
    }

    const paramName = element.parentElement.nextElementSibling
        .textContent.trim().split(' ')[0];
    const fullParamName = paramPrefix ? `${paramPrefix}${paramName}` : paramName;

    element.setAttribute('title', fullParamName);
});

/**
 * Copy the list of feature names formatted as a Python list.
 *
 * @param {HTMLElement} element - The copy button inside a `.features` block; its siblings
 *   contain a `details` element and a table containing feature named.
 * @returns {boolean} Always returns `false` so callers can prevent the default click behavior.
 */
function copyFeatureNamesToClipboard(element) {
    var detailsElem = element.closest('.features').querySelector('details');
    var wasOpen = detailsElem.open;
    detailsElem.open = true;
    var content = element.closest('.features').querySelector('tbody')
                  .innerText.trim();
    if (!wasOpen) detailsElem.open = false;
    const rows = content.split('\n').map(row => `    "${row}"`);
    const formattedText = `[\n${rows.join(',\n')},\n]`;
    const originalHTML = element.innerHTML.replace('âœ”', '');
    const originalStyle = element.style;
    const copyMark = document.createElement('span');
    copyMark.innerHTML = 'âœ”';
    copyMark.style.color = 'blue';
    copyMark.style.fontSize = '1em';

    navigator.clipboard.writeText(formattedText)
        .then(() => {
            element.style.display = 'none';
            element.parentElement.appendChild(copyMark);

            setTimeout(() => {
                copyMark.remove();
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 1000);
        })
        .catch(err => {
            console.error('Failed to copy:', err);
            element.style.color = 'orange';
            element.innerHTML = "Failed!";
            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 1000);
        });
    return false;
}
/**
 * Adapted from Skrub
 * https://github.com/skrub-data/skrub/blob/403466d1d5d4dc76a7ef569b3f8228db59a31dc3/skrub/_reporting/_data/templates/report.js#L789
 * @returns "light" or "dark"
 */
function detectTheme(element) {
    const body = document.querySelector('body');

    // Check VSCode theme
    const themeKindAttr = body.getAttribute('data-vscode-theme-kind');
    const themeNameAttr = body.getAttribute('data-vscode-theme-name');

    if (themeKindAttr && themeNameAttr) {
        const themeKind = themeKindAttr.toLowerCase();
        const themeName = themeNameAttr.toLowerCase();

        if (themeKind.includes("dark") || themeName.includes("dark")) {
            return "dark";
        }
        if (themeKind.includes("light") || themeName.includes("light")) {
            return "light";
        }
    }

    // Check Jupyter theme
    if (body.getAttribute('data-jp-theme-light') === 'false') {
        return 'dark';
    } else if (body.getAttribute('data-jp-theme-light') === 'true') {
        return 'light';
    }

    // Guess based on a parent element's color
    const color = window.getComputedStyle(element.parentNode, null).getPropertyValue('color');
    const match = color.match(/^rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*$/i);
    if (match) {
        const [r, g, b] = [
            parseFloat(match[1]),
            parseFloat(match[2]),
            parseFloat(match[3])
        ];

        // https://en.wikipedia.org/wiki/HSL_and_HSV#Lightness
        const luma = 0.299 * r + 0.587 * g + 0.114 * b;

        if (luma > 180) {
            // If the text is very bright we have a dark theme
            return 'dark';
        }
        if (luma < 75) {
            // If the text is very dark we have a light theme
            return 'light';
        }
        // Otherwise fall back to the next heuristic.
    }

    // Fallback to system preference
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}


function forceTheme(elementId) {
    const estimatorElement = document.querySelector(`#${elementId}`);
    if (estimatorElement === null) {
        console.error(`Element with id ${elementId} not found.`);
    } else {
        const theme = detectTheme(estimatorElement);
        estimatorElement.classList.add(theme);
    }
}

forceTheme('sk-container-id-2');</script></body>




```python
rf_predictions = rf_model.predict(X_test)
```


```python
rf_mae = mean_absolute_error(y_test, rf_predictions)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))

rf_mape = mean_absolute_percentage_error(y_test, rf_predictions) * 100

print("Random Forest Results")
print("-" * 30)
print(f"MAE  : {rf_mae:.2f}")
print(f"RMSE : {rf_rmse:.2f}")
print(f"MAPE : {rf_mape:.2f}%")
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[137], line 5
          1 rf_mae = mean_absolute_error(y_test, rf_predictions)
          3 rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
    ----> 5 rf_mape = mean_absolute_percentage_error(y_test, rf_predictions) * 100
          7 print("Random Forest Results")
          8 print("-" * 30)
    

    NameError: name 'mean_absolute_percentage_error' is not defined



```python
plt.figure(figsize=(15,6))

plt.plot(y_test.index, y_test, label="Actual", linewidth=2)

plt.plot(y_test.index, rf_predictions, label="Random Forest", linewidth=2)

plt.title("Random Forest Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel("Children in HHS Care")
plt.legend()
plt.grid(True)

plt.show()
```


    
![png](output_100_0.png)
    



```python
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance)
```

                                                Feature  Importance
    4                                             Lag_1    0.718893
    7                                    Rolling_Mean_7    0.209808
    5                                             Lag_7    0.028152
    8                                   Rolling_Mean_14    0.027378
    6                                            Lag_14    0.006027
    1                           Children in CBP custody    0.004780
    2           Children transferred out of CBP custody    0.001406
    0   Children apprehended and placed in CBP custody*    0.000676
    13                                            Month    0.000662
    3                 Children discharged from HHS Care    0.000517
    9                                     Rolling_STD_7    0.000414
    11                                     Net_Pressure    0.000407
    10                                   Rolling_STD_14    0.000404
    12                                      Day_of_Week    0.000391
    15                                             Year    0.000057
    14                                          Quarter    0.000028
    


```python
plt.figure(figsize=(10,6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.gca().invert_yaxis()

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.show()
```


    
![png](output_102_0.png)
    



```python
from sklearn.ensemble import GradientBoostingRegressor
```


```python
# ==========================================
# Gradient Boosting Model
# ==========================================

gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gb_model.fit(X_train, y_train)
```




<style>.sk-global {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;
}

.sk-global.light {
  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: black;
  --sklearn-color-background: white;
  --sklearn-color-border-box: black;
  --sklearn-color-icon: #696969;
}

.sk-global.dark {
  --sklearn-color-text-on-default-background: white;
  --sklearn-color-background: #111;
  --sklearn-color-border-box: white;
  --sklearn-color-icon: #878787;
}

.sk-global {
  color: var(--sklearn-color-text);
}

.sk-global pre {
  padding: 0;
}

.sk-global input.sk-hidden--visually {
  border: 0;
  clip-path: inset(100%);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

.sk-global div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

.sk-global div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

.sk-global div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

.sk-global div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

.sk-global div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

.sk-global div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

.sk-global div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

.sk-global div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

.sk-global div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

.sk-global div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

.sk-global div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
.sk-global label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: center;
  justify-content: center;
  gap: 0.5em;
}

.sk-global label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

.sk-global label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

.sk-global label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

.sk-global div.sk-toggleable__content {
  display: none;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

.sk-global div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

.sk-global div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

.sk-global div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

.sk-global input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  display: block;
  width: 100%;
  overflow: visible;
}

.sk-global input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

.sk-global div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

.sk-global div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
.sk-global div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

.sk-global div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

.sk-global div.sk-label label.sk-toggleable__label,
.sk-global div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
.sk-global div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
.sk-global div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

.sk-global div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  line-height: 1.2em;
}

.sk-global div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
.sk-global div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

.sk-global div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
.sk-global div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

.sk-global div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-unfitted-level-0);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-3) 1pt solid;
  color: var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3) 1pt solid;
  color: var(--sklearn-color-fitted-level-3);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  border: var(--sklearn-color-fitted-level-0) 1pt solid;
  color: var(--sklearn-color-unfitted-level-0);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  border: var(--sklearn-color-fitted-level-0) 1pt solid;
  color: var(--sklearn-color-fitted-level-0);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

.sk-global a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-unfitted-level-0);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

.sk-global a.estimator_doc_link.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
.sk-global a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

.sk-global a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}

.sk-top-container.sk-global {
  /* pydata-sphinx-theme hides overflow, so scrolling is disabled.
   We need to set it to !important and add tabindex="0" in the HTML
   to allow keyboard-only users to navigate the display. */
  overflow-x: scroll !important;
  max-width: 100%;
}

.estimator-table {
    font-family: monospace;
}

.estimator-table summary {
    padding: .5rem;
    cursor: pointer;
}

.estimator-table summary::marker {
    font-size: 0.7rem;
}

.estimator-table details[open] {
    padding-left: 0.1rem;
    padding-right: 0.1rem;
    padding-bottom: 0.3rem;
}

.estimator-table .parameters-table {
    margin-left: auto !important;
    margin-right: auto !important;
    margin-top: 0;
}

.estimator-table .parameters-table tr:nth-child(odd) {
    background-color: #fff;
}

.estimator-table .parameters-table tr:nth-child(even) {
    background-color: #f6f6f6;
}

.estimator-table .parameters-table tr:hover td {
    background-color: #e0e0e0;
}

.estimator-table table :is(td, th) {
    border: 1px solid rgba(106, 105, 104, 0.232);
}

/*
    `table td`is set in notebook with right text-align.
    We need to overwrite it.
*/
.estimator-table table td.param {
    text-align: left;
    position: relative;
    padding: 0;
}

.user-set td {
    color:rgb(255, 94, 0);
    text-align: left !important;
}

.user-set td.value {
    color:rgb(255, 94, 0);
    background-color: transparent;
}

.default td, .estimator-table th {
    color: black;
    text-align: left !important;
}

.user-set td i,
.default td i {
    color: black;
}

td.fitted-att-type {
    white-space: preserve nowrap;
}

/*
    Styles for parameter documentation links
    We need styling for visited so jupyter doesn't overwrite it
*/
a.param-doc-link,
a.param-doc-link:link,
a.param-doc-link:visited {
    text-decoration: underline dashed;
    text-underline-offset: .3em;
    color: inherit;
    display: block;
    padding: .5em;
}

@supports(anchor-name: --doc-link) {
    a.param-doc-link,
    a.param-doc-link:link,
    a.param-doc-link:visited {
    anchor-name: --doc-link;
    }
}

/* "hack" to make the entire area of the cell containing the link clickable */
a.param-doc-link::before {
    position: absolute;
    content: "";
    inset: 0;
}

.param-doc-description {
    display: none;
    position: absolute;
    z-index: 9999;
    left: 0;
    padding: .5ex;
    margin-left: 1.5em;
    color: var(--sklearn-color-text);
    box-shadow: .3em .3em .4em #999;
    width: max-content;
    text-align: left;
    max-height: 10em;
    overflow-y: auto;

    /* unfitted */
    background: var(--sklearn-color-unfitted-level-0);
    border: thin solid var(--sklearn-color-unfitted-level-3);
}

@supports(position-area: center right) {
    .param-doc-description {
    position-area: center right;
    position: fixed;
    margin-left: 0;
    }
}

/* Fitted state for parameter tooltips */
.fitted .param-doc-description {
    /* fitted */
    background: var(--sklearn-color-fitted-level-0);
    border: thin solid var(--sklearn-color-fitted-level-3);
}

.param-doc-link:hover .param-doc-description {
    display: block;
}

.copy-paste-icon {
    background-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NDggNTEyIj48IS0tIUZvbnQgQXdlc29tZSBGcmVlIDYuNy4yIGJ5IEBmb250YXdlc29tZSAtIGh0dHBzOi8vZm9udGF3ZXNvbWUuY29tIExpY2Vuc2UgLSBodHRwczovL2ZvbnRhd2Vzb21lLmNvbS9saWNlbnNlL2ZyZWUgQ29weXJpZ2h0IDIwMjUgRm9udGljb25zLCBJbmMuLS0+PHBhdGggZD0iTTIwOCAwTDMzMi4xIDBjMTIuNyAwIDI0LjkgNS4xIDMzLjkgMTQuMWw2Ny45IDY3LjljOSA5IDE0LjEgMjEuMiAxNC4xIDMzLjlMNDQ4IDMzNmMwIDI2LjUtMjEuNSA0OC00OCA0OGwtMTkyIDBjLTI2LjUgMC00OC0yMS41LTQ4LTQ4bDAtMjg4YzAtMjYuNSAyMS41LTQ4IDQ4LTQ4ek00OCAxMjhsODAgMCAwIDY0LTY0IDAgMCAyNTYgMTkyIDAgMC0zMiA2NCAwIDAgNDhjMCAyNi41LTIxLjUgNDgtNDggNDhMNDggNTEyYy0yNi41IDAtNDgtMjEuNS00OC00OEwwIDE3NmMwLTI2LjUgMjEuNS00OCA0OC00OHoiLz48L3N2Zz4=);
    background-repeat: no-repeat;
    background-size: 14px 14px;
    background-position: 0;
    display: inline-block;
    width: 14px;
    height: 14px;
    cursor: pointer;
}

.features {
  font-family: monospace;
  cursor: pointer;
  background-color: var(--sklearn-color-unfitted-level-0);
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: .20em;
  margin-bottom: 0.5em;
  font-size: inherit; /* Needed for jupyter */
}

.features.fitted {
  background-color: var(--sklearn-color-fitted-level-0);
}

.features summary {
  cursor: pointer;
  display: flex;
  margin-bottom: 0;
  text-align: center;
  align-items: center;
  justify-content: center;
  gap: 0.5em;
  padding: .25em;
}

.features details[open] > summary {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
  border-radius: .20em 0 0 0;
}

.features.fitted details[open] > summary {
  background-color: var(--sklearn-color-fitted-level-2);
  border-radius: .20em 0 0 0;
}

.features details > summary .arrow::before {
  content: "▸";
  color: grey;
}

.features details[open] > summary .arrow::before {
  content: "▾";
}

.features details:hover > summary {
  margin: 0;
  background-color: var(--sklearn-color-unfitted-level-2);
}

.features.fitted details:hover > summary {
  margin: 0;
  background-color: var(--sklearn-color-fitted-level-2);
}

.features .features-container {
  max-width: 15em;
  max-height: 10em;
  overflow: auto;
  scrollbar-width: thin;
  padding: .25em 0.1rem;
  background-color: var(--sklearn-color-unfitted-level-0);
  border-radius: 0 0 .5em .5em;
}

.features.fitted .features-container {
  background-color: var(--sklearn-color-fitted-level-0);
}

.features .image-container {
  block-size: 1em;
  inline-size: 1em;
  padding: 0;
  margin: 0%;
  display: flex;
  justify-content: center;
  align-items: center;
}

.features .copy-paste-icon {
  background-size: 1em 1em;
  width: 1em;
  height: 1em;
  filter: grayscale(100%) opacity(60%);
}

.features .features-container table {
  width: 100%;
  margin: 0.01em;
}

.features .features-container table tr:nth-child(odd) {
  background-color: #fff;
}

.features .features-container table tr:nth-child(even) {
  background-color: #f6f6f6;
}

.features .features-container table tr:hover {
  background-color: #e0e0e0;
}

.features .features-container table {
  table-layout: inherit;
}

.features .features-container table td {
  text-align: left;
  padding: 0 0.5em;
  border: 1px solid rgba(106, 105, 104, 0.232);
  white-space: nowrap;
  color: var(--sklearn-color-text);
}

.total_features {
  display: flex;
  justify-content: center;
  margin-top: 0.5em;
}
</style><body><div id="sk-container-id-3" tabindex="0" class="sk-top-container sk-global"><div class="sk-text-repr-fallback"><pre>GradientBoostingRegressor(learning_rate=0.05, n_estimators=200, random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually sk-global" id="sk-estimator-id-3" type="checkbox" checked><label for="sk-estimator-id-3" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>GradientBoostingRegressor</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html">?<span>Documentation for GradientBoostingRegressor</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted" data-param-prefix="">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('learning_rate',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-learning_rate;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=learning_rate,-float%2C%20default%3D0.1">
            learning_rate
            <span class="param-doc-description"
            style="position-anchor: --doc-link-learning_rate;">
            learning_rate: float, default=0.1<br><br>Learning rate shrinks the contribution of each tree by `learning_rate`.<br>There is a trade-off between learning_rate and n_estimators.<br>Values must be in the range `[0.0, inf)`.</span>
        </a>
    </td>
            <td class="value">0.05</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('n_estimators',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-n_estimators;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=n_estimators,-int%2C%20default%3D100">
            n_estimators
            <span class="param-doc-description"
            style="position-anchor: --doc-link-n_estimators;">
            n_estimators: int, default=100<br><br>The number of boosting stages to perform. Gradient boosting<br>is fairly robust to over-fitting so a large number usually<br>results in better performance.<br>Values must be in the range `[1, inf)`.</span>
        </a>
    </td>
            <td class="value">200</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('random_state',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-random_state;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=random_state,-int%2C%20RandomState%20instance%20or%20None%2C%20default%3DNone">
            random_state
            <span class="param-doc-description"
            style="position-anchor: --doc-link-random_state;">
            random_state: int, RandomState instance or None, default=None<br><br>Controls the random seed given to each Tree estimator at each<br>boosting iteration.<br>In addition, it controls the random permutation of the features at<br>each split (see Notes for more details).<br>It also controls the random splitting of the training data to obtain a<br>validation set if `n_iter_no_change` is not None.<br>Pass an int for reproducible output across multiple function calls.<br>See :term:`Glossary &lt;random_state&gt;`.</span>
        </a>
    </td>
            <td class="value">42</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('loss',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-loss;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=loss,-%7B%27squared_error%27%2C%20%27absolute_error%27%2C%20%27huber%27%2C%20%27quantile%27%7D%2C%20%20%20%20%20%20%20%20%20%20%20%20%20default%3D%27squared_error%27">
            loss
            <span class="param-doc-description"
            style="position-anchor: --doc-link-loss;">
            loss: {&#x27;squared_error&#x27;, &#x27;absolute_error&#x27;, &#x27;huber&#x27;, &#x27;quantile&#x27;},             default=&#x27;squared_error&#x27;<br><br>Loss function to be optimized. &#x27;squared_error&#x27; refers to the squared<br>error for regression. &#x27;absolute_error&#x27; refers to the absolute error of<br>regression and is a robust loss function. &#x27;huber&#x27; is a<br>combination of the two. &#x27;quantile&#x27; allows quantile regression (use<br>`alpha` to specify the quantile).<br>See<br>:ref:`sphx_glr_auto_examples_ensemble_plot_gradient_boosting_quantile.py`<br>for an example that demonstrates quantile regression for creating<br>prediction intervals with `loss=&#x27;quantile&#x27;`.</span>
        </a>
    </td>
            <td class="value">&#x27;squared_error&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('subsample',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-subsample;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=subsample,-float%2C%20default%3D1.0">
            subsample
            <span class="param-doc-description"
            style="position-anchor: --doc-link-subsample;">
            subsample: float, default=1.0<br><br>The fraction of samples to be used for fitting the individual base<br>learners. If smaller than 1.0 this results in Stochastic Gradient<br>Boosting. `subsample` interacts with the parameter `n_estimators`.<br>Choosing `subsample &lt; 1.0` leads to a reduction of variance<br>and an increase in bias.<br>Values must be in the range `(0.0, 1.0]`.</span>
        </a>
    </td>
            <td class="value">1.0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('criterion',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-criterion;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=criterion,-%7B%27friedman_mse%27%2C%20%27squared_error%27%7D%2C%20default%3D%27friedman_mse%27">
            criterion
            <span class="param-doc-description"
            style="position-anchor: --doc-link-criterion;">
            criterion: {&#x27;friedman_mse&#x27;, &#x27;squared_error&#x27;}, default=&#x27;friedman_mse&#x27;<br><br>This parameter has no effect.<br><br>.. versionadded:: 0.18<br><br>.. deprecated:: 1.9<br>   `criterion` is deprecated and will be removed in 1.11.</span>
        </a>
    </td>
            <td class="value">&#x27;deprecated&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_samples_split',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-min_samples_split;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=min_samples_split,-int%20or%20float%2C%20default%3D2">
            min_samples_split
            <span class="param-doc-description"
            style="position-anchor: --doc-link-min_samples_split;">
            min_samples_split: int or float, default=2<br><br>The minimum number of samples required to split an internal node:<br><br>- If int, values must be in the range `[2, inf)`.<br>- If float, values must be in the range `(0.0, 1.0]` and `min_samples_split`<br>  will be `ceil(min_samples_split * n_samples)`.<br><br>.. versionchanged:: 0.18<br>   Added float values for fractions.</span>
        </a>
    </td>
            <td class="value">2</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_samples_leaf',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-min_samples_leaf;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=min_samples_leaf,-int%20or%20float%2C%20default%3D1">
            min_samples_leaf
            <span class="param-doc-description"
            style="position-anchor: --doc-link-min_samples_leaf;">
            min_samples_leaf: int or float, default=1<br><br>The minimum number of samples required to be at a leaf node.<br>A split point at any depth will only be considered if it leaves at<br>least ``min_samples_leaf`` training samples in each of the left and<br>right branches.  This may have the effect of smoothing the model,<br>especially in regression.<br><br>- If int, values must be in the range `[1, inf)`.<br>- If float, values must be in the range `(0.0, 1.0)` and `min_samples_leaf`<br>  will be `ceil(min_samples_leaf * n_samples)`.<br><br>.. versionchanged:: 0.18<br>   Added float values for fractions.</span>
        </a>
    </td>
            <td class="value">1</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_weight_fraction_leaf',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-min_weight_fraction_leaf;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=min_weight_fraction_leaf,-float%2C%20default%3D0.0">
            min_weight_fraction_leaf
            <span class="param-doc-description"
            style="position-anchor: --doc-link-min_weight_fraction_leaf;">
            min_weight_fraction_leaf: float, default=0.0<br><br>The minimum weighted fraction of the sum total of weights (of all<br>the input samples) required to be at a leaf node. Samples have<br>equal weight when sample_weight is not provided.<br>Values must be in the range `[0.0, 0.5]`.</span>
        </a>
    </td>
            <td class="value">0.0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_depth',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-max_depth;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=max_depth,-int%20or%20None%2C%20default%3D3">
            max_depth
            <span class="param-doc-description"
            style="position-anchor: --doc-link-max_depth;">
            max_depth: int or None, default=3<br><br>Maximum depth of the individual regression estimators. The maximum<br>depth limits the number of nodes in the tree. Tune this parameter<br>for best performance; the best value depends on the interaction<br>of the input variables. If None, then nodes are expanded until<br>all leaves are pure or until all leaves contain less than<br>min_samples_split samples.<br>If int, values must be in the range `[1, inf)`.</span>
        </a>
    </td>
            <td class="value">3</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_impurity_decrease',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-min_impurity_decrease;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=min_impurity_decrease,-float%2C%20default%3D0.0">
            min_impurity_decrease
            <span class="param-doc-description"
            style="position-anchor: --doc-link-min_impurity_decrease;">
            min_impurity_decrease: float, default=0.0<br><br>A node will be split if this split induces a decrease of the impurity<br>greater than or equal to this value.<br>Values must be in the range `[0.0, inf)`.<br><br>The weighted impurity decrease equation is the following::<br><br>    N_t / N * (impurity - N_t_R / N_t * right_impurity<br>                        - N_t_L / N_t * left_impurity)<br><br>where ``N`` is the total number of samples, ``N_t`` is the number of<br>samples at the current node, ``N_t_L`` is the number of samples in the<br>left child, and ``N_t_R`` is the number of samples in the right child.<br><br>``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,<br>if ``sample_weight`` is passed.<br><br>.. versionadded:: 0.19</span>
        </a>
    </td>
            <td class="value">0.0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('init',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-init;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=init,-estimator%20or%20%27zero%27%2C%20default%3DNone">
            init
            <span class="param-doc-description"
            style="position-anchor: --doc-link-init;">
            init: estimator or &#x27;zero&#x27;, default=None<br><br>An estimator object that is used to compute the initial predictions.<br>``init`` has to provide :term:`fit` and :term:`predict`. If &#x27;zero&#x27;, the<br>initial raw predictions are set to zero. By default a<br>``DummyEstimator`` is used, predicting either the average target value<br>(for loss=&#x27;squared_error&#x27;), or a quantile for the other losses.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_features',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-max_features;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=max_features,-%7B%27sqrt%27%2C%20%27log2%27%7D%2C%20int%20or%20float%2C%20default%3DNone">
            max_features
            <span class="param-doc-description"
            style="position-anchor: --doc-link-max_features;">
            max_features: {&#x27;sqrt&#x27;, &#x27;log2&#x27;}, int or float, default=None<br><br>The number of features to consider when looking for the best split:<br><br>- If int, values must be in the range `[1, inf)`.<br>- If float, values must be in the range `(0.0, 1.0]` and the features<br>  considered at each split will be `max(1, int(max_features * n_features_in_))`.<br>- If &quot;sqrt&quot;, then `max_features=sqrt(n_features)`.<br>- If &quot;log2&quot;, then `max_features=log2(n_features)`.<br>- If None, then `max_features=n_features`.<br><br>Choosing `max_features &lt; n_features` leads to a reduction of variance<br>and an increase in bias.<br><br>Note: the search for a split does not stop until at least one<br>valid partition of the node samples is found, even if it requires to<br>effectively inspect more than ``max_features`` features.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('alpha',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-alpha;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=alpha,-float%2C%20default%3D0.9">
            alpha
            <span class="param-doc-description"
            style="position-anchor: --doc-link-alpha;">
            alpha: float, default=0.9<br><br>The alpha-quantile of the huber loss function and the quantile<br>loss function. Only if ``loss=&#x27;huber&#x27;`` or ``loss=&#x27;quantile&#x27;``.<br>Values must be in the range `(0.0, 1.0)`.</span>
        </a>
    </td>
            <td class="value">0.9</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('verbose',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-verbose;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=verbose,-int%2C%20default%3D0">
            verbose
            <span class="param-doc-description"
            style="position-anchor: --doc-link-verbose;">
            verbose: int, default=0<br><br>Enable verbose output. If 1 then it prints progress and performance<br>once in a while (the more trees the lower the frequency). If greater<br>than 1 then it prints progress and performance for every tree.<br>Values must be in the range `[0, inf)`.</span>
        </a>
    </td>
            <td class="value">0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_leaf_nodes',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-max_leaf_nodes;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=max_leaf_nodes,-int%2C%20default%3DNone">
            max_leaf_nodes
            <span class="param-doc-description"
            style="position-anchor: --doc-link-max_leaf_nodes;">
            max_leaf_nodes: int, default=None<br><br>Grow trees with ``max_leaf_nodes`` in best-first fashion.<br>Best nodes are defined as relative reduction in impurity.<br>Values must be in the range `[2, inf)`.<br>If None, then unlimited number of leaf nodes.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('warm_start',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-warm_start;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=warm_start,-bool%2C%20default%3DFalse">
            warm_start
            <span class="param-doc-description"
            style="position-anchor: --doc-link-warm_start;">
            warm_start: bool, default=False<br><br>When set to ``True``, reuse the solution of the previous call to fit<br>and add more estimators to the ensemble, otherwise, just erase the<br>previous solution. See :term:`the Glossary &lt;warm_start&gt;`.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('validation_fraction',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-validation_fraction;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=validation_fraction,-float%2C%20default%3D0.1">
            validation_fraction
            <span class="param-doc-description"
            style="position-anchor: --doc-link-validation_fraction;">
            validation_fraction: float, default=0.1<br><br>The proportion of training data to set aside as validation set for<br>early stopping. Values must be in the range `(0.0, 1.0)`.<br>Only used if ``n_iter_no_change`` is set to an integer.<br><br>.. versionadded:: 0.20</span>
        </a>
    </td>
            <td class="value">0.1</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('n_iter_no_change',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-n_iter_no_change;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=n_iter_no_change,-int%2C%20default%3DNone">
            n_iter_no_change
            <span class="param-doc-description"
            style="position-anchor: --doc-link-n_iter_no_change;">
            n_iter_no_change: int, default=None<br><br>``n_iter_no_change`` is used to decide if early stopping will be used<br>to terminate training when validation score is not improving. By<br>default it is set to None to disable early stopping. If set to a<br>number, it will set aside ``validation_fraction`` size of the training<br>data as validation and terminate training when validation score is not<br>improving in all of the previous ``n_iter_no_change`` numbers of<br>iterations.<br>Values must be in the range `[1, inf)`.<br>See<br>:ref:`sphx_glr_auto_examples_ensemble_plot_gradient_boosting_early_stopping.py`.<br><br>.. versionadded:: 0.20</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('tol',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-tol;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=tol,-float%2C%20default%3D1e-4">
            tol
            <span class="param-doc-description"
            style="position-anchor: --doc-link-tol;">
            tol: float, default=1e-4<br><br>Tolerance for the early stopping. When the loss is not improving<br>by at least tol for ``n_iter_no_change`` iterations (if set to a<br>number), the training stops.<br>Values must be in the range `[0.0, inf)`.<br><br>.. versionadded:: 0.20</span>
        </a>
    </td>
            <td class="value">0.0001</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('ccp_alpha',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-ccp_alpha;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=ccp_alpha,-non-negative%20float%2C%20default%3D0.0">
            ccp_alpha
            <span class="param-doc-description"
            style="position-anchor: --doc-link-ccp_alpha;">
            ccp_alpha: non-negative float, default=0.0<br><br>Complexity parameter used for Minimal Cost-Complexity Pruning. The<br>subtree with the largest cost complexity that is smaller than<br>``ccp_alpha`` will be chosen. By default, no pruning is performed.<br>Values must be in the range `[0.0, inf)`.<br>See :ref:`minimal_cost_complexity_pruning` for details. See<br>:ref:`sphx_glr_auto_examples_tree_plot_cost_complexity_pruning.py`<br>for an example of such pruning.<br><br>.. versionadded:: 0.22</span>
        </a>
    </td>
            <td class="value">0.0</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>

        <div class="estimator-table">
            <details>
                <summary>Fitted attributes</summary>
                <table class="parameters-table">
                    <tbody>
                        <tr>
                        <th>Name</th>
                        <th>Type</th>
                        <th>Value</th>
                        </tr>

       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-estimators_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=estimators_,-int">
            estimators_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-estimators_;">
            estimators_: ndarray of DecisionTreeRegressor of shape (n_estimators, 1)<br><br>The collection of fitted sub-estimators.</span>
        </a>
    </td>
           <td class="fitted-att-type">ndarray[object](200, 1)</td>
           <td>[[DecisionTreeRegressor(max_depth=3,
                        random_state=RandomState(MT19937) at 0x187EA4BE440)],
 [DecisionTreeRegressor(max_depth=3,
                        random_state=RandomState(MT19937) at 0x187EA4BE440)],
 [DecisionTreeRegressor(max_depth=3,
                        random_state=RandomState(MT19937) at 0x187EA4BE440)],
 ...,
 [DecisionTreeRegressor(max_depth=3,
                        random_state=RandomState(MT19937) at 0x187EA4BE440)],
 [DecisionTreeRegressor(max_depth=3,
                        random_state=RandomState(MT19937) at 0x187EA4BE440)],
 [DecisionTreeRegressor(max_depth=3,
                        random_state=RandomState(MT19937) at 0x187EA4BE440)]]</td>


       </tr>


       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-feature_importances_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=feature_importances_,-ndarray%20of%20shape%20%28n_features%2C%29">
            feature_importances_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-feature_importances_;">
            feature_importances_: ndarray of shape (n_features,)<br><br>The impurity-based feature importances.<br>The higher, the more important the feature.<br>The importance of a feature is computed as the (normalized)<br>total reduction of the MSE brought by that feature.  It is also<br>known as the Gini importance.<br><br>Warning: impurity-based feature importances can be misleading for<br>high cardinality features (many unique values). See<br>:func:`sklearn.inspection.permutation_importance` as an alternative.</span>
        </a>
    </td>
           <td class="fitted-att-type">ndarray[float64](16,)</td>
           <td>[0.,0.,0.,...,0.,0.,0.]</td>


       </tr>


       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-feature_names_in_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=feature_names_in_,-ndarray%20of%20shape%20%28n_features_in_%2C%29">
            feature_names_in_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-feature_names_in_;">
            feature_names_in_: ndarray of shape (`n_features_in_`,)<br><br>Names of features seen during :term:`fit`. Defined only when `X`<br>has feature names that are all strings.<br><br>.. versionadded:: 1.0</span>
        </a>
    </td>
           <td class="fitted-att-type">ndarray[object](16,)</td>
           <td>[&#x27;Children apprehended and placed in CBP custody*&#x27;,
 &#x27;Children in CBP custody&#x27;,&#x27;Children transferred out of CBP custody&#x27;,...,
 &#x27;Month&#x27;,&#x27;Quarter&#x27;,&#x27;Year&#x27;]</td>


       </tr>


       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-init_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=init_,-estimator">
            init_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-init_;">
            init_: estimator<br><br>The estimator that provides the initial predictions. Set via the ``init``<br>argument.</span>
        </a>
    </td>
           <td class="fitted-att-type">DummyRegressor</td>
           <td>DummyRegressor()</td>


       </tr>


       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-max_features_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=max_features_,-int">
            max_features_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-max_features_;">
            max_features_: int<br><br>The inferred value of max_features.</span>
        </a>
    </td>
           <td class="fitted-att-type">int</td>
           <td>16</td>


       </tr>


       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-n_estimators_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=n_estimators_,-int">
            n_estimators_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-n_estimators_;">
            n_estimators_: int<br><br>The number of estimators as selected by early stopping (if<br>``n_iter_no_change`` is specified). Otherwise it is set to<br>``n_estimators``.</span>
        </a>
    </td>
           <td class="fitted-att-type">int</td>
           <td>200</td>


       </tr>


       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-n_features_in_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=n_features_in_,-int">
            n_features_in_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-n_features_in_;">
            n_features_in_: int<br><br>Number of features seen during :term:`fit`.<br><br>.. versionadded:: 0.24</span>
        </a>
    </td>
           <td class="fitted-att-type">int</td>
           <td>16</td>


       </tr>


       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-n_trees_per_iteration_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=n_trees_per_iteration_,-int">
            n_trees_per_iteration_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-n_trees_per_iteration_;">
            n_trees_per_iteration_: int<br><br>The number of trees that are built at each iteration. For regressors, this is<br>always 1.<br><br>.. versionadded:: 1.4.0</span>
        </a>
    </td>
           <td class="fitted-att-type">int</td>
           <td>1</td>


       </tr>


       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-train_score_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=train_score_,-ndarray%20of%20shape%20%28n_estimators%2C%29">
            train_score_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-train_score_;">
            train_score_: ndarray of shape (n_estimators,)<br><br>The i-th score ``train_score_[i]`` is the loss of the<br>model at iteration ``i`` on the in-bag sample.<br>If ``subsample == 1`` this is the loss on the training data.</span>
        </a>
    </td>
           <td class="fitted-att-type">ndarray[float64](200,)</td>
           <td>[5113476.59,4627116.99,4187821.2 ,...,   2667.49,   2651.62,   2637.14]</td>


       </tr>

                    </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div><script>/*  Authors: The scikit-learn developers
 SPDX-License-Identifier: BSD-3-Clause
*/

function copyToClipboard(text, element) {
    // Get the parameter prefix from the closest toggleable content
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const fullParamName = paramPrefix ? `${paramPrefix}${text}` : text;

    const originalStyle = element.style;
    const computedStyle = window.getComputedStyle(element);
    const originalWidth = computedStyle.width;
    const originalHTML = element.innerHTML.replace('Copied!', '');

    navigator.clipboard.writeText(fullParamName)
        .then(() => {
            element.style.width = originalWidth;
            element.style.color = 'green';
            element.innerHTML = "Copied!";

            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        })
        .catch(err => {
            console.error('Failed to copy:', err);
            element.style.color = 'red';
            element.innerHTML = "Failed!";
            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        });
    return false;
}

document.querySelectorAll('.copy-paste-icon').forEach(function(element) {
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';

    const parent = element.parentElement;
    if (!parent || !parent.nextElementSibling) {
        console.warn('Expected copy-paste icon is missing from the DOM structure');
        return;
    }

    const paramName = element.parentElement.nextElementSibling
        .textContent.trim().split(' ')[0];
    const fullParamName = paramPrefix ? `${paramPrefix}${paramName}` : paramName;

    element.setAttribute('title', fullParamName);
});

/**
 * Copy the list of feature names formatted as a Python list.
 *
 * @param {HTMLElement} element - The copy button inside a `.features` block; its siblings
 *   contain a `details` element and a table containing feature named.
 * @returns {boolean} Always returns `false` so callers can prevent the default click behavior.
 */
function copyFeatureNamesToClipboard(element) {
    var detailsElem = element.closest('.features').querySelector('details');
    var wasOpen = detailsElem.open;
    detailsElem.open = true;
    var content = element.closest('.features').querySelector('tbody')
                  .innerText.trim();
    if (!wasOpen) detailsElem.open = false;
    const rows = content.split('\n').map(row => `    "${row}"`);
    const formattedText = `[\n${rows.join(',\n')},\n]`;
    const originalHTML = element.innerHTML.replace('âœ”', '');
    const originalStyle = element.style;
    const copyMark = document.createElement('span');
    copyMark.innerHTML = 'âœ”';
    copyMark.style.color = 'blue';
    copyMark.style.fontSize = '1em';

    navigator.clipboard.writeText(formattedText)
        .then(() => {
            element.style.display = 'none';
            element.parentElement.appendChild(copyMark);

            setTimeout(() => {
                copyMark.remove();
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 1000);
        })
        .catch(err => {
            console.error('Failed to copy:', err);
            element.style.color = 'orange';
            element.innerHTML = "Failed!";
            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 1000);
        });
    return false;
}
/**
 * Adapted from Skrub
 * https://github.com/skrub-data/skrub/blob/403466d1d5d4dc76a7ef569b3f8228db59a31dc3/skrub/_reporting/_data/templates/report.js#L789
 * @returns "light" or "dark"
 */
function detectTheme(element) {
    const body = document.querySelector('body');

    // Check VSCode theme
    const themeKindAttr = body.getAttribute('data-vscode-theme-kind');
    const themeNameAttr = body.getAttribute('data-vscode-theme-name');

    if (themeKindAttr && themeNameAttr) {
        const themeKind = themeKindAttr.toLowerCase();
        const themeName = themeNameAttr.toLowerCase();

        if (themeKind.includes("dark") || themeName.includes("dark")) {
            return "dark";
        }
        if (themeKind.includes("light") || themeName.includes("light")) {
            return "light";
        }
    }

    // Check Jupyter theme
    if (body.getAttribute('data-jp-theme-light') === 'false') {
        return 'dark';
    } else if (body.getAttribute('data-jp-theme-light') === 'true') {
        return 'light';
    }

    // Guess based on a parent element's color
    const color = window.getComputedStyle(element.parentNode, null).getPropertyValue('color');
    const match = color.match(/^rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*$/i);
    if (match) {
        const [r, g, b] = [
            parseFloat(match[1]),
            parseFloat(match[2]),
            parseFloat(match[3])
        ];

        // https://en.wikipedia.org/wiki/HSL_and_HSV#Lightness
        const luma = 0.299 * r + 0.587 * g + 0.114 * b;

        if (luma > 180) {
            // If the text is very bright we have a dark theme
            return 'dark';
        }
        if (luma < 75) {
            // If the text is very dark we have a light theme
            return 'light';
        }
        // Otherwise fall back to the next heuristic.
    }

    // Fallback to system preference
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}


function forceTheme(elementId) {
    const estimatorElement = document.querySelector(`#${elementId}`);
    if (estimatorElement === null) {
        console.error(`Element with id ${elementId} not found.`);
    } else {
        const theme = detectTheme(estimatorElement);
        estimatorElement.classList.add(theme);
    }
}

forceTheme('sk-container-id-3');</script></body>




```python
gb_predictions = gb_model.predict(X_test)
```


```python
gb_mae = mean_absolute_error(y_test, gb_predictions)

gb_rmse = np.sqrt(mean_squared_error(y_test, gb_predictions))

gb_mape = mean_absolute_percentage_error(
    y_test,
    gb_predictions
) * 100

print("Gradient Boosting Results")
print("-"*35)

print(f"MAE  : {gb_mae:.2f}")
print(f"RMSE : {gb_rmse:.2f}")
print(f"MAPE : {gb_mape:.2f}%")
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[144], line 5
          1 gb_mae = mean_absolute_error(y_test, gb_predictions)
          3 gb_rmse = np.sqrt(mean_squared_error(y_test, gb_predictions))
    ----> 5 gb_mape = mean_absolute_percentage_error(
          6     y_test,
          7     gb_predictions
          8 ) * 100
         10 print("Gradient Boosting Results")
         11 print("-"*35)
    

    NameError: name 'mean_absolute_percentage_error' is not defined



```python


plt.figure(figsize=(15,6))

plt.plot(y_test.index,
         y_test,
         label="Actual",
         linewidth=2)

plt.plot(y_test.index,
         gb_predictions,
         label="Gradient Boosting",
         linewidth=2)

plt.title("Gradient Boosting Forecast")

plt.xlabel("Date")

plt.ylabel("Children in HHS Care")

plt.legend()

plt.grid(True)

plt.show()
```


    
![png](output_107_0.png)
    



```python
gb_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": gb_model.feature_importances_

})

gb_importance = gb_importance.sort_values(
    by="Importance",
    ascending=False
)

gb_importance
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature</th>
      <th>Importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>Lag_1</td>
      <td>0.795275</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Rolling_Mean_7</td>
      <td>0.138164</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Lag_7</td>
      <td>0.024126</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Rolling_Mean_14</td>
      <td>0.022086</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Lag_14</td>
      <td>0.012214</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Month</td>
      <td>0.003903</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Children in CBP custody</td>
      <td>0.000984</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Day_of_Week</td>
      <td>0.000829</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Children apprehended and placed in CBP custody*</td>
      <td>0.000527</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Rolling_STD_7</td>
      <td>0.000478</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Year</td>
      <td>0.000448</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Net_Pressure</td>
      <td>0.000332</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Children transferred out of CBP custody</td>
      <td>0.000302</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Quarter</td>
      <td>0.000182</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Rolling_STD_14</td>
      <td>0.000109</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Children discharged from HHS Care</td>
      <td>0.000043</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(10,6))

plt.barh(
    gb_importance["Feature"],
    gb_importance["Importance"]
)

plt.gca().invert_yaxis()

plt.title("Gradient Boosting Feature Importance")

plt.xlabel("Importance")

plt.show()
```


    
![png](output_109_0.png)
    



```python
from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gb_model.fit(X_train, y_train)
```




<style>.sk-global {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;
}

.sk-global.light {
  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: black;
  --sklearn-color-background: white;
  --sklearn-color-border-box: black;
  --sklearn-color-icon: #696969;
}

.sk-global.dark {
  --sklearn-color-text-on-default-background: white;
  --sklearn-color-background: #111;
  --sklearn-color-border-box: white;
  --sklearn-color-icon: #878787;
}

.sk-global {
  color: var(--sklearn-color-text);
}

.sk-global pre {
  padding: 0;
}

.sk-global input.sk-hidden--visually {
  border: 0;
  clip-path: inset(100%);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

.sk-global div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

.sk-global div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

.sk-global div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

.sk-global div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

.sk-global div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

.sk-global div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

.sk-global div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

.sk-global div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

.sk-global div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

.sk-global div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

.sk-global div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
.sk-global label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: center;
  justify-content: center;
  gap: 0.5em;
}

.sk-global label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

.sk-global label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

.sk-global label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

.sk-global div.sk-toggleable__content {
  display: none;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

.sk-global div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

.sk-global div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

.sk-global div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

.sk-global input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  display: block;
  width: 100%;
  overflow: visible;
}

.sk-global input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

.sk-global div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

.sk-global div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
.sk-global div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

.sk-global div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

.sk-global div.sk-label label.sk-toggleable__label,
.sk-global div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
.sk-global div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
.sk-global div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

.sk-global div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  line-height: 1.2em;
}

.sk-global div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
.sk-global div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

.sk-global div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
.sk-global div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

.sk-global div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-unfitted-level-0);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-3) 1pt solid;
  color: var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3) 1pt solid;
  color: var(--sklearn-color-fitted-level-3);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  border: var(--sklearn-color-fitted-level-0) 1pt solid;
  color: var(--sklearn-color-unfitted-level-0);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  border: var(--sklearn-color-fitted-level-0) 1pt solid;
  color: var(--sklearn-color-fitted-level-0);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

.sk-global a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-unfitted-level-0);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

.sk-global a.estimator_doc_link.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
.sk-global a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

.sk-global a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}

.sk-top-container.sk-global {
  /* pydata-sphinx-theme hides overflow, so scrolling is disabled.
   We need to set it to !important and add tabindex="0" in the HTML
   to allow keyboard-only users to navigate the display. */
  overflow-x: scroll !important;
  max-width: 100%;
}

.estimator-table {
    font-family: monospace;
}

.estimator-table summary {
    padding: .5rem;
    cursor: pointer;
}

.estimator-table summary::marker {
    font-size: 0.7rem;
}

.estimator-table details[open] {
    padding-left: 0.1rem;
    padding-right: 0.1rem;
    padding-bottom: 0.3rem;
}

.estimator-table .parameters-table {
    margin-left: auto !important;
    margin-right: auto !important;
    margin-top: 0;
}

.estimator-table .parameters-table tr:nth-child(odd) {
    background-color: #fff;
}

.estimator-table .parameters-table tr:nth-child(even) {
    background-color: #f6f6f6;
}

.estimator-table .parameters-table tr:hover td {
    background-color: #e0e0e0;
}

.estimator-table table :is(td, th) {
    border: 1px solid rgba(106, 105, 104, 0.232);
}

/*
    `table td`is set in notebook with right text-align.
    We need to overwrite it.
*/
.estimator-table table td.param {
    text-align: left;
    position: relative;
    padding: 0;
}

.user-set td {
    color:rgb(255, 94, 0);
    text-align: left !important;
}

.user-set td.value {
    color:rgb(255, 94, 0);
    background-color: transparent;
}

.default td, .estimator-table th {
    color: black;
    text-align: left !important;
}

.user-set td i,
.default td i {
    color: black;
}

td.fitted-att-type {
    white-space: preserve nowrap;
}

/*
    Styles for parameter documentation links
    We need styling for visited so jupyter doesn't overwrite it
*/
a.param-doc-link,
a.param-doc-link:link,
a.param-doc-link:visited {
    text-decoration: underline dashed;
    text-underline-offset: .3em;
    color: inherit;
    display: block;
    padding: .5em;
}

@supports(anchor-name: --doc-link) {
    a.param-doc-link,
    a.param-doc-link:link,
    a.param-doc-link:visited {
    anchor-name: --doc-link;
    }
}

/* "hack" to make the entire area of the cell containing the link clickable */
a.param-doc-link::before {
    position: absolute;
    content: "";
    inset: 0;
}

.param-doc-description {
    display: none;
    position: absolute;
    z-index: 9999;
    left: 0;
    padding: .5ex;
    margin-left: 1.5em;
    color: var(--sklearn-color-text);
    box-shadow: .3em .3em .4em #999;
    width: max-content;
    text-align: left;
    max-height: 10em;
    overflow-y: auto;

    /* unfitted */
    background: var(--sklearn-color-unfitted-level-0);
    border: thin solid var(--sklearn-color-unfitted-level-3);
}

@supports(position-area: center right) {
    .param-doc-description {
    position-area: center right;
    position: fixed;
    margin-left: 0;
    }
}

/* Fitted state for parameter tooltips */
.fitted .param-doc-description {
    /* fitted */
    background: var(--sklearn-color-fitted-level-0);
    border: thin solid var(--sklearn-color-fitted-level-3);
}

.param-doc-link:hover .param-doc-description {
    display: block;
}

.copy-paste-icon {
    background-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NDggNTEyIj48IS0tIUZvbnQgQXdlc29tZSBGcmVlIDYuNy4yIGJ5IEBmb250YXdlc29tZSAtIGh0dHBzOi8vZm9udGF3ZXNvbWUuY29tIExpY2Vuc2UgLSBodHRwczovL2ZvbnRhd2Vzb21lLmNvbS9saWNlbnNlL2ZyZWUgQ29weXJpZ2h0IDIwMjUgRm9udGljb25zLCBJbmMuLS0+PHBhdGggZD0iTTIwOCAwTDMzMi4xIDBjMTIuNyAwIDI0LjkgNS4xIDMzLjkgMTQuMWw2Ny45IDY3LjljOSA5IDE0LjEgMjEuMiAxNC4xIDMzLjlMNDQ4IDMzNmMwIDI2LjUtMjEuNSA0OC00OCA0OGwtMTkyIDBjLTI2LjUgMC00OC0yMS41LTQ4LTQ4bDAtMjg4YzAtMjYuNSAyMS41LTQ4IDQ4LTQ4ek00OCAxMjhsODAgMCAwIDY0LTY0IDAgMCAyNTYgMTkyIDAgMC0zMiA2NCAwIDAgNDhjMCAyNi41LTIxLjUgNDgtNDggNDhMNDggNTEyYy0yNi41IDAtNDgtMjEuNS00OC00OEwwIDE3NmMwLTI2LjUgMjEuNS00OCA0OC00OHoiLz48L3N2Zz4=);
    background-repeat: no-repeat;
    background-size: 14px 14px;
    background-position: 0;
    display: inline-block;
    width: 14px;
    height: 14px;
    cursor: pointer;
}

.features {
  font-family: monospace;
  cursor: pointer;
  background-color: var(--sklearn-color-unfitted-level-0);
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: .20em;
  margin-bottom: 0.5em;
  font-size: inherit; /* Needed for jupyter */
}

.features.fitted {
  background-color: var(--sklearn-color-fitted-level-0);
}

.features summary {
  cursor: pointer;
  display: flex;
  margin-bottom: 0;
  text-align: center;
  align-items: center;
  justify-content: center;
  gap: 0.5em;
  padding: .25em;
}

.features details[open] > summary {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
  border-radius: .20em 0 0 0;
}

.features.fitted details[open] > summary {
  background-color: var(--sklearn-color-fitted-level-2);
  border-radius: .20em 0 0 0;
}

.features details > summary .arrow::before {
  content: "▸";
  color: grey;
}

.features details[open] > summary .arrow::before {
  content: "▾";
}

.features details:hover > summary {
  margin: 0;
  background-color: var(--sklearn-color-unfitted-level-2);
}

.features.fitted details:hover > summary {
  margin: 0;
  background-color: var(--sklearn-color-fitted-level-2);
}

.features .features-container {
  max-width: 15em;
  max-height: 10em;
  overflow: auto;
  scrollbar-width: thin;
  padding: .25em 0.1rem;
  background-color: var(--sklearn-color-unfitted-level-0);
  border-radius: 0 0 .5em .5em;
}

.features.fitted .features-container {
  background-color: var(--sklearn-color-fitted-level-0);
}

.features .image-container {
  block-size: 1em;
  inline-size: 1em;
  padding: 0;
  margin: 0%;
  display: flex;
  justify-content: center;
  align-items: center;
}

.features .copy-paste-icon {
  background-size: 1em 1em;
  width: 1em;
  height: 1em;
  filter: grayscale(100%) opacity(60%);
}

.features .features-container table {
  width: 100%;
  margin: 0.01em;
}

.features .features-container table tr:nth-child(odd) {
  background-color: #fff;
}

.features .features-container table tr:nth-child(even) {
  background-color: #f6f6f6;
}

.features .features-container table tr:hover {
  background-color: #e0e0e0;
}

.features .features-container table {
  table-layout: inherit;
}

.features .features-container table td {
  text-align: left;
  padding: 0 0.5em;
  border: 1px solid rgba(106, 105, 104, 0.232);
  white-space: nowrap;
  color: var(--sklearn-color-text);
}

.total_features {
  display: flex;
  justify-content: center;
  margin-top: 0.5em;
}
</style><body><div id="sk-container-id-4" tabindex="0" class="sk-top-container sk-global"><div class="sk-text-repr-fallback"><pre>GradientBoostingRegressor(learning_rate=0.05, n_estimators=200, random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually sk-global" id="sk-estimator-id-4" type="checkbox" checked><label for="sk-estimator-id-4" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>GradientBoostingRegressor</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html">?<span>Documentation for GradientBoostingRegressor</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted" data-param-prefix="">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('learning_rate',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-learning_rate;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=learning_rate,-float%2C%20default%3D0.1">
            learning_rate
            <span class="param-doc-description"
            style="position-anchor: --doc-link-learning_rate;">
            learning_rate: float, default=0.1<br><br>Learning rate shrinks the contribution of each tree by `learning_rate`.<br>There is a trade-off between learning_rate and n_estimators.<br>Values must be in the range `[0.0, inf)`.</span>
        </a>
    </td>
            <td class="value">0.05</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('n_estimators',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-n_estimators;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=n_estimators,-int%2C%20default%3D100">
            n_estimators
            <span class="param-doc-description"
            style="position-anchor: --doc-link-n_estimators;">
            n_estimators: int, default=100<br><br>The number of boosting stages to perform. Gradient boosting<br>is fairly robust to over-fitting so a large number usually<br>results in better performance.<br>Values must be in the range `[1, inf)`.</span>
        </a>
    </td>
            <td class="value">200</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('random_state',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-random_state;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=random_state,-int%2C%20RandomState%20instance%20or%20None%2C%20default%3DNone">
            random_state
            <span class="param-doc-description"
            style="position-anchor: --doc-link-random_state;">
            random_state: int, RandomState instance or None, default=None<br><br>Controls the random seed given to each Tree estimator at each<br>boosting iteration.<br>In addition, it controls the random permutation of the features at<br>each split (see Notes for more details).<br>It also controls the random splitting of the training data to obtain a<br>validation set if `n_iter_no_change` is not None.<br>Pass an int for reproducible output across multiple function calls.<br>See :term:`Glossary &lt;random_state&gt;`.</span>
        </a>
    </td>
            <td class="value">42</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('loss',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-loss;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=loss,-%7B%27squared_error%27%2C%20%27absolute_error%27%2C%20%27huber%27%2C%20%27quantile%27%7D%2C%20%20%20%20%20%20%20%20%20%20%20%20%20default%3D%27squared_error%27">
            loss
            <span class="param-doc-description"
            style="position-anchor: --doc-link-loss;">
            loss: {&#x27;squared_error&#x27;, &#x27;absolute_error&#x27;, &#x27;huber&#x27;, &#x27;quantile&#x27;},             default=&#x27;squared_error&#x27;<br><br>Loss function to be optimized. &#x27;squared_error&#x27; refers to the squared<br>error for regression. &#x27;absolute_error&#x27; refers to the absolute error of<br>regression and is a robust loss function. &#x27;huber&#x27; is a<br>combination of the two. &#x27;quantile&#x27; allows quantile regression (use<br>`alpha` to specify the quantile).<br>See<br>:ref:`sphx_glr_auto_examples_ensemble_plot_gradient_boosting_quantile.py`<br>for an example that demonstrates quantile regression for creating<br>prediction intervals with `loss=&#x27;quantile&#x27;`.</span>
        </a>
    </td>
            <td class="value">&#x27;squared_error&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('subsample',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-subsample;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=subsample,-float%2C%20default%3D1.0">
            subsample
            <span class="param-doc-description"
            style="position-anchor: --doc-link-subsample;">
            subsample: float, default=1.0<br><br>The fraction of samples to be used for fitting the individual base<br>learners. If smaller than 1.0 this results in Stochastic Gradient<br>Boosting. `subsample` interacts with the parameter `n_estimators`.<br>Choosing `subsample &lt; 1.0` leads to a reduction of variance<br>and an increase in bias.<br>Values must be in the range `(0.0, 1.0]`.</span>
        </a>
    </td>
            <td class="value">1.0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('criterion',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-criterion;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=criterion,-%7B%27friedman_mse%27%2C%20%27squared_error%27%7D%2C%20default%3D%27friedman_mse%27">
            criterion
            <span class="param-doc-description"
            style="position-anchor: --doc-link-criterion;">
            criterion: {&#x27;friedman_mse&#x27;, &#x27;squared_error&#x27;}, default=&#x27;friedman_mse&#x27;<br><br>This parameter has no effect.<br><br>.. versionadded:: 0.18<br><br>.. deprecated:: 1.9<br>   `criterion` is deprecated and will be removed in 1.11.</span>
        </a>
    </td>
            <td class="value">&#x27;deprecated&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_samples_split',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-min_samples_split;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=min_samples_split,-int%20or%20float%2C%20default%3D2">
            min_samples_split
            <span class="param-doc-description"
            style="position-anchor: --doc-link-min_samples_split;">
            min_samples_split: int or float, default=2<br><br>The minimum number of samples required to split an internal node:<br><br>- If int, values must be in the range `[2, inf)`.<br>- If float, values must be in the range `(0.0, 1.0]` and `min_samples_split`<br>  will be `ceil(min_samples_split * n_samples)`.<br><br>.. versionchanged:: 0.18<br>   Added float values for fractions.</span>
        </a>
    </td>
            <td class="value">2</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_samples_leaf',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-min_samples_leaf;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=min_samples_leaf,-int%20or%20float%2C%20default%3D1">
            min_samples_leaf
            <span class="param-doc-description"
            style="position-anchor: --doc-link-min_samples_leaf;">
            min_samples_leaf: int or float, default=1<br><br>The minimum number of samples required to be at a leaf node.<br>A split point at any depth will only be considered if it leaves at<br>least ``min_samples_leaf`` training samples in each of the left and<br>right branches.  This may have the effect of smoothing the model,<br>especially in regression.<br><br>- If int, values must be in the range `[1, inf)`.<br>- If float, values must be in the range `(0.0, 1.0)` and `min_samples_leaf`<br>  will be `ceil(min_samples_leaf * n_samples)`.<br><br>.. versionchanged:: 0.18<br>   Added float values for fractions.</span>
        </a>
    </td>
            <td class="value">1</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_weight_fraction_leaf',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-min_weight_fraction_leaf;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=min_weight_fraction_leaf,-float%2C%20default%3D0.0">
            min_weight_fraction_leaf
            <span class="param-doc-description"
            style="position-anchor: --doc-link-min_weight_fraction_leaf;">
            min_weight_fraction_leaf: float, default=0.0<br><br>The minimum weighted fraction of the sum total of weights (of all<br>the input samples) required to be at a leaf node. Samples have<br>equal weight when sample_weight is not provided.<br>Values must be in the range `[0.0, 0.5]`.</span>
        </a>
    </td>
            <td class="value">0.0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_depth',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-max_depth;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=max_depth,-int%20or%20None%2C%20default%3D3">
            max_depth
            <span class="param-doc-description"
            style="position-anchor: --doc-link-max_depth;">
            max_depth: int or None, default=3<br><br>Maximum depth of the individual regression estimators. The maximum<br>depth limits the number of nodes in the tree. Tune this parameter<br>for best performance; the best value depends on the interaction<br>of the input variables. If None, then nodes are expanded until<br>all leaves are pure or until all leaves contain less than<br>min_samples_split samples.<br>If int, values must be in the range `[1, inf)`.</span>
        </a>
    </td>
            <td class="value">3</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_impurity_decrease',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-min_impurity_decrease;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=min_impurity_decrease,-float%2C%20default%3D0.0">
            min_impurity_decrease
            <span class="param-doc-description"
            style="position-anchor: --doc-link-min_impurity_decrease;">
            min_impurity_decrease: float, default=0.0<br><br>A node will be split if this split induces a decrease of the impurity<br>greater than or equal to this value.<br>Values must be in the range `[0.0, inf)`.<br><br>The weighted impurity decrease equation is the following::<br><br>    N_t / N * (impurity - N_t_R / N_t * right_impurity<br>                        - N_t_L / N_t * left_impurity)<br><br>where ``N`` is the total number of samples, ``N_t`` is the number of<br>samples at the current node, ``N_t_L`` is the number of samples in the<br>left child, and ``N_t_R`` is the number of samples in the right child.<br><br>``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,<br>if ``sample_weight`` is passed.<br><br>.. versionadded:: 0.19</span>
        </a>
    </td>
            <td class="value">0.0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('init',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-init;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=init,-estimator%20or%20%27zero%27%2C%20default%3DNone">
            init
            <span class="param-doc-description"
            style="position-anchor: --doc-link-init;">
            init: estimator or &#x27;zero&#x27;, default=None<br><br>An estimator object that is used to compute the initial predictions.<br>``init`` has to provide :term:`fit` and :term:`predict`. If &#x27;zero&#x27;, the<br>initial raw predictions are set to zero. By default a<br>``DummyEstimator`` is used, predicting either the average target value<br>(for loss=&#x27;squared_error&#x27;), or a quantile for the other losses.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_features',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-max_features;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=max_features,-%7B%27sqrt%27%2C%20%27log2%27%7D%2C%20int%20or%20float%2C%20default%3DNone">
            max_features
            <span class="param-doc-description"
            style="position-anchor: --doc-link-max_features;">
            max_features: {&#x27;sqrt&#x27;, &#x27;log2&#x27;}, int or float, default=None<br><br>The number of features to consider when looking for the best split:<br><br>- If int, values must be in the range `[1, inf)`.<br>- If float, values must be in the range `(0.0, 1.0]` and the features<br>  considered at each split will be `max(1, int(max_features * n_features_in_))`.<br>- If &quot;sqrt&quot;, then `max_features=sqrt(n_features)`.<br>- If &quot;log2&quot;, then `max_features=log2(n_features)`.<br>- If None, then `max_features=n_features`.<br><br>Choosing `max_features &lt; n_features` leads to a reduction of variance<br>and an increase in bias.<br><br>Note: the search for a split does not stop until at least one<br>valid partition of the node samples is found, even if it requires to<br>effectively inspect more than ``max_features`` features.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('alpha',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-alpha;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=alpha,-float%2C%20default%3D0.9">
            alpha
            <span class="param-doc-description"
            style="position-anchor: --doc-link-alpha;">
            alpha: float, default=0.9<br><br>The alpha-quantile of the huber loss function and the quantile<br>loss function. Only if ``loss=&#x27;huber&#x27;`` or ``loss=&#x27;quantile&#x27;``.<br>Values must be in the range `(0.0, 1.0)`.</span>
        </a>
    </td>
            <td class="value">0.9</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('verbose',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-verbose;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=verbose,-int%2C%20default%3D0">
            verbose
            <span class="param-doc-description"
            style="position-anchor: --doc-link-verbose;">
            verbose: int, default=0<br><br>Enable verbose output. If 1 then it prints progress and performance<br>once in a while (the more trees the lower the frequency). If greater<br>than 1 then it prints progress and performance for every tree.<br>Values must be in the range `[0, inf)`.</span>
        </a>
    </td>
            <td class="value">0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_leaf_nodes',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-max_leaf_nodes;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=max_leaf_nodes,-int%2C%20default%3DNone">
            max_leaf_nodes
            <span class="param-doc-description"
            style="position-anchor: --doc-link-max_leaf_nodes;">
            max_leaf_nodes: int, default=None<br><br>Grow trees with ``max_leaf_nodes`` in best-first fashion.<br>Best nodes are defined as relative reduction in impurity.<br>Values must be in the range `[2, inf)`.<br>If None, then unlimited number of leaf nodes.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('warm_start',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-warm_start;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=warm_start,-bool%2C%20default%3DFalse">
            warm_start
            <span class="param-doc-description"
            style="position-anchor: --doc-link-warm_start;">
            warm_start: bool, default=False<br><br>When set to ``True``, reuse the solution of the previous call to fit<br>and add more estimators to the ensemble, otherwise, just erase the<br>previous solution. See :term:`the Glossary &lt;warm_start&gt;`.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('validation_fraction',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-validation_fraction;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=validation_fraction,-float%2C%20default%3D0.1">
            validation_fraction
            <span class="param-doc-description"
            style="position-anchor: --doc-link-validation_fraction;">
            validation_fraction: float, default=0.1<br><br>The proportion of training data to set aside as validation set for<br>early stopping. Values must be in the range `(0.0, 1.0)`.<br>Only used if ``n_iter_no_change`` is set to an integer.<br><br>.. versionadded:: 0.20</span>
        </a>
    </td>
            <td class="value">0.1</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('n_iter_no_change',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-n_iter_no_change;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=n_iter_no_change,-int%2C%20default%3DNone">
            n_iter_no_change
            <span class="param-doc-description"
            style="position-anchor: --doc-link-n_iter_no_change;">
            n_iter_no_change: int, default=None<br><br>``n_iter_no_change`` is used to decide if early stopping will be used<br>to terminate training when validation score is not improving. By<br>default it is set to None to disable early stopping. If set to a<br>number, it will set aside ``validation_fraction`` size of the training<br>data as validation and terminate training when validation score is not<br>improving in all of the previous ``n_iter_no_change`` numbers of<br>iterations.<br>Values must be in the range `[1, inf)`.<br>See<br>:ref:`sphx_glr_auto_examples_ensemble_plot_gradient_boosting_early_stopping.py`.<br><br>.. versionadded:: 0.20</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('tol',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-tol;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=tol,-float%2C%20default%3D1e-4">
            tol
            <span class="param-doc-description"
            style="position-anchor: --doc-link-tol;">
            tol: float, default=1e-4<br><br>Tolerance for the early stopping. When the loss is not improving<br>by at least tol for ``n_iter_no_change`` iterations (if set to a<br>number), the training stops.<br>Values must be in the range `[0.0, inf)`.<br><br>.. versionadded:: 0.20</span>
        </a>
    </td>
            <td class="value">0.0001</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('ccp_alpha',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-ccp_alpha;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=ccp_alpha,-non-negative%20float%2C%20default%3D0.0">
            ccp_alpha
            <span class="param-doc-description"
            style="position-anchor: --doc-link-ccp_alpha;">
            ccp_alpha: non-negative float, default=0.0<br><br>Complexity parameter used for Minimal Cost-Complexity Pruning. The<br>subtree with the largest cost complexity that is smaller than<br>``ccp_alpha`` will be chosen. By default, no pruning is performed.<br>Values must be in the range `[0.0, inf)`.<br>See :ref:`minimal_cost_complexity_pruning` for details. See<br>:ref:`sphx_glr_auto_examples_tree_plot_cost_complexity_pruning.py`<br>for an example of such pruning.<br><br>.. versionadded:: 0.22</span>
        </a>
    </td>
            <td class="value">0.0</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>

        <div class="estimator-table">
            <details>
                <summary>Fitted attributes</summary>
                <table class="parameters-table">
                    <tbody>
                        <tr>
                        <th>Name</th>
                        <th>Type</th>
                        <th>Value</th>
                        </tr>

       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-estimators_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=estimators_,-int">
            estimators_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-estimators_;">
            estimators_: ndarray of DecisionTreeRegressor of shape (n_estimators, 1)<br><br>The collection of fitted sub-estimators.</span>
        </a>
    </td>
           <td class="fitted-att-type">ndarray[object](200, 1)</td>
           <td>[[DecisionTreeRegressor(max_depth=3,
                        random_state=RandomState(MT19937) at 0x187EA4BDE40)],
 [DecisionTreeRegressor(max_depth=3,
                        random_state=RandomState(MT19937) at 0x187EA4BDE40)],
 [DecisionTreeRegressor(max_depth=3,
                        random_state=RandomState(MT19937) at 0x187EA4BDE40)],
 ...,
 [DecisionTreeRegressor(max_depth=3,
                        random_state=RandomState(MT19937) at 0x187EA4BDE40)],
 [DecisionTreeRegressor(max_depth=3,
                        random_state=RandomState(MT19937) at 0x187EA4BDE40)],
 [DecisionTreeRegressor(max_depth=3,
                        random_state=RandomState(MT19937) at 0x187EA4BDE40)]]</td>


       </tr>


       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-feature_importances_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=feature_importances_,-ndarray%20of%20shape%20%28n_features%2C%29">
            feature_importances_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-feature_importances_;">
            feature_importances_: ndarray of shape (n_features,)<br><br>The impurity-based feature importances.<br>The higher, the more important the feature.<br>The importance of a feature is computed as the (normalized)<br>total reduction of the MSE brought by that feature.  It is also<br>known as the Gini importance.<br><br>Warning: impurity-based feature importances can be misleading for<br>high cardinality features (many unique values). See<br>:func:`sklearn.inspection.permutation_importance` as an alternative.</span>
        </a>
    </td>
           <td class="fitted-att-type">ndarray[float64](16,)</td>
           <td>[0.,0.,0.,...,0.,0.,0.]</td>


       </tr>


       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-feature_names_in_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=feature_names_in_,-ndarray%20of%20shape%20%28n_features_in_%2C%29">
            feature_names_in_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-feature_names_in_;">
            feature_names_in_: ndarray of shape (`n_features_in_`,)<br><br>Names of features seen during :term:`fit`. Defined only when `X`<br>has feature names that are all strings.<br><br>.. versionadded:: 1.0</span>
        </a>
    </td>
           <td class="fitted-att-type">ndarray[object](16,)</td>
           <td>[&#x27;Children apprehended and placed in CBP custody*&#x27;,
 &#x27;Children in CBP custody&#x27;,&#x27;Children transferred out of CBP custody&#x27;,...,
 &#x27;Month&#x27;,&#x27;Quarter&#x27;,&#x27;Year&#x27;]</td>


       </tr>


       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-init_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=init_,-estimator">
            init_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-init_;">
            init_: estimator<br><br>The estimator that provides the initial predictions. Set via the ``init``<br>argument.</span>
        </a>
    </td>
           <td class="fitted-att-type">DummyRegressor</td>
           <td>DummyRegressor()</td>


       </tr>


       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-max_features_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=max_features_,-int">
            max_features_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-max_features_;">
            max_features_: int<br><br>The inferred value of max_features.</span>
        </a>
    </td>
           <td class="fitted-att-type">int</td>
           <td>16</td>


       </tr>


       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-n_estimators_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=n_estimators_,-int">
            n_estimators_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-n_estimators_;">
            n_estimators_: int<br><br>The number of estimators as selected by early stopping (if<br>``n_iter_no_change`` is specified). Otherwise it is set to<br>``n_estimators``.</span>
        </a>
    </td>
           <td class="fitted-att-type">int</td>
           <td>200</td>


       </tr>


       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-n_features_in_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=n_features_in_,-int">
            n_features_in_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-n_features_in_;">
            n_features_in_: int<br><br>Number of features seen during :term:`fit`.<br><br>.. versionadded:: 0.24</span>
        </a>
    </td>
           <td class="fitted-att-type">int</td>
           <td>16</td>


       </tr>


       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-n_trees_per_iteration_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=n_trees_per_iteration_,-int">
            n_trees_per_iteration_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-n_trees_per_iteration_;">
            n_trees_per_iteration_: int<br><br>The number of trees that are built at each iteration. For regressors, this is<br>always 1.<br><br>.. versionadded:: 1.4.0</span>
        </a>
    </td>
           <td class="fitted-att-type">int</td>
           <td>1</td>


       </tr>


       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-train_score_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#:~:text=train_score_,-ndarray%20of%20shape%20%28n_estimators%2C%29">
            train_score_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-train_score_;">
            train_score_: ndarray of shape (n_estimators,)<br><br>The i-th score ``train_score_[i]`` is the loss of the<br>model at iteration ``i`` on the in-bag sample.<br>If ``subsample == 1`` this is the loss on the training data.</span>
        </a>
    </td>
           <td class="fitted-att-type">ndarray[float64](200,)</td>
           <td>[5113476.59,4627116.99,4187821.2 ,...,   2667.49,   2651.62,   2637.14]</td>


       </tr>

                    </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div><script>/*  Authors: The scikit-learn developers
 SPDX-License-Identifier: BSD-3-Clause
*/

function copyToClipboard(text, element) {
    // Get the parameter prefix from the closest toggleable content
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const fullParamName = paramPrefix ? `${paramPrefix}${text}` : text;

    const originalStyle = element.style;
    const computedStyle = window.getComputedStyle(element);
    const originalWidth = computedStyle.width;
    const originalHTML = element.innerHTML.replace('Copied!', '');

    navigator.clipboard.writeText(fullParamName)
        .then(() => {
            element.style.width = originalWidth;
            element.style.color = 'green';
            element.innerHTML = "Copied!";

            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        })
        .catch(err => {
            console.error('Failed to copy:', err);
            element.style.color = 'red';
            element.innerHTML = "Failed!";
            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        });
    return false;
}

document.querySelectorAll('.copy-paste-icon').forEach(function(element) {
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';

    const parent = element.parentElement;
    if (!parent || !parent.nextElementSibling) {
        console.warn('Expected copy-paste icon is missing from the DOM structure');
        return;
    }

    const paramName = element.parentElement.nextElementSibling
        .textContent.trim().split(' ')[0];
    const fullParamName = paramPrefix ? `${paramPrefix}${paramName}` : paramName;

    element.setAttribute('title', fullParamName);
});

/**
 * Copy the list of feature names formatted as a Python list.
 *
 * @param {HTMLElement} element - The copy button inside a `.features` block; its siblings
 *   contain a `details` element and a table containing feature named.
 * @returns {boolean} Always returns `false` so callers can prevent the default click behavior.
 */
function copyFeatureNamesToClipboard(element) {
    var detailsElem = element.closest('.features').querySelector('details');
    var wasOpen = detailsElem.open;
    detailsElem.open = true;
    var content = element.closest('.features').querySelector('tbody')
                  .innerText.trim();
    if (!wasOpen) detailsElem.open = false;
    const rows = content.split('\n').map(row => `    "${row}"`);
    const formattedText = `[\n${rows.join(',\n')},\n]`;
    const originalHTML = element.innerHTML.replace('âœ”', '');
    const originalStyle = element.style;
    const copyMark = document.createElement('span');
    copyMark.innerHTML = 'âœ”';
    copyMark.style.color = 'blue';
    copyMark.style.fontSize = '1em';

    navigator.clipboard.writeText(formattedText)
        .then(() => {
            element.style.display = 'none';
            element.parentElement.appendChild(copyMark);

            setTimeout(() => {
                copyMark.remove();
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 1000);
        })
        .catch(err => {
            console.error('Failed to copy:', err);
            element.style.color = 'orange';
            element.innerHTML = "Failed!";
            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 1000);
        });
    return false;
}
/**
 * Adapted from Skrub
 * https://github.com/skrub-data/skrub/blob/403466d1d5d4dc76a7ef569b3f8228db59a31dc3/skrub/_reporting/_data/templates/report.js#L789
 * @returns "light" or "dark"
 */
function detectTheme(element) {
    const body = document.querySelector('body');

    // Check VSCode theme
    const themeKindAttr = body.getAttribute('data-vscode-theme-kind');
    const themeNameAttr = body.getAttribute('data-vscode-theme-name');

    if (themeKindAttr && themeNameAttr) {
        const themeKind = themeKindAttr.toLowerCase();
        const themeName = themeNameAttr.toLowerCase();

        if (themeKind.includes("dark") || themeName.includes("dark")) {
            return "dark";
        }
        if (themeKind.includes("light") || themeName.includes("light")) {
            return "light";
        }
    }

    // Check Jupyter theme
    if (body.getAttribute('data-jp-theme-light') === 'false') {
        return 'dark';
    } else if (body.getAttribute('data-jp-theme-light') === 'true') {
        return 'light';
    }

    // Guess based on a parent element's color
    const color = window.getComputedStyle(element.parentNode, null).getPropertyValue('color');
    const match = color.match(/^rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*$/i);
    if (match) {
        const [r, g, b] = [
            parseFloat(match[1]),
            parseFloat(match[2]),
            parseFloat(match[3])
        ];

        // https://en.wikipedia.org/wiki/HSL_and_HSV#Lightness
        const luma = 0.299 * r + 0.587 * g + 0.114 * b;

        if (luma > 180) {
            // If the text is very bright we have a dark theme
            return 'dark';
        }
        if (luma < 75) {
            // If the text is very dark we have a light theme
            return 'light';
        }
        // Otherwise fall back to the next heuristic.
    }

    // Fallback to system preference
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}


function forceTheme(elementId) {
    const estimatorElement = document.querySelector(`#${elementId}`);
    if (estimatorElement === null) {
        console.error(`Element with id ${elementId} not found.`);
    } else {
        const theme = detectTheme(estimatorElement);
        estimatorElement.classList.add(theme);
    }
}

forceTheme('sk-container-id-4');</script></body>




```python
gb_predictions = gb_model.predict(X_test)
```


```python
gb_mae = mean_absolute_error(y_test, gb_predictions)

gb_rmse = np.sqrt(mean_squared_error(y_test, gb_predictions))

gb_mape = mean_absolute_percentage_error(
    y_test,
    gb_predictions
) * 100

print("Gradient Boosting Results")
print(f"MAE  : {gb_mae:.2f}")
print(f"RMSE : {gb_rmse:.2f}")
print(f"MAPE : {gb_mape:.2f}%")
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[151], line 5
          1 gb_mae = mean_absolute_error(y_test, gb_predictions)
          3 gb_rmse = np.sqrt(mean_squared_error(y_test, gb_predictions))
    ----> 5 gb_mape = mean_absolute_percentage_error(
          6     y_test,
          7     gb_predictions
          8 ) * 100
         10 print("Gradient Boosting Results")
         11 print(f"MAE  : {gb_mae:.2f}")
    

    NameError: name 'mean_absolute_percentage_error' is not defined



```python

print("gb_mae :", 'gb_mae' in globals())
print("gb_rmse:", 'gb_rmse' in globals())
print("gb_mape:", 'gb_mape' in globals())
```

    gb_mae : True
    gb_rmse: True
    gb_mape: False
    


```python

from sklearn.metrics import mean_absolute_percentage_error
```


```python
gb_mape = mean_absolute_percentage_error(
    y_test,
    gb_predictions
) * 100

print("Gradient Boosting Results")
print(f"MAE  : {gb_mae:.2f}")
print(f"RMSE : {gb_rmse:.2f}")
print(f"MAPE : {gb_mape:.2f}%")
```

    Gradient Boosting Results
    MAE  : 61.76
    RMSE : 82.73
    MAPE : 2.84%
    


```python
variables = [
    "naive_mae", "naive_rmse", "naive_mape",
    "moving_mae", "moving_rmse", "moving_mape",
    "exp_mae", "exp_rmse", "exp_mape",
    "arima_mae", "arima_rmse", "arima_mape",
    "rf_mae", "rf_rmse", "rf_mape",
    "gb_mae", "gb_rmse", "gb_mape"
]

for var in variables:
    print(f"{var}: {'✅ Exists' if var in globals() else '❌ Missing'}")
```

    naive_mae: ✅ Exists
    naive_rmse: ✅ Exists
    naive_mape: ✅ Exists
    moving_mae: ✅ Exists
    moving_rmse: ✅ Exists
    moving_mape: ✅ Exists
    exp_mae: ✅ Exists
    exp_rmse: ✅ Exists
    exp_mape: ✅ Exists
    arima_mae: ✅ Exists
    arima_rmse: ✅ Exists
    arima_mape: ✅ Exists
    rf_mae: ✅ Exists
    rf_rmse: ✅ Exists
    rf_mape: ✅ Exists
    gb_mae: ✅ Exists
    gb_rmse: ✅ Exists
    gb_mape: ✅ Exists
    


```python
results = {
    "Naive Forecast": [naive_mae, naive_rmse, naive_mape],
    "Moving Average": [moving_mae, moving_rmse, moving_mape],
    "Exponential Smoothing": [exp_mae, exp_rmse, exp_mape],
    "ARIMA": [arima_mae, arima_rmse, arima_mape],
    "Random Forest": [rf_mae, rf_rmse, rf_mape],
    "Gradient Boosting": [gb_mae, gb_rmse, gb_mape]
}

comparison = pd.DataFrame(
    results,
    index=["MAE", "RMSE", "MAPE"]
).T

comparison = comparison.sort_values("MAE")

comparison
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MAE</th>
      <th>RMSE</th>
      <th>MAPE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Naive Forecast</th>
      <td>10.304965</td>
      <td>14.075531</td>
      <td>0.456147</td>
    </tr>
    <tr>
      <th>Moving Average</th>
      <td>33.229376</td>
      <td>40.683292</td>
      <td>1.480261</td>
    </tr>
    <tr>
      <th>Gradient Boosting</th>
      <td>61.762405</td>
      <td>82.734925</td>
      <td>2.836391</td>
    </tr>
    <tr>
      <th>Random Forest</th>
      <td>66.439296</td>
      <td>88.804823</td>
      <td>3.081338</td>
    </tr>
    <tr>
      <th>ARIMA</th>
      <td>197.555783</td>
      <td>241.430596</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Exponential Smoothing</th>
      <td>732.498944</td>
      <td>813.978493</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python

# ==========================================
# Train ARIMA on Full Dataset
# ==========================================

from statsmodels.tsa.arima.model import ARIMA

final_arima = ARIMA(
    df["Children in HHS Care"],
    order=(1,1,1)
)

final_arima_fit = final_arima.fit()

print(final_arima_fit.summary())
```

    C:\Users\pccli\AppData\Local\Programs\Python\Python313\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    C:\Users\pccli\AppData\Local\Programs\Python\Python313\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    C:\Users\pccli\AppData\Local\Programs\Python\Python313\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    

                                    SARIMAX Results                                 
    ================================================================================
    Dep. Variable:     Children in HHS Care   No. Observations:                  720
    Model:                   ARIMA(1, 1, 1)   Log Likelihood               -4535.258
    Date:                  Thu, 09 Jul 2026   AIC                           9076.517
    Time:                          15:46:35   BIC                           9090.251
    Sample:                               0   HQIC                          9081.819
                                      - 720                                         
    Covariance Type:                    opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    ar.L1          0.0725      0.179      0.404      0.686      -0.279       0.424
    ma.L1          0.1248      0.179      0.699      0.485      -0.225       0.475
    sigma2      1.766e+04    468.137     37.733      0.000    1.67e+04    1.86e+04
    ===================================================================================
    Ljung-Box (L1) (Q):                   0.07   Jarque-Bera (JB):              2222.04
    Prob(Q):                              0.79   Prob(JB):                         0.00
    Heteroskedasticity (H):               0.10   Skew:                            -1.45
    Prob(H) (two-sided):                  0.00   Kurtosis:                        11.11
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
    


```python
# ==========================================
# Future Forecast
# ==========================================

forecast_steps = 30

future_forecast = final_arima_fit.forecast(
    steps=forecast_steps
)

print(future_forecast)
```

    720    2486.630918
    721    2486.821659
    722    2486.835487
    723    2486.836490
    724    2486.836562
    725    2486.836568
    726    2486.836568
    727    2486.836568
    728    2486.836568
    729    2486.836568
    730    2486.836568
    731    2486.836568
    732    2486.836568
    733    2486.836568
    734    2486.836568
    735    2486.836568
    736    2486.836568
    737    2486.836568
    738    2486.836568
    739    2486.836568
    740    2486.836568
    741    2486.836568
    742    2486.836568
    743    2486.836568
    744    2486.836568
    745    2486.836568
    746    2486.836568
    747    2486.836568
    748    2486.836568
    749    2486.836568
    Name: predicted_mean, dtype: float64
    

    C:\Users\pccli\AppData\Local\Programs\Python\Python313\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
      return get_prediction_index(
    C:\Users\pccli\AppData\Local\Programs\Python\Python313\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: FutureWarning: No supported index is available. In the next version, calling this method in a model without a supported index will result in an exception.
      return get_prediction_index(
    


```python

forecast_result = final_arima_fit.get_forecast(
    steps=forecast_steps
)

forecast_mean = forecast_result.predicted_mean

confidence_interval = forecast_result.conf_int()

print(confidence_interval.head())
```

         lower Children in HHS Care  upper Children in HHS Care
    720                 2226.137798                 2747.124038
    721                 2080.448805                 2893.194512
    722                 1972.287178                 3001.383797
    723                 1883.055342                 3090.617637
    724                 1805.400019                 3168.273106
    

    C:\Users\pccli\AppData\Local\Programs\Python\Python313\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
      return get_prediction_index(
    


```python
plt.figure(figsize=(15,6))

# Historical data
plt.plot(
    df.index,
    df["Children in HHS Care"],
    label="Historical Data"
)

# Forecast
plt.plot(
    forecast_mean.index,
    forecast_mean,
    color="red",
    label="Forecast"
)

# Confidence Interval
plt.fill_between(
    forecast_mean.index,
    confidence_interval.iloc[:,0],
    confidence_interval.iloc[:,1],
    alpha=0.3,
    label="95% Confidence Interval"
)

plt.title("Future HHS Care Load Forecast")
plt.xlabel("Observation")
plt.ylabel("Children in HHS Care")

plt.legend()

plt.grid(True)

plt.show()
```


    
![png](output_121_0.png)
    



```python

# ==========================================
# Discharge Demand Forecast
# ==========================================

discharge_series = df["Children discharged from HHS Care"]

discharge_model = ARIMA(
    discharge_series,
    order=(1,1,1)
)

discharge_fit = discharge_model.fit()

print(discharge_fit.summary())
```

    C:\Users\pccli\AppData\Local\Programs\Python\Python313\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    C:\Users\pccli\AppData\Local\Programs\Python\Python313\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    C:\Users\pccli\AppData\Local\Programs\Python\Python313\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    C:\Users\pccli\AppData\Local\Programs\Python\Python313\Lib\site-packages\statsmodels\tsa\statespace\sarimax.py:978: UserWarning: Non-invertible starting MA parameters found. Using zeros as starting parameters.
      warn('Non-invertible starting MA parameters found.'
    

                                           SARIMAX Results                                       
    =============================================================================================
    Dep. Variable:     Children discharged from HHS Care   No. Observations:                  720
    Model:                                ARIMA(1, 1, 1)   Log Likelihood               -3799.663
    Date:                               Thu, 09 Jul 2026   AIC                           7605.327
    Time:                                       15:53:34   BIC                           7619.060
    Sample:                                            0   HQIC                          7610.629
                                                   - 720                                         
    Covariance Type:                                 opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    ar.L1          0.2977      0.035      8.559      0.000       0.230       0.366
    ma.L1         -0.8905      0.018    -50.667      0.000      -0.925      -0.856
    sigma2      2275.8730     99.829     22.798      0.000    2080.212    2471.534
    ===================================================================================
    Ljung-Box (L1) (Q):                  14.85   Jarque-Bera (JB):                38.25
    Prob(Q):                              0.00   Prob(JB):                         0.00
    Heteroskedasticity (H):               0.07   Skew:                             0.25
    Prob(H) (two-sided):                  0.00   Kurtosis:                         4.01
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
    


```python
discharge_forecast = discharge_fit.get_forecast(
    steps=30
)

discharge_mean = discharge_forecast.predicted_mean

discharge_ci = discharge_forecast.conf_int()

print(discharge_mean.head())
```

    720    11.277456
    721    10.467034
    722    10.225794
    723    10.153984
    724    10.132609
    Name: predicted_mean, dtype: float64
    

    C:\Users\pccli\AppData\Local\Programs\Python\Python313\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
      return get_prediction_index(
    C:\Users\pccli\AppData\Local\Programs\Python\Python313\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:837: FutureWarning: No supported index is available. In the next version, calling this method in a model without a supported index will result in an exception.
      return get_prediction_index(
    


```python

plt.figure(figsize=(15,6))

plt.plot(
    discharge_series,
    label="Historical Discharges"
)

plt.plot(
    discharge_mean.index,
    discharge_mean,
    color="green",
    linewidth=2,
    label="Forecast"
)

plt.fill_between(
    discharge_mean.index,
    discharge_ci.iloc[:,0],
    discharge_ci.iloc[:,1],
    color="green",
    alpha=0.25,
    label="95% Confidence Interval"
)

plt.title("Future Discharge Demand Forecast")
plt.xlabel("Time")
plt.ylabel("Children Discharged")

plt.legend()

plt.grid(True)

plt.show()
```


    
![png](output_124_0.png)
    



```python

```


```python

```


```python
pip install streamlit
```

    Requirement already satisfied: streamlit in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (1.59.1)
    Requirement already satisfied: altair!=5.4.0,!=5.4.1,<7,>=4.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (6.2.2)
    Requirement already satisfied: blinker<2,>=1.5.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (1.9.0)
    Requirement already satisfied: cachetools<8,>=5.5 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (7.1.4)
    Requirement already satisfied: click<9,>=7.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (8.4.2)
    Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (3.1.50)
    Requirement already satisfied: numpy<3,>=1.23 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (2.3.4)
    Requirement already satisfied: packaging>=20 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (25.0)
    Requirement already satisfied: pandas<4,>=1.4.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (2.3.3)
    Requirement already satisfied: pillow<13,>=7.1.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (12.0.0)
    Requirement already satisfied: pydeck<1,>=0.8.0b4 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (0.9.3)
    Requirement already satisfied: protobuf<8,>=3.20 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (7.35.1)
    Requirement already satisfied: pyarrow>=7.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (24.0.0)
    Requirement already satisfied: requests<3,>=2.27 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (2.32.3)
    Requirement already satisfied: tenacity<10,>=8.1.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (9.1.4)
    Requirement already satisfied: toml<2,>=0.10.1 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (0.10.2)
    Requirement already satisfied: typing-extensions<5,>=4.10.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (4.13.2)
    Requirement already satisfied: starlette>=0.40.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (1.3.1)
    Requirement already satisfied: uvicorn>=0.30.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (0.51.0)
    Requirement already satisfied: httptools>=0.6.3 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (0.8.0)
    Requirement already satisfied: anyio>=4.0.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (4.9.0)
    Requirement already satisfied: python-multipart>=0.0.10 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (0.0.32)
    Requirement already satisfied: websockets>=12.0.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (16.0)
    Requirement already satisfied: itsdangerous>=2.1.2 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (2.2.0)
    Requirement already satisfied: watchdog<7,>=2.1.5 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (6.0.0)
    Requirement already satisfied: jinja2 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (3.1.6)
    Requirement already satisfied: jsonschema>=3.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (4.24.0)
    Requirement already satisfied: narwhals>=2.4.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (2.23.0)
    Requirement already satisfied: colorama in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from click<9,>=7.0->streamlit) (0.4.6)
    Requirement already satisfied: gitdb<5,>=4.0.1 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.12)
    Requirement already satisfied: smmap<6,>=3.0.1 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.3)
    Requirement already satisfied: python-dateutil>=2.8.2 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from pandas<4,>=1.4.0->streamlit) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from pandas<4,>=1.4.0->streamlit) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from pandas<4,>=1.4.0->streamlit) (2025.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from requests<3,>=2.27->streamlit) (3.4.2)
    Requirement already satisfied: idna<4,>=2.5 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from requests<3,>=2.27->streamlit) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from requests<3,>=2.27->streamlit) (2.4.0)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from requests<3,>=2.27->streamlit) (2025.4.26)
    Requirement already satisfied: sniffio>=1.1 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from anyio>=4.0.0->streamlit) (1.3.1)
    Requirement already satisfied: MarkupSafe>=2.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from jinja2->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (3.0.2)
    Requirement already satisfied: attrs>=22.2.0 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (25.3.0)
    Requirement already satisfied: jsonschema-specifications>=2023.03.6 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (2025.4.1)
    Requirement already satisfied: referencing>=0.28.4 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (0.36.2)
    Requirement already satisfied: rpds-py>=0.7.1 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (0.25.1)
    Requirement already satisfied: six>=1.5 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from python-dateutil>=2.8.2->pandas<4,>=1.4.0->streamlit) (1.17.0)
    Requirement already satisfied: h11>=0.8 in c:\users\pccli\appdata\local\programs\python\python313\lib\site-packages (from uvicorn>=0.30.0->streamlit) (0.16.0)
    Note: you may need to restart the kernel to use updated packages.
    

    
    [notice] A new release of pip is available: 25.1.1 -> 26.1.2
    [notice] To update, run: python.exe -m pip install --upgrade pip
    


```python

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[166], line 1
    ----> 1 app.py
    

    NameError: name 'app' is not defined



```python
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
```


```python
st.set_page_config(
    page_title="HHS Care Load Forecast",
    layout="wide"
)

st.title("Predictive Forecasting of Care Load & Placement Demand")

st.markdown(
"""
Forecasting future HHS care load and discharge demand
using Time-Series and Machine Learning models.
"""
)
```

    2026-07-09 16:04:56.714 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2026-07-09 16:04:56.718 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2026-07-09 16:04:57.668 
      [33m[1mWarning:[0m to view this Streamlit app on a browser, run it with the following
      command:
    
        streamlit run C:\Users\pccli\AppData\Local\Programs\Python\Python313\Lib\site-packages\ipykernel_launcher.py [ARGUMENTS]
    2026-07-09 16:04:57.669 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2026-07-09 16:04:57.671 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2026-07-09 16:04:57.673 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2026-07-09 16:04:57.675 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2026-07-09 16:04:57.677 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    




    DeltaGenerator()




```python
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Current HHS Care",
        int(df["Children in HHS Care"].iloc[-1])
    )

with col2:
    st.metric(
        "Current Discharges",
        int(df["Children discharged from HHS Care"].iloc[-1])
    )

with col3:
    st.metric(
        "Best Model",
        "Naïve Forecast"
    )

with col4:
    st.metric(
        "Forecast Accuracy",
        "99.54%"
    )
```

    2026-07-09 16:05:07.467 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2026-07-09 16:05:07.470 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2026-07-09 16:05:07.474 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2026-07-09 16:05:07.478 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2026-07-09 16:05:07.480 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2026-07-09 16:05:07.484 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2026-07-09 16:05:07.487 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2026-07-09 16:05:07.490 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2026-07-09 16:05:07.493 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2026-07-09 16:05:07.495 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2026-07-09 16:05:07.496 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2026-07-09 16:05:07.499 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2026-07-09 16:05:07.500 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2026-07-09 16:05:07.502 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2026-07-09 16:05:07.505 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2026-07-09 16:05:07.507 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    2026-07-09 16:05:07.508 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
    


```python

```

