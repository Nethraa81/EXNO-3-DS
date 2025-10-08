## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
<img width="690" height="417" alt="image" src="https://github.com/user-attachments/assets/6a32bc66-35d0-4b79-a7d3-8845d20a29b0" />

```
# ORDINAL ENCODING
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
<img width="391" height="219" alt="image" src="https://github.com/user-attachments/assets/3df6b980-42b5-4a25-b2a5-cc0c12219fee" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="530" height="416" alt="image" src="https://github.com/user-attachments/assets/cb678e25-64af-43be-a99d-9ba4f3e058e6" />

```
 # Label Encoder ( orders in alphabetical order)
 le=LabelEncoder()
 dfc=df.copy()
 dfc['ord_2']=le.fit_transform(dfc['ord_2'])
 dfc
```
<img width="662" height="409" alt="image" src="https://github.com/user-attachments/assets/28ff6acf-e59d-44d6-8b2f-3af166e91d9c" />

```
 # ONE HOT ENCODING
 from sklearn.preprocessing import OneHotEncoder
 ohe=OneHotEncoder(sparse_output=False)
 df2=df.copy()
 enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]])) # Orders in Alphabetical Order Blue , Green, Red
 df2=pd.concat([df2,enc],axis=1)
 df2
```
<img width="535" height="422" alt="image" src="https://github.com/user-attachments/assets/7cdd4fa6-d08b-4f20-967b-e406181fc74a" />

```
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="805" height="429" alt="image" src="https://github.com/user-attachments/assets/c56df048-3b36-49df-b701-d7dfca6942e2" />

```
pip install --upgrade category_encoders
```
<img width="1376" height="375" alt="image" src="https://github.com/user-attachments/assets/41b10d2f-03a5-4323-b90f-8ab5ae99d3f2" />

```
 # BINARY ENCODER
 from category_encoders import BinaryEncoder
 df=pd.read_csv("data.csv")
 df
```
<img width="603" height="416" alt="image" src="https://github.com/user-attachments/assets/229dd561-8220-4ad3-a772-666bc0c1dbf5" />

```
 be=BinaryEncoder()
 nd=be.fit_transform(df['Ord_2'])
 dfb=pd.concat([df,nd],axis=1)
 dfb
```
<img width="801" height="417" alt="image" src="https://github.com/user-attachments/assets/1622e617-eb03-43a5-8118-df9f28e62acd" />

```
 # MEAN ENCODING
 from category_encoders import TargetEncoder
 te=TargetEncoder()
 CC=df.copy()
 new=te.fit_transform(X=CC["City"],y=CC["Target"])
 CC=pd.concat([CC,new],axis=1)
 CC
```
<img width="660" height="422" alt="image" src="https://github.com/user-attachments/assets/9119f34f-f59b-4acc-b558-bed45875b92f" />

```
 import pandas as pd
 from scipy import stats
 import numpy as np
 df=pd.read_csv("Data_to_Transform.csv")
 df
```
<img width="890" height="494" alt="image" src="https://github.com/user-attachments/assets/0157aeed-6793-4561-aa1b-bb4ea86220dc" />

```
df.skew()
```
<img width="394" height="109" alt="image" src="https://github.com/user-attachments/assets/f4e73843-d957-4360-b3b6-1d7b6c8d50eb" />

```
 # 1. LOG TRANSFORMATION
 np.log(df["Highly Positive Skew"])
```
<img width="566" height="258" alt="image" src="https://github.com/user-attachments/assets/71e72bee-c067-490a-9177-59c7e25588dd" />

```
# 2. RECIPROCAL TRANSFORMATION
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="606" height="263" alt="image" src="https://github.com/user-attachments/assets/132d3587-1342-42f7-b16f-e94ee3ad7ff2" />

```
 # 4. SQUARE ROOT TRANSFORMATION
 np.sqrt(df["Highly Positive Skew"])
```
<img width="577" height="263" alt="image" src="https://github.com/user-attachments/assets/77f1e3a1-6271-4d14-8095-36a53bfc42dd" />

```
 # 5. SQUARE TRANSFORMATION
 np.square(df["Highly Positive Skew"])
```
<img width="601" height="262" alt="image" src="https://github.com/user-attachments/assets/b886f7fc-a7b3-4b95-9e6a-fa1415fad483" />

```
# POWER TRANSFORMATIONS
#        BOX COX
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="1138" height="486" alt="image" src="https://github.com/user-attachments/assets/7b9dcc2d-be8b-4c70-96c3-97eb0e4bfda5" />

```
df.skew()
```
<img width="474" height="137" alt="image" src="https://github.com/user-attachments/assets/d8dabafe-9f47-4803-abae-6bfa4b94b712" />

```
 # YEO_JOHNSON
 df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
 df.skew()
```
<img width="538" height="161" alt="image" src="https://github.com/user-attachments/assets/f455e40e-3be2-4a70-aad8-0878fc7cd5a5" />

```
 #QUANTILE TRANSFORMATION
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal')
 df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 df
```

<img width="1377" height="512" alt="image" src="https://github.com/user-attachments/assets/437ef0b5-9210-4c37-8892-92ab72049b1c" />

```
 import seaborn as sns
 import statsmodels.api as sm # STATS MODEL- STATISTICAL MODEL TO VISUALIZE DISTRIBUTION
 import matplotlib.pyplot as plt
 sm.qqplot(df["Moderate Negative Skew"],line='45') # QQ - QUANTILE QUANTILE PLOT
 plt.show()
```

<img width="896" height="528" alt="image" src="https://github.com/user-attachments/assets/75ad6307-b92b-4924-b2b6-3fd30b31989b" />

```
 sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45') # RECIPROCAL
 plt.show()
```

<img width="763" height="531" alt="image" src="https://github.com/user-attachments/assets/e7168c8b-6307-41ee-8284-5d72506e94bb" />

```
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
 df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 sm.qqplot(df["Moderate Negative Skew"],line='45')
 plt.show()
```

<img width="836" height="534" alt="image" src="https://github.com/user-attachments/assets/df9f2618-ad0e-48e4-8372-0241f37b02b6" />


# RESULT:

Thus the program to read the given data and perform Feature Encoding and Transformation process and save the data to a file has been executed successfully.

       
