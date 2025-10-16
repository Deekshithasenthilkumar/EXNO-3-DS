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
```python
import pandas as pd
import numpy as np
from scipy import stats
df=pd.read_csv("data.csv")
df
```
<img width="579" height="429" alt="image" src="https://github.com/user-attachments/assets/daf74e15-cf99-4806-9210-0c6ed84f5e85" />

```python
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
climate=['Cold','Warm','Hot','Very Hot']
ele=OrdinalEncoder(categories=[climate])
ele.fit_transform(df[["Ord_1"]])
```
<img width="155" height="232" alt="image" src="https://github.com/user-attachments/assets/98799a34-6323-4b33-803b-9fc23710c372" />

```python
df['bo2']=ele.fit_transform(df[['Ord_1']])
df
```
<img width="604" height="433" alt="image" src="https://github.com/user-attachments/assets/b01e8bf4-54b2-47a5-978a-5913ffed41cf" />

```python
le=LabelEncoder()
df2=df.copy()
df2['Ord_2']=le.fit_transform(df2['Ord_2'])
df2
```
<img width="566" height="431" alt="image" src="https://github.com/user-attachments/assets/0a50254d-cd03-4f26-80d4-e38657a61174" />

```python
from sklearn.preprocessing import OneHotEncoder 
ohe=OneHotEncoder()
df3=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df[['City']]))
df2=pd.concat([enc,df3],axis=1)
df2
```
<img width="1039" height="427" alt="image" src="https://github.com/user-attachments/assets/9d57e5ab-862e-460c-a94c-b9db347453fc" />

```python
pd.get_dummies(df,columns=['City'])
```
<img width="1027" height="439" alt="image" src="https://github.com/user-attachments/assets/a56030c7-199f-4e7a-9bb9-9debd23a5081" />

```python
pip install --upgrade category_encoders
```
<img width="1543" height="353" alt="image" src="https://github.com/user-attachments/assets/b84e8315-1bf4-4a61-a844-a79b7179fd92" />

```python
from category_encoders import BinaryEncoder
```
from category_encoders import BinaryEncoder

```python
import pandas as pd
```
df=pd.read_csv("C:\Users\priya\Downloads\data.csv")

```python
be=BinaryEncoder()
```
nd=be.fit_transform(df['Ord_2'])

```python
df1=pd.concat([df,nd],axis=1)
```
df1=df.copy()

```python
df1
```
<img width="656" height="448" alt="image" src="https://github.com/user-attachments/assets/a568c499-5412-4c24-a503-83a6606a5995" />

```python
from category_encoders import TargetEncoder
```
te=TargetEncoder()

```python
cc=df.copy()
```
new=te.fit_transform(X=cc["City"],y=cc["Target"])

```python
cc=pd.concat([cc,new],axis=1)
cc
```

<img width="750" height="465" alt="image" src="https://github.com/user-attachments/assets/840f0595-4f4e-45cb-a910-9ac907ce785a" />

``python
import pandas as pd
from scipy import stats
import numpy as np
```
```python
df=pd.read_csv("Data_to_Transform.csv")
df
```

<img width="1041" height="535" alt="image" src="https://github.com/user-attachments/assets/e8478bb1-69df-41ec-8216-44117067f9fb" />

```python
df.skew()
```
<img width="392" height="128" alt="image" src="https://github.com/user-attachments/assets/47862a54-8b12-4d05-b641-2528fe4b108a" />

```python
np.log(df["Highly Positive Skew"])
```
<img width="650" height="305" alt="image" src="https://github.com/user-attachments/assets/50346f72-295f-4444-b2c4-76a256694b50" />

```python
np.reciprocal(df["Highly Positive Skew"])
```
<img width="740" height="302" alt="image" src="https://github.com/user-attachments/assets/775c6d67-772d-4197-8545-dc5c4668c3ec" />

```python
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="667" height="302" alt="image" src="https://github.com/user-attachments/assets/fe78dcb0-77fc-4f5e-98bb-07ccc7f8a946" />

```python
np.square(df["Highly Positive Skew"])
```
<img width="658" height="286" alt="image" src="https://github.com/user-attachments/assets/a28bd238-113b-400e-8adb-882d88d88e9c" />

```python
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="1341" height="541" alt="image" src="https://github.com/user-attachments/assets/8c833781-dae4-4ce0-919c-179ad4adf548" />

```python
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df['Moderate Negative Skew'])
```
from sklearn.preprocessing import QuantileTransformer

```python
qt=QuantileTransformer(output_distribution='normal')
```
```python
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

<img width="1611" height="582" alt="image" src="https://github.com/user-attachments/assets/5dde07b4-8905-410d-bd60-801cc0640e87" />

```python
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
```
```python
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```
<img width="1018" height="598" alt="image" src="https://github.com/user-attachments/assets/6315b13b-e5a0-403f-9954-e72fe6f413b4" />

```python
sm.qqplot(df['Moderate Negative Skew_1'],line='45')
```
<img width="1058" height="592" alt="image" src="https://github.com/user-attachments/assets/1c161539-770a-44e0-85ad-23a4dc27738f" />

```python
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```
<img width="1002" height="596" alt="image" src="https://github.com/user-attachments/assets/c8f16279-738d-4c60-9f25-6d735e444eab" />

```python
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
```
<img width="1066" height="587" alt="image" src="https://github.com/user-attachments/assets/9ac0b47e-cce6-4fcd-a752-45825a3f1639" />

```python
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
```
<img width="936" height="577" alt="image" src="https://github.com/user-attachments/assets/364201a2-8dac-4f9d-8599-852ed1d1aa9a" />





# RESULT:
Thus the given data,Feature Encoding,Transformation process and save the data to a file was performed successfully.

       
