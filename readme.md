# 제주대학교 인공지능 수업 개인 프로젝트

## 주제 : 다이아몬드 가격 예측

### 참고 페이지 

https://www.kaggle.com/shrutimechlearn/types-of-regression-and-stats-in-depth

https://github.com/yungbyun/mllib \ prediction_util.py



### 코드

```python
import pandas as pd
import numpy as np
from mllib.prediction_util import PredictionUtil
```


```python
dia = PredictionUtil()
```

> 데이터셋 
>
> carat : 다이아몬드의 캐럿 무게 (0.2 - 5.01)
>
> cut : 절단 품질
>
> color :  J(최악)에서 D(최고)까지의 다이아몬드 색상
>
> clarity : 다이아몬드의 투명도 측정( I1 < SI2 < SI1 < VS2 < VS1 < VVS2 < VVS1 < IF)
>
> depth : 총 깊이 백분율
>
> table : 가장 넓은 지점을 기준으로 다이아몬드 상단의 테이블 너비
>
> price : 가격
>
> x : 길이 (mm)
>
> y : 너비 (mm)
>
> z : 깊이 (mm)


```python
df = dia.read('diamonds.csv')
```

       Unnamed: 0  carat        cut color clarity  depth  table  price     x  \
    0           1   0.23      Ideal     E     SI2   61.5   55.0    326  3.95   
    1           2   0.21    Premium     E     SI1   59.8   61.0    326  3.89   
    2           3   0.23       Good     E     VS1   56.9   65.0    327  4.05   
    3           4   0.29    Premium     I     VS2   62.4   58.0    334  4.20   
    4           5   0.31       Good     J     SI2   63.3   58.0    335  4.34   
    5           6   0.24  Very Good     J    VVS2   62.8   57.0    336  3.94   
    6           7   0.24  Very Good     I    VVS1   62.3   57.0    336  3.95   
    7           8   0.26  Very Good     H     SI1   61.9   55.0    337  4.07   
    8           9   0.22       Fair     E     VS2   65.1   61.0    337  3.87   
    9          10   0.23  Very Good     H     VS1   59.4   61.0    338  4.00   
    
          y     z  
    0  3.98  2.43  
    1  3.84  2.31  
    2  4.07  2.31  
    3  4.23  2.63  
    4  4.35  2.75  
    5  3.96  2.48  
    6  3.98  2.47  
    7  4.11  2.53  
    8  3.78  2.49  
    9  4.05  2.39  



>  필요없는 칼럼 지우기

```python
df = dia.drop(['Unnamed: 0'])
```



> 유니크한 컬럼 개수


```python
dia.show_unique_column()
```

    carat : 273
    cut : 5
    color : 7
    clarity : 8
    depth : 184
    table : 127
    price : 11602
    x : 554
    y : 552
    z : 375



> heatmap으로 상관관계 파악

> 가장 강한 관련을 가지고 있는 컬럼 : carat
>
> 가격과 밀접한 관련이 있는 컬럼 : x, y, z
>
> 가격과 관련이 없는 컬럼 : depth, table

```python
col = ['z','y','x','price','table','depth','carat']
dia.heatmap(col)
```

![output_5_0](.\readme_image\output_5_0.png)



> 데이터셋 통계


```python
dia.df.describe()
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
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>price</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>53940.000000</td>
      <td>53940.000000</td>
      <td>53940.000000</td>
      <td>53940.000000</td>
      <td>53940.000000</td>
      <td>53940.000000</td>
      <td>53940.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.797940</td>
      <td>61.749405</td>
      <td>57.457184</td>
      <td>3932.799722</td>
      <td>5.731157</td>
      <td>5.734526</td>
      <td>3.538734</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.474011</td>
      <td>1.432621</td>
      <td>2.234491</td>
      <td>3989.439738</td>
      <td>1.121761</td>
      <td>1.142135</td>
      <td>0.705699</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.200000</td>
      <td>43.000000</td>
      <td>43.000000</td>
      <td>326.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.400000</td>
      <td>61.000000</td>
      <td>56.000000</td>
      <td>950.000000</td>
      <td>4.710000</td>
      <td>4.720000</td>
      <td>2.910000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.700000</td>
      <td>61.800000</td>
      <td>57.000000</td>
      <td>2401.000000</td>
      <td>5.700000</td>
      <td>5.710000</td>
      <td>3.530000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.040000</td>
      <td>62.500000</td>
      <td>59.000000</td>
      <td>5324.250000</td>
      <td>6.540000</td>
      <td>6.540000</td>
      <td>4.040000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.010000</td>
      <td>79.000000</td>
      <td>95.000000</td>
      <td>18823.000000</td>
      <td>10.740000</td>
      <td>58.900000</td>
      <td>31.800000</td>
    </tr>
  </tbody>
</table>
</div>



> 데이터셋 null값 확인


```python
print("x = 0: {}개 ".format((dia.df.x==0).sum()))
print("y = 0: {}개 ".format((dia.df.y==0).sum()))
print("z = 0: {}개 ".format((dia.df.z==0).sum()))
print("depth = 0: {}개 ".format((dia.df.depth==0).sum()))
```

    x = 0: 8개 
    y = 0: 7개 
    z = 0: 20개 
    depth = 0: 0개 

> 데이터셋  null값 삭제

```python
dia.df[['x','y','z']] = dia.df[['x','y','z']].replace(0,np.NaN)
```


```python
dia.df.isnull().sum()
```


    carat       0
    cut         0
    color       0
    clarity     0
    depth       0
    table       0
    price       0
    x           8
    y           7
    z          20
    dtype: int64


```python
dia.df.dropna(inplace=True)
```


```python
dia.df.shape
```


    (53920, 10)


```python
dia.df.isnull().sum()
```


    carat      0
    cut        0
    color      0
    clarity    0
    depth      0
    table      0
    price      0
    x          0
    y          0
    z          0
    dtype: int64



> boxplot


```python
dia.boxplot('cut','price')
```


![output_13_0](.\readme_image\output_13_0.png)

```python
dia.boxplot('color','price')
```


![output_14_0](.\readme_image\output_14_0.png)

```python
dia.boxplot('clarity','price')
```


![output_15_0](.\readme_image\output_15_0.png)



> 숫자형이 아닌 데이터 one_hot_encoding으로 수치화

```python
one_hot_df = pd.get_dummies(dia.df)
```


```python
one_hot_df.head()
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
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>price</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>cut_Fair</th>
      <th>cut_Good</th>
      <th>cut_Ideal</th>
      <th>...</th>
      <th>color_I</th>
      <th>color_J</th>
      <th>clarity_I1</th>
      <th>clarity_IF</th>
      <th>clarity_SI1</th>
      <th>clarity_SI2</th>
      <th>clarity_VS1</th>
      <th>clarity_VS2</th>
      <th>clarity_VVS1</th>
      <th>clarity_VVS2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.23</td>
      <td>61.5</td>
      <td>55.0</td>
      <td>326</td>
      <td>3.95</td>
      <td>3.98</td>
      <td>2.43</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.21</td>
      <td>59.8</td>
      <td>61.0</td>
      <td>326</td>
      <td>3.89</td>
      <td>3.84</td>
      <td>2.31</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.23</td>
      <td>56.9</td>
      <td>65.0</td>
      <td>327</td>
      <td>4.05</td>
      <td>4.07</td>
      <td>2.31</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.29</td>
      <td>62.4</td>
      <td>58.0</td>
      <td>334</td>
      <td>4.20</td>
      <td>4.23</td>
      <td>2.63</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.31</td>
      <td>63.3</td>
      <td>58.0</td>
      <td>335</td>
      <td>4.34</td>
      <td>4.35</td>
      <td>2.75</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>


```python
cols = one_hot_df.columns
dia.df = pd.DataFrame(one_hot_df,columns= cols)
```


```python
dia.df.head()
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
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>price</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>cut_Fair</th>
      <th>cut_Good</th>
      <th>cut_Ideal</th>
      <th>...</th>
      <th>color_I</th>
      <th>color_J</th>
      <th>clarity_I1</th>
      <th>clarity_IF</th>
      <th>clarity_SI1</th>
      <th>clarity_SI2</th>
      <th>clarity_VS1</th>
      <th>clarity_VS2</th>
      <th>clarity_VVS1</th>
      <th>clarity_VVS2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.23</td>
      <td>61.5</td>
      <td>55.0</td>
      <td>326</td>
      <td>3.95</td>
      <td>3.98</td>
      <td>2.43</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.21</td>
      <td>59.8</td>
      <td>61.0</td>
      <td>326</td>
      <td>3.89</td>
      <td>3.84</td>
      <td>2.31</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.23</td>
      <td>56.9</td>
      <td>65.0</td>
      <td>327</td>
      <td>4.05</td>
      <td>4.07</td>
      <td>2.31</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.29</td>
      <td>62.4</td>
      <td>58.0</td>
      <td>334</td>
      <td>4.20</td>
      <td>4.23</td>
      <td>2.63</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.31</td>
      <td>63.3</td>
      <td>58.0</td>
      <td>335</td>
      <td>4.34</td>
      <td>4.35</td>
      <td>2.75</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>



> 데이터 전처리
>
> 평균이 0과 표준편차가 1이 되도록 변환


```python
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
temp =  pd.DataFrame(sc_X.fit_transform(dia.df[['carat','depth','x','y','z','table']]),columns=['carat','depth','x','y','z','table'],index=dia.df.index)
```


```python
temp.head()
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
      <th>carat</th>
      <th>depth</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>table</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.198204</td>
      <td>-0.174203</td>
      <td>-1.591573</td>
      <td>-1.539219</td>
      <td>-1.580084</td>
      <td>-1.099725</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.240417</td>
      <td>-1.361090</td>
      <td>-1.645173</td>
      <td>-1.662014</td>
      <td>-1.750896</td>
      <td>1.585988</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.198204</td>
      <td>-3.385781</td>
      <td>-1.502241</td>
      <td>-1.460280</td>
      <td>-1.750896</td>
      <td>3.376463</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.071566</td>
      <td>0.454149</td>
      <td>-1.368242</td>
      <td>-1.319943</td>
      <td>-1.295396</td>
      <td>0.243131</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.029353</td>
      <td>1.082501</td>
      <td>-1.243176</td>
      <td>-1.214690</td>
      <td>-1.124583</td>
      <td>0.243131</td>
    </tr>
  </tbody>
</table>
</div>


```python
dia.df[['carat','depth','x','y','z','table']] = temp[['carat','depth','x','y','z','table']]
```


```python
dia.df.head()
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
      <th>carat</th>
      <th>depth</th>
      <th>table</th>
      <th>price</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>cut_Fair</th>
      <th>cut_Good</th>
      <th>cut_Ideal</th>
      <th>...</th>
      <th>color_I</th>
      <th>color_J</th>
      <th>clarity_I1</th>
      <th>clarity_IF</th>
      <th>clarity_SI1</th>
      <th>clarity_SI2</th>
      <th>clarity_VS1</th>
      <th>clarity_VS2</th>
      <th>clarity_VVS1</th>
      <th>clarity_VVS2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.198204</td>
      <td>-0.174203</td>
      <td>-1.099725</td>
      <td>326</td>
      <td>-1.591573</td>
      <td>-1.539219</td>
      <td>-1.580084</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.240417</td>
      <td>-1.361090</td>
      <td>1.585988</td>
      <td>326</td>
      <td>-1.645173</td>
      <td>-1.662014</td>
      <td>-1.750896</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.198204</td>
      <td>-3.385781</td>
      <td>3.376463</td>
      <td>327</td>
      <td>-1.502241</td>
      <td>-1.460280</td>
      <td>-1.750896</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.071566</td>
      <td>0.454149</td>
      <td>0.243131</td>
      <td>334</td>
      <td>-1.368242</td>
      <td>-1.319943</td>
      <td>-1.295396</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.029353</td>
      <td>1.082501</td>
      <td>0.243131</td>
      <td>335</td>
      <td>-1.243176</td>
      <td>-1.214690</td>
      <td>-1.124583</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>



> 데이터 전처리가 끝난 후 전체 칼럼에 대한 heatmap


```python
dia.heatmap(dia.df.columns)
```


![output_24_0](.\readme_image\output_24_0.png)



> 예측모델 정확도 확인

1) 가장 연관성이 높은 4가지 컬럼으로 가격 예측

```python
dia.run_all(['carat','x','y','z'],'price')
```

    [ 6092.27544782 11662.85095528  3096.41064734 ...   622.14507178
      8153.80395129  6958.8699804 ] 
     (10784,)
    LR - 0.851
    K-NR - 0.877
    [12725.  10477.   2800.5 ...  1050.   6884.   6620. ] 
     (10784,)
    DT - 0.754
    Random F. - 0.863

논리회귀 알고리즘 : 85.1%

근접이웃 알고리즘 : 87.7%

결정트리 알고리즘 : 75.4%

랜덤포레스트 알고리즘 : 86.3%

근접이웃 , 랜덤포레스트 , 논리회귀 , 결정트리 알고리즘 순으로 정확도가 높이 측정됨



2) 모든 컬럼들로 가격 예측

```python
dia.run_all(['carat', 'depth', 'table', 'x', 'y', 'z', 'cut_Fair', 'cut_Good',
       'cut_Ideal', 'cut_Premium', 'cut_Very Good', 'color_D', 'color_E',
       'color_F', 'color_G', 'color_H', 'color_I', 'color_J', 'clarity_I1',
       'clarity_IF', 'clarity_SI1', 'clarity_SI2', 'clarity_VS1',
       'clarity_VS2', 'clarity_VVS1', 'clarity_VVS2'],'price')
```

    [ 5673.30424535 10716.48086467  4088.415403   ...  1138.56737294
      9510.38267357  7620.7706207 ] 
     (10784,)
    LR - 0.919
    K-NR - 0.959
    [ 6328.  8457.  3759. ...  1033. 10255.  6963.] 
     (10784,)
    DT - 0.961
    Random F. - 0.979

논리회귀 알고리즘 : 91.9%

근접이웃 알고리즘 : 95.9%

결정트리 알고리즘 : 96.1%

랜덤포레스트 알고리즘 : 97.9%

랜덤포레스트,결정트리,근접이웃,논리회귀 알고리즘 순으로 정확도가 높이 측정됨

또한, 연관성이 높은 4가지 컬럼을 빼서 예측한 것보다 모든 칼럼을 넣어 예측할때가 더 높은 정확도로 측정됨