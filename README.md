# inotebook

## Kaggle
## 1.1 Intro to Machine Learning

A decision tree algorithm is a machine learning algorithm that uses a decision tree to make predictions. 

It follows a tree-like model of decisions and their possible consequences. The algorithm works by recursively splitting the data into subsets based on the most significant feature at each node of the tree.
![image.png](attachment:image.png)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data  # özellikler
y = iris.target  # hedef değişken

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar Ağacı sınıflandırıcısını tanımla ve eğit
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Test seti üzerinde tahmin yap
y_pred = clf.predict(X_test)

# Doğruluk (accuracy) hesapla
accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk (Accuracy):", accuracy)

## 1.2 Underfitting & Overfitting

The model is built according to the patterns obtained from the training data.
the model can overlearn or underlearn.
In this case, the model will not be able to make adequate predictions and the error rate in our predictions will be high.


## 1.2.1 Underfitting

Modelin verilerdeki temel örüntüleri yakalamak için çok basit olması ve bu nedenle kötü performans göstermesi 

## 1.2.2 Overfitting

Model, eğitim için kullanılan veri seti üzerinde gereğinden fazla çalışıp ezber yapmaya başlamışsa ya da eğitim seti tek düze ise overfitting olma riski büyük demektir.
![uo.PNG](attachment:uo.PNG)
## 1.3 Pandas

## 1.3.1 iloc & loc

Loc komutu ile etiket kullananarak verilere ulaşırken, iloc komutunda satır ve sütün index numarası ile verilere ulaşır, Yani loc komutunu kullanırken satır yada kolon ismi belirtirken, iloc komutunda satır yada sütünün index numarasını belirtmez.

## 1.4 Feature Engineering 

The goal of feature engineering is simply to make your data better suited to the problem at hand.

![1666105956546](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/b819196a-316e-4802-a2ac-b0b190d33901)

- It is the process of selecting, manipulating, and transforming raw data into features that can be used in supervised learning and preparing the data for machine learning models. 

It is the process of using domain knowledge to create features or input variables that help machine learning algorithms perform better.

It helps to increase the predictive power of the model.

improve a model's predictive performance
reduce computational or data needs
improve interpretability of the results

## 1.4.1 linear model 

describe a continuous response variable as a function of one or more predictor variables
![image.png](attachment:image.png)
## 1.5 Mutual Information

Mutual information from the field of information theory is the application of information acquisition to feature selection.

It is calculated between two variables and measures the reduction in uncertainty for one variable given a known value of the other variable.

Mutual information is a lot like correlation in that it measures a relationship between two quantities. The advantage of mutual information is that it can detect any kind of relationship, while correlation only detects linear relationships.

Mutual information describes relationships in terms of uncertainty.

![image.png](attachment:image.png)
Left: Mutual information increases as the dependence between feature and target becomes tighter. 

Right: Mutual information can capture any kind of association (not just linear, like correlation.
## 1.6 Outlier Detection

Outlier detection is the process of detecting outliers, or a data point that is far away from the average, and depending on what you are trying to accomplish, potentially removing or resolving them from the analysis to prevent any potential skewing

![not22.PNG](attachment:not22.PNG)
## 1.7 K-Means Algoritması

![k1](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/dbd81a8d-94f8-4d61-b5c4-e1b370f4b821)

K-Means Clustering is an Unsupervised Machine Learning algorithm, which groups the unlabeled dataset into different clusters. 

![k1.PNG](attachment:k1.PNG)
![k2.PNG](attachment:k2.PNG)
## 1.8 Deep Learning

Deep Learning is a machine learning method. 

It enables the training of artificial intelligence to predict outputs with a given data set.

Both supervised and unsupervised learning can be used to train AI.

## 1.9 Data Cleaning



### 1.9.1 Scaling

Scaling (ölçeklendirme) verilerin aralığını değiştirir. 
!pip show seaborn
# https://medium.com/@meritshot/standardization-v-s-normalization-6f93225fbd84#:~:text=Standardization%2C%20interestingly%2C%20refers%20to%20setting,data%20onto%20the%20unit%20sphere.

import numpy as np
# import minmax_scaling
from mlxtend.preprocessing import minmax_scaling
#import plt
import matplotlib.pyplot as plt
# import mixtend
import seaborn as sns

# generate 1000 data points randomly drawn from an exponential distribution
original_data = np.random.exponential(size=1000)

# mix-max scale the data between 0 and 1
scaled_data = minmax_scaling(original_data, columns=[0])

# plot both together to compare
fig, ax = plt.subplots(1, 2, figsize=(15, 3))
sns.histplot(original_data, ax=ax[0], kde=True, legend=False)
ax[0].set_title("Original Data")
sns.histplot(scaled_data, ax=ax[1], kde=True, legend=False)
ax[1].set_title("Scaled data")
plt.show()
scaled_data[0:2]
pip install mlxtend
![s1.PNG](attachment:s1.PNG)

### 1.9.2 Normalization (normalleştirme) 

Verilerin dağıtım şeklini değiştirir.Verileri 0-1 arasına sıkıştırır
# import  stats
from scipy import stats
 
# normalize the exponential data with boxcox
normalized_data = stats.boxcox(original_data)

# plot both together to compare
fig, ax=plt.subplots(1, 2, figsize=(15, 3))
sns.histplot(original_data, ax=ax[0], kde=True, legend=False)
ax[0].set_title("Original Data")
sns.histplot(normalized_data[0], ax=ax[1], kde=True, legend=False)
ax[1].set_title("Normalized data")
plt.show()
![image.png](attachment:image.png)
### 1.9.3 Embedding : Bir dilin veya verilen verideki kelimelerin tek tek, daha az boyutlu bir uzayda gerçek değerli vektörler olarak ifade edilmesidir.

Bir nesneyi başka bir nesnenin içine gömmek, yerleştirmek.

![Embedding](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*SYiW1MUZul1NvL1kc1RxwQ.png)
# LLM Bootcamp

![llm1](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/c6d6b628-4384-4487-a848-e28d29d70214)

![llm2](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/8e780dce-d81d-43d6-9bf4-013649d901d3)

![llm3](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/db4b9412-b616-4063-9159-17c83556ef2b)

![llm4](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/11f23374-376e-4a76-80ca-9e5eb89683ad)

## Zero Shot

![Zero Shot](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/4031544d-bc1c-4d7b-9679-6065d74b1a08)

## One-hot Encoding

![One-hot Encoding](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/707300af-61b3-4631-b1d5-5bef9b86f52f)

## Few-Shot Learning

![image](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/1a0870be-b41c-4d15-9bef-bed132df511d)

prompt = f"""
Your task is to answer in a consistent style.

<child>: Teach me about patience.

<grandparent>: The river that carves the deepest \ 
valley flows from a modest spring; the \ 
grandest symphony originates from a single note; \ 
the most intricate tapestry begins with a solitary thread.

<child>: Teach me about resilience.
"""
response = get_completion(prompt)
print(response)
# ChatGPT Prompt Engineering for Developers 

![t1](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/9d96ea8c-5633-4041-94ac-af7d86325178)

Delimiters can be anything like: ```, """, < >, <tag> </tag>
text = f"""
You should express what you want a model to do by \ 
providing instructions that are as clear and \ 
specific as you can possibly make them. \ 
This will guide the model towards the desired output, \ 
and reduce the chances of receiving irrelevant \ 
or incorrect responses. Don't confuse writing a \ 
clear prompt with writing a short prompt. \ 
In many cases, longer prompts provide more clarity \ 
and context for the model, which can lead to \ 
more detailed and relevant outputs.
"""
prompt = f"""
Summarize the text delimited by triple backticks \ 
into a single sentence.
```{text}```
"""
response = get_completion(prompt)
print(response)


## LLM Bootcamp

![image.png](attachment:image.png)
![image.png](attachment:image.png)
![image.png](attachment:image.png)
![image.png](attachment:image.png)

## 1.1 Zero Shot

![Zero Shot](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/4031544d-bc1c-4d7b-9679-6065d74b1a08)

## 1.2 One-hot Encoding

![One-hot Encoding](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/707300af-61b3-4631-b1d5-5bef9b86f52f)
## 1.3 Few-Shot Learning

![image](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/1a0870be-b41c-4d15-9bef-bed132df511d)
