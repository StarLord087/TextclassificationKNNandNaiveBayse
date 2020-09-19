#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import nltk
import os
import string 
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords
from num2words import num2words
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
import math
import heapq
import re
import copy
import random
import pandas as pd
from unidecode import unidecode
from sklearn.manifold import TSNE
from statistics import mode, StatisticsError 
import gc


# In[3]:


gc.collect()


# In[4]:


ps = PorterStemmer()


# ### Loading Data

# In[5]:


path = 'C:\\Users\\shekh\\Desktop\\Shekhar\\IR\\Assignment 1\\20_newsgroups'
folders = []
files = []
docs = []
names = []

#loading folders
for folder in os.listdir(path):
    folders.append(folder)
    
#print(folders)

subset = ['comp.graphics', 'sci.med','talk.politics.misc', 'rec.sport.hockey', 'sci.space']

docsdic = {}
docsdicindex = {}

#loading files from each folder 
for folder in folders:
    if folder in subset:
        newpath = path +'\\'+folder
        for file in tqdm(os.listdir(newpath)):
            try:
                f = open(newpath+'\\'+file,"r")
                #print(doc)
                doc = f.read()
                #adding files to list
                docs.append(doc)
                if folder in docsdic:
                    docsdic[folder].append(doc)
                else:
                    docsdic[folder] = [doc]
                names.append((file,folder))
                f.close()
            except:
                pass
            
      


# In[6]:


def listtodict(keys, values):
    keys = keys.copy()
    values = values.copy()
    dic = {}
    for key in range(len(keys)): 
        for value in values: 
            dic[key] = value
            values.remove(value)
            break  
    return dic


# In[7]:


def remove_metadata(documents):
    pos = documents.index('\n\n')
    fixed_docs = documents[pos:]
    return fixed_docs


# In[8]:


def apply_proter_stemmer(string):
    for i in range(len(string)):
        string[i] = ps.stem(string[i])
    return string
        


# In[9]:


def remove_stopword(string):
    stop_words = set(stopwords.words('english'))
    data = [w for w in string if not w in stop_words]
    return data


# In[10]:


def convert_num(string):
    for i in range(len(string)):
        try:
            if(string[i].isnumeric()):
                string[i] = num2words(string[i])
        except:
            continue
    return string


# In[11]:


def remove_nonascii(string):
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)


# In[12]:


def preprocessing(newdataset):
    dataset = newdataset.copy()
    for i in tqdm(range(len(dataset))):
        dataset[i] = dataset[i].lower()
        dataset[i] = remove_nonascii(dataset[i])
        dataset[i] = dataset[i].translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
        dataset[i] = dataset[i].split()
        dataset[i] = remove_stopword(dataset[i])
        dataset[i] = convert_num(dataset[i])
        dataset[i] = apply_proter_stemmer(dataset[i])
    return dataset


# In[13]:


def querypreprocessing(stringdata):
    lower = stringdata.lower()
    punc = lower.translate(str.maketrans('','',string.punctuation))
    spl = punc.split()
    removestop = remove_stopword(spl)
    removenum = convert_num(removestop)
    stemmedquery = apply_proter_stemmer(removenum)
    
    for word in stemmedquery:
        if word not in vocab:
            stemmedquery.remove(word)
    
    return stemmedquery


# In[14]:


def create_vocabulary(data):
    vocab = set()
    for doc in data:
        for word in doc:
            #print(word)
            vocab.add(word)
    return list(vocab)


# In[15]:


def traintest_split(Dic, ratio):
    D = copy.deepcopy(Dic)
    traindic = {}
    trainindex = {}
    testindex = {}
    test = []
    traindata = []
    keys = list(D.keys())
    
    for key in tqdm(range(len(keys))):
        traindic[keys[key]] = random.sample(D[keys[key]],int(len(D[keys[key]])*(ratio/100)))
        traindata.extend(traindic[keys[key]])
        for data in D[keys[key]]:
            if data not in traindic[keys[key]]:
                test.append(data)
                if keys[key] in testindex:
                    testindex[keys[key]].append(D[keys[key]].index(data)+(key*1000))
                else:
                    testindex[keys[key]] = [D[keys[key]].index(data)+(key*1000)]
            else:
                if keys[key] in trainindex:
                    trainindex[keys[key]].append(D[keys[key]].index(data)+(key*1000))
                else:
                    trainindex[keys[key]] = [D[keys[key]].index(data)+(key*1000)]
                
    return traindic, test, trainindex,testindex, traindata


# In[16]:


def create_dataframe(vocab, data):
    df = pd.DataFrame()
    df['vocab'] = vocab
    
    
    tfall = []
    
    for i in tqdm(range(len(data))):
        tf1 = []
        for word in vocab:
            tf1.append(termfreq(data[i],word))
        #print(len(tf1))
        tfall.append(tf1)
    #print(tfall)
    
    for i in tqdm(range(len(tfall))):
        df.insert(i+1, i, tfall[i],allow_duplicates = False)
    
    df.set_index('vocab', inplace = True)
    
    return df


# In[17]:


def get_mutual_information(class_tf_idf_series, class_tf_df_list):
    words = class_tf_idf_series.index
    Mutual_Information = {}
    Mutual_Information_each_class = pd.DataFrame()
    
    class_tf = pd.concat(class_tf_df_list, axis=1)
    class_tf.replace(np.nan, 0, inplace=True)
#     display(class_tf[class_tf>0])
    
    for i, class_tf_1 in tqdm(enumerate(class_tf_df_list)):
        in_class_df = class_tf.iloc[:, i*len(class_tf_1.columns):(i+1)*len(class_tf_1.columns)]
        not_in_class_df = class_tf.drop(in_class_df.columns, axis=1)
        
        # number of docs containing term and present in the class
        n11 = in_class_df[in_class_df>0].count(axis=1)

        # number of docs present in the class which doesn't have the term
        n01 = in_class_df[in_class_df == 0].count(axis=1)

        # number of docs containing term but does not present in the class
        n10 = not_in_class_df[not_in_class_df>0].count(axis=1)

        # number of docs doesn't have the term and not present in class
        n00 = not_in_class_df[not_in_class_df == 0].count(axis=1)
        
        n = n11+n01+n10+n00
        a = (n11/n)*np.log2((n*n11)/((n10+n11)*(n01+n11)))
        b = (n01/n)*np.log2((n*n01)/((n00+n01)*(n01+n11)))
        c = (n10/n)*np.log2((n*n10)/((n10+n11)*(n10+n00)))
        d = (n00/n)*np.log2((n*n00)/((n01+n00)*(n00+n10)))
            
        a.replace(np.nan, 0, inplace=True)
        b.replace(np.nan, 0, inplace=True)
        c.replace(np.nan, 0, inplace=True)
        d.replace(np.nan, 0, inplace=True)
    
        MI = a+b+c+d
        Mutual_Information_each_class = pd.concat([Mutual_Information_each_class, MI], axis=1)

    Mutual_Information_each_class.columns = class_tf_idf_series.columns
        
    return Mutual_Information_each_class


# In[18]:


def termfreq(data, word):
    if data.count(word) == 0:
        return 0.0
    else:
        return 1+(math.log10(data.count(word)))
        #return(data.count(word)/len(data))


# In[19]:


def accuracy(docs, result, data):
    correct = 0
    
    for i,j in zip(docs,result):
        m = []
        if i in data[j]:
            correct += 1
            m.append(1)
        else:
            m.append(0)
        
    return(correct/len(result))


# In[20]:


def tfidf_feature_selection(data):
    termsinclass = []
    wholedic = {}
    for key in data.keys():
        terms = []
        for docs in data[key]:
            terms.extend(docs)
            wholedic[key] = terms
        termsinclass.append(terms)
   # print(len(termsinclass))
    
    vocab = create_vocabulary(termsinclass)
    
    df = pd.DataFrame(columns = list(data.keys()))
    
    df['vocab'] = vocab
    df.set_index('vocab', inplace = True)
    #print(vocab == df.index.values)
    #termfreq(wholedic, "")
    keys = list(wholedic.keys())
    #print(keys)
    for key in keys:
        series = []
        for term in tqdm(df.index.values):
            series.append(termfreq(wholedic.get(key), term))
        df.loc[:,key] = series
    
    nonzero = np.count_nonzero(df,axis = 1) 
    #print(nonzero[3])
    idf = []
    for i in tqdm(range(len(nonzero))):
        #print(nonzero[i])
        idf.append(1/(1+(math.log10(df.shape[1]/nonzero[i]))))
    
    df['idf'] = idf
    
    return df


# In[21]:


def l1(x):
    x_new = x/np.sqrt(np.sum(np.square(x)))
    return x_new 


# In[22]:


def create_tfidf(df1):
    df = df1.copy()
    df.iloc[:, 0:-1] = df.iloc[:, 0:-1].mul(df.iloc[:, -1], axis=0)
    #print(df)
    return df


# In[23]:


def select_top(dataframe, percent):
    featuresdataframe = dataframe.copy()
    if 'idf' in featuresdataframe.columns:
        featuresdataframe = featuresdataframe.drop(labels = 'idf', axis = 1)
    feat = []
    for column in featuresdataframe.columns:
        feat.append(featuresdataframe[column].sort_values(ascending = False, axis=0).index.tolist())
        #print(featuresdataframe[column].sort_values(ascending = False, axis=0).index)
    #print(len(feat[1]))
    
    for i in range(len(feat)):
        feat[i] = feat[i][:int(len(feat[i])*(percent/100))]
    #print(len(feat[1]))
    
    return feat


# In[205]:


def reduce_docs(features, dic):
    data = copy.deepcopy(dic)
#     for key in data.keys():
#         print(len(data[key][1]))
    
    for (i,key) in zip(features, data.keys()):
        
        newlist = []
        for j in data[key]:
            #print(i)
            new = []
            for term in j:
                if term in i:
                    new.append(term)
            newlist.append(new)
       # print(len(newlist))
        data[key] = newlist
                
    return data


# In[25]:


def confusion_matrix(D, result, testdata):
    compgraphic = []
    hockey = []
    scimed = []
    space = []
    politics = []
    
    keys = list(D.keys())
    #print(keys)
    
    for i in range(len(result)):
        if result[i] =='comp.graphics':
            compgraphic.append(i)
        elif result[i] == 'sci.med':
            scimed.append(i)
        elif result[i] == 'talk.politics.misc':
            politics.append(i)
        elif result[i] == 'sci.space':
            space.append(i)
        else:
            hockey.append(i)
    comp = np.zeros(5)
    hock = np.zeros(5)
    sci = np.zeros(5)
    spa = np.zeros(5)
    pol = np.zeros(5)
    for i in compgraphic:
        for j in range(len(keys)):
            if testdata[i] in D[keys[j]]:
                #print(keys[j])
                comp[j] += 1
    
    for i in hockey:
        for j in range(len(keys)):
            if testdata[i] in D[keys[j]]:
                #print(keys[j])
                hock[j] += 1

    for i in scimed:
        for j in range(len(keys)):
            if testdata[i] in D[keys[j]]:
                #print(keys[j])
                sci[j] += 1
                
    for i in space:
        for j in range(len(keys)):
            if testdata[i] in D[keys[j]]:
                #print(keys[j])
                spa[j] += 1
                
    for i in politics:
        for j in range(len(keys)):
            if testdata[i] in D[keys[j]]:
                #print(keys[j])
                pol[j] += 1

    confusion = np.array([comp, hock,sci,spa,pol], np.int32)
    
    confusiondf = pd.DataFrame(data = confusion, index = keys, columns = ['comp.graphics-pred', 'rec.sport.hockey-pred', 'sci.med-pred', 'sci.space-pred', 'talk.politics.misc-pred'])
    
    return confusiondf


# In[26]:


def polling(k, similarity, train_index):
    index = list(similarity.nlargest(k).index)
    selected = []
    
    for i in index:
        for key in train_index.keys():
            if i in train_index[key]:
                selected.append(key)
                
    try:
        result = mode(selected)
        return result
    except StatisticsError:
        return selected[0]


# In[27]:


def knn(train_df, test_df,k, train_index):
    train = copy.deepcopy(train_df)
    test = copy.deepcopy(test_df)
    
#     trainnormalised = train.apply(lambda col: l1(col),axis = 0)
#     testnormalised = test.apply(lambda col: l1(col),axis = 0)
    
    traintranspose = train.transpose()
#    print( test.shape,traintranspose.shape)
    similarities = traintranspose.dot(test)
    result = []
#     print(type(similarities[0]))
    for column in tqdm(similarities.columns):
        result.append(polling(k,similarities[column],train_index))
    
    return result


# In[28]:


docsdic.keys()


# In[29]:


idtoname = listtodict(docs, names)


# ### Performing text preproccesing on all docs in each class

# In[30]:


for key in docsdic.keys():
    docsdic[key] = preprocessing(docsdic[key])


# ### Splitting Data into Train and test set of specified ratios

# In[42]:


train_data_dic80, test_data20, train_index80,test_index20, train_data80 = traintest_split(docsdic, 80)


# In[58]:


train_data_dic70, test_data30, train_index70,test_index30, train_data70 = traintest_split(docsdic, 70)


# In[44]:


train_data_dic50, test_data50, train_index50,test_index50, train_data50 = traintest_split(docsdic, 50)


# In[60]:


# print(len(train_data80), len(train_data70), len(train_data50))
# print(len(test_data20),len(test_data30),len(test_data50))


# In[ ]:


#train_index80['sci.med']


# ## TF-IDF

# ### Calculating TF-IDF feature selection for each split of data
# please be patient this will take time 

# In[61]:


tfidffeatures80 = tfidf_feature_selection(train_data_dic80)


# In[62]:


tfidffeatures70 = tfidf_feature_selection(train_data_dic70)


# In[63]:


tfidffeatures50 = tfidf_feature_selection(train_data_dic50)


# In[64]:


tfidfdataframe80 = create_tfidf(tfidffeatures80)
tfidfdataframe70 = create_tfidf(tfidffeatures70)
tfidfdataframe50 = create_tfidf(tfidffeatures50)


# ### Selecting top 30% of the features

# In[65]:


selected_features80 = select_top(tfidfdataframe80, 30)
selected_features70 = select_top(tfidfdataframe70, 30)
selected_features50 = select_top(tfidfdataframe50, 30)


# ### Removing word from docs that were not selected in feature selection

# In[66]:


reducedtrain80 = reduce_docs(selected_features80, train_data_dic80)
reducedtrain70 = reduce_docs(selected_features70, train_data_dic70)
reducedtrain50 = reduce_docs(selected_features50, train_data_dic50)


# ### Creating vocabulary for each train set

# In[67]:


vocabuarytrain80 = []
vm = []
for key in reducedtrain80.keys():
    vm.extend(create_vocabulary(reducedtrain80[key]))
vocabuarytrain80 = list(set(vm))

vocabuarytrain70 = []
vm2 = []
for key in reducedtrain70.keys():
    vm2.extend(create_vocabulary(reducedtrain70[key]))
vocabuarytrain70 = list(set(vm2))

vocabuarytrain50 = []
vm3 = []
for key in reducedtrain50.keys():
    vm3.extend(create_vocabulary(reducedtrain50[key]))
vocabuarytrain50 = list(set(vm3))

print(len(vocabuarytrain80),len(vocabuarytrain70),len(vocabuarytrain50))


# In[69]:


reducedtraindata80 = []
for key in reducedtrain80.keys():
    for j in reducedtrain80[key]:
        reducedtraindata80.append(j)

reducedtraindata70 = []
for key in reducedtrain70.keys():
    for j in reducedtrain70[key]:
        reducedtraindata70.append(j)
        
reducedtraindata50 = []
for key in reducedtrain50.keys():
    for j in reducedtrain50[key]:
        reducedtraindata50.append(j)


# ### Creating Dataframe for KNN calculation

# In[70]:


tfdataframetrain80 = create_dataframe(vocabuarytrain80,reducedtraindata80)


# In[71]:


tfdataframetrain70 = create_dataframe(vocabuarytrain70,reducedtraindata70)


# In[72]:


tfdataframetrain50 = create_dataframe(vocabuarytrain50,reducedtraindata50)


# In[73]:


tfdataframetest20 = create_dataframe(vocabuarytrain80,test_data20)


# In[74]:


tfdataframetest30 = create_dataframe(vocabuarytrain70,test_data30)


# In[75]:


tfdataframetest50 = create_dataframe(vocabuarytrain50,test_data50)


# In[78]:


tftrainindex80 = []
for key in train_index80.keys():
    for i in train_index80[key]:
        tftrainindex80.append(i)
tftrainindex70 = []
for key in train_index70.keys():
    for i in train_index70[key]:
        tftrainindex70.append(i)
        
tftrainindex50 = []        
for key in train_index50.keys():
    for i in train_index50[key]:
        tftrainindex50.append(i)
#print(len(tftrainindex80),len(tftrainindex70),len(tftrainindex50))


# In[79]:


#print(len(tftrainindex80),len(tftrainindex70),len(tftrainindex50))


# In[80]:


tfdataframetrain80.columns = tftrainindex80
tfdataframetrain70.columns = tftrainindex70
tfdataframetrain50.columns = tftrainindex50


# ### Apply L1 norm to both Test DataFrame and Train DataFrame

# In[81]:


trainnormalised80 = tfdataframetrain80.apply(lambda col: l1(col),axis = 0)
testnormalised20 = tfdataframetest20.apply(lambda col: l1(col),axis = 0)
trainnormalised70 = tfdataframetrain70.apply(lambda col: l1(col),axis = 0)
testnormalised30 = tfdataframetest30.apply(lambda col: l1(col),axis = 0)
trainnormalised50 = tfdataframetrain50.apply(lambda col: l1(col),axis = 0)
testnormalised50 = tfdataframetest50.apply(lambda col: l1(col),axis = 0)


# In[82]:


print(type(test_data20))


# In[145]:


acculist8020 = []


# ### Run KNN on each split of data for K = 1,3,5
# This will only take a few seconds

# #### 80-20 Split

# In[146]:


result80k1 = knn(trainnormalised80,testnormalised20,1, train_index80)


# In[147]:


result80k3 = knn(trainnormalised80,testnormalised20,3, train_index80)


# In[148]:


result80k5 = knn(trainnormalised80,testnormalised20,5, train_index80)


# In[150]:


acculist8020.append(accuracy(test_data20,result80k1,docsdic))
acculist8020.append(accuracy(test_data20,result80k3,docsdic))
acculist8020.append(accuracy(test_data20,result80k5,docsdic))


# ### Print accuracy with Confusion Matrix

# In[151]:


print("K=1")
print("Accuracy with tfidf feature selection over 80-20 split",acculist8020[0])
confusion_matrix(docsdic,result80k1,test_data20)


# In[152]:


print("K=3")
print("Accuracy with tfidf feature selection over 80-20 split",acculist8020[1])
confusion_matrix(docsdic,result80k3,test_data20)


# In[153]:


print("K=5")
print("Accuracy with tfidf feature selection over 80-20 split",acculist8020[2])
confusion_matrix(docsdic,result80k5,test_data20)


# In[158]:


print(acculist8020)


# In[264]:


plt.plot([1,3,5],acculist8020)
plt.title("k vs Accuracy 80-20 TF-IDF feature")
plt.show()


# #### 70-30 Split

# In[154]:


acculist7030 = []


# In[155]:


result70k1 = knn(trainnormalised70,testnormalised30,1, train_index70)


# In[156]:


result70k3 = knn(trainnormalised70,testnormalised30,3, train_index70)


# In[157]:


result70k5 = knn(trainnormalised70,testnormalised30,5, train_index70)


# In[159]:


acculist7030.append(accuracy(test_data30,result70k1,docsdic))
acculist7030.append(accuracy(test_data30,result70k3,docsdic))
acculist7030.append(accuracy(test_data30,result70k5,docsdic))


# In[161]:


print("K=1")
print("Accuracy with tfidf feature selection over 70-30 split",acculist7030[0])
confusion_matrix(docsdic,result70k1,test_data30)


# In[162]:


print("K=3")
print("Accuracy with tfidf feature selection over 70-30 split",acculist7030[1])
confusion_matrix(docsdic,result70k3,test_data30)


# In[163]:


print("K=5")
print("Accuracy with tfidf feature selection over 70-30 split",acculist7030[2])
confusion_matrix(docsdic,result70k5,test_data30)


# In[164]:


print(acculist7030)


# In[263]:


plt.plot([1,3,5],acculist7030)
plt.title("k vs Accuracy 70-30 TF-IDF feature")
plt.show()


# #### 50-50 Split

# In[165]:


acculist5050 = []


# In[166]:


result50k1 = knn(trainnormalised50,testnormalised50,1, train_index50)


# In[167]:


result50k3 = knn(trainnormalised50,testnormalised50,3, train_index50)


# In[168]:


result50k5 = knn(trainnormalised50,testnormalised50,5, train_index50)


# In[169]:


acculist5050.append(accuracy(test_data50,result50k1,docsdic))
acculist5050.append(accuracy(test_data50,result50k3,docsdic))
acculist5050.append(accuracy(test_data50,result50k5,docsdic))


# In[262]:


print("K=1")
print("Accuracy with tfidf feature selection over 50-50 split",acculist5050[0])
confusion_matrix(docsdic,result50k1,test_data50)


# In[261]:


print("K=3")
print("Accuracy with tfidf feature selection over 50-50 split",acculist5050[1])
confusion_matrix(docsdic,result50k3,test_data50)


# In[260]:


print("K=5")
print("Accuracy with tfidf feature selection over 50-50 split",acculist5050[2])
confusion_matrix(docsdic,result50k5,test_data50)


# In[129]:


print(acculist5050)


# In[259]:


plt.plot([1,3,5],acculist5050)
plt.title("k vs Accuracy 50-50 TF-IDF feature")
plt.show()


# ### K vs Accuracy for each split of data , K = 1,3,5 with TF-IDF

# In[270]:


plt.plot([1,3,5],acculist7030, label = '70-30')
plt.plot([1,3,5],acculist8020, label = '80-20')
plt.plot([1,3,5],acculist5050, label = '50-50')
plt.legend()
plt.title("k vs Accuracy of all split TF-IDF feature")
plt.show()


# In[ ]:


# vocabularytrain80 = create_vocabulary(train_data80)


# In[179]:


vocabcomp = create_vocabulary(docsdic['comp.graphics'])
vocabhockey = create_vocabulary(docsdic['rec.sport.hockey'])
vocabscimed = create_vocabulary(docsdic['sci.med'])
vocabscispace = create_vocabulary(docsdic['sci.space'])
vocabpolitics = create_vocabulary(docsdic['talk.politics.misc'])


# In[180]:


dfclasscomp = create_dataframe(vocabcomp,docsdic['comp.graphics'])
dfclasshock = create_dataframe(vocabhockey,docsdic['rec.sport.hockey'])
dfclassmed = create_dataframe(vocabscimed,docsdic['sci.med'])
dfclassspace = create_dataframe(vocabscispace,docsdic['sci.space'])
dfclasspol = create_dataframe(vocabpolitics,docsdic['talk.politics.misc'])


# In[181]:


dflist = []
dflist.append(dfclasscomp)
dflist.append(dfclasshock)
dflist.append(dfclassmed)
dflist.append(dfclassspace)
dflist.append(dfclasspol)


# ## Mutual Information

# ### Calculating Mutual Information for each split of Train Data

# In[176]:


mitffeatures80 = copy.deepcopy(tfidffeatures80)
mitffeatures80 = mitffeatures80.drop(labels = 'idf', axis = 1)
mitffeatures70 = copy.deepcopy(tfidffeatures70)
mitffeatures70 = mitffeatures70.drop(labels = 'idf', axis = 1)
mitffeatures50 = copy.deepcopy(tfidffeatures50)
mitffeatures50 = mitffeatures50.drop(labels = 'idf', axis = 1)


# In[177]:


import warnings
warnings.filterwarnings("ignore")


# In[182]:


midf80 = get_mutual_information(mitffeatures80,dflist)


# In[189]:


midf70 = get_mutual_information(mitffeatures70, dflist)


# In[188]:


midf50 = get_mutual_information(mitffeatures50, dflist)


# ### Selecting top 30% of the word from the features

# In[195]:


midf80selected = select_top(midf80, 30)


# In[196]:


midf70selected = select_top(midf70, 30)


# In[197]:


midf50selected = select_top(midf50, 30)


# #### Removing words from the docs that were not selected in the feature selection

# In[206]:


reducedtrain80mi = reduce_docs(midf80selected, train_data_dic80)
# reducedtrain70 = reduce_docs(midf70selected, train_data_dic70)
# reducedtrain50 = reduce_docs(midf50selected, train_data_dic50)


# In[207]:


reducedtrain70mi = reduce_docs(midf70selected, train_data_dic70)


# In[208]:


reducedtrain50mi = reduce_docs(midf50selected, train_data_dic50)


# In[209]:


vocabuarytrain80mi = []
vmi = []
for key in reducedtrain80mi.keys():
    vmi.extend(create_vocabulary(reducedtrain80mi[key]))
vocabuarytrain80mi = list(set(vmi))

vocabuarytrain70mi = []
vmi2 = []
for key in reducedtrain70mi.keys():
    vmi2.extend(create_vocabulary(reducedtrain70mi[key]))
vocabuarytrain70mi = list(set(vmi2))

vocabuarytrain50mi = []
vmi3 = []
for key in reducedtrain50.keys():
    vmi3.extend(create_vocabulary(reducedtrain50mi[key]))
vocabuarytrain50mi = list(set(vmi3))

#print(len(vocabuarytrain80mi),len(vocabuarytrain70mi),len(vocabuarytrain50mi))


# In[211]:


reducedtraindata80mi = []
for key in reducedtrain80mi.keys():
    for j in reducedtrain80mi[key]:
        reducedtraindata80mi.append(j)

reducedtraindata70mi = []
for key in reducedtrain70mi.keys():
    for j in reducedtrain70mi[key]:
        reducedtraindata70mi.append(j)
        
reducedtraindata50mi = []
for key in reducedtrain50mi.keys():
    for j in reducedtrain50mi[key]:
        reducedtraindata50mi.append(j)


# ### Create Test and Train Dataframe for KNN Calculation
# this take only a few minutes

# In[212]:


tfdataframetrain80mi = create_dataframe(vocabuarytrain80mi,reducedtraindata80mi)


# In[213]:


tfdataframetrain70mi = create_dataframe(vocabuarytrain70mi,reducedtraindata70mi)


# In[214]:


tfdataframetrain50mi = create_dataframe(vocabuarytrain50mi,reducedtraindata50mi)


# In[215]:


tfdataframetest20mi = create_dataframe(vocabuarytrain80mi,test_data20)


# In[216]:


tfdataframetest30mi = create_dataframe(vocabuarytrain70mi,test_data30)


# In[217]:


tfdataframetest50mi = create_dataframe(vocabuarytrain50mi,test_data50)


# In[218]:


tftrainindex80mi = []
for key in train_index80.keys():
    for i in train_index80[key]:
        tftrainindex80mi.append(i)
tftrainindex70mi = []
for key in train_index70.keys():
    for i in train_index70[key]:
        tftrainindex70mi.append(i)
        
tftrainindex50mi = []        
for key in train_index50.keys():
    for i in train_index50[key]:
        tftrainindex50mi.append(i)
#print(len(tftrainindex80mi),len(tftrainindex70mi),len(tftrainindex50mi))


# In[219]:


tfdataframetrain80mi.columns = tftrainindex80mi
tfdataframetrain70mi.columns = tftrainindex70mi
tfdataframetrain50mi.columns = tftrainindex50mi


# ### Applying L1 norm to both Test and Train DataFrames

# In[221]:


trainnormalised80mi = tfdataframetrain80mi.apply(lambda col: l1(col),axis = 0)
testnormalised20mi = tfdataframetest20mi.apply(lambda col: l1(col),axis = 0)
trainnormalised70mi = tfdataframetrain70mi.apply(lambda col: l1(col),axis = 0)
testnormalised30mi = tfdataframetest30mi.apply(lambda col: l1(col),axis = 0)
trainnormalised50mi = tfdataframetrain50mi.apply(lambda col: l1(col),axis = 0)
testnormalised50mi = tfdataframetest50mi.apply(lambda col: l1(col),axis = 0)


# ### Calculate KNN with MI feature selection for each split of Data for K = 1,3,5

# #### 80-20 split

# In[222]:


acculist8020mi = []


# In[225]:


result80mik1 = knn(trainnormalised80mi,testnormalised20mi,1, train_index80)


# In[226]:


result80mik3 = knn(trainnormalised80mi,testnormalised20mi,3, train_index80)


# In[227]:


result80mik5 = knn(trainnormalised80mi,testnormalised20mi,5, train_index80)


# ### Calculating Accuracy

# In[229]:


acculist8020mi.append(accuracy(test_data20,result80mik1,docsdic))
acculist8020mi.append(accuracy(test_data20,result80mik3,docsdic))
acculist8020mi.append(accuracy(test_data20,result80mik5,docsdic))


# In[231]:


print("K=1")
print("Accuracy with MI feature selection over 80-20 split",acculist8020mi[0])
confusion_matrix(docsdic,result80mik1,test_data20)


# In[232]:


print("K=3")
print("Accuracy with MI feature selection over 80-20 split",acculist8020mi[1])
confusion_matrix(docsdic,result80mik3,test_data20)


# In[234]:


print("K=5")
print("Accuracy with MI feature selection over 80-20 split",acculist8020mi[2])
confusion_matrix(docsdic,result80mik5,test_data20)


# #### K vs Accuracy for 80-20 split

# In[251]:


plt.plot([1,3,5],acculist8020mi)
plt.title("k vs Accuracy 80-20 MI")
plt.show()


# #### 70-30 split

# In[235]:


acculist7030mi = []


# In[236]:


result70mik1 = knn(trainnormalised70mi,testnormalised30mi,1, train_index70)
result70mik3 = knn(trainnormalised70mi,testnormalised30mi,3, train_index70)
result70mik5 = knn(trainnormalised70mi,testnormalised30mi,5, train_index70)


# ### Calculate Accuracy 70-30

# In[237]:


acculist7030mi.append(accuracy(test_data30,result70mik1,docsdic))
acculist7030mi.append(accuracy(test_data30,result70mik3,docsdic))
acculist7030mi.append(accuracy(test_data30,result70mik5,docsdic))


# In[238]:


print("K=1")
print("Accuracy with MI feature selection over 70-30 split",acculist7030mi[0])
confusion_matrix(docsdic,result70mik1,test_data30)


# In[239]:


print("K=3")
print("Accuracy with MI feature selection over 70-30 split",acculist7030mi[1])
confusion_matrix(docsdic,result70mik3,test_data30)


# In[240]:


print("K=5")
print("Accuracy with MI feature selection over 70-30 split",acculist7030mi[2])
confusion_matrix(docsdic,result70mik5,test_data30)


# ### K vs Accuracy for 70-30 split MI K = 1,3,5

# In[250]:


plt.plot([1,3,5],acculist7030mi)
plt.title("k vs Accuracy 70-30 MI")
plt.show()


# #### 50-50 split

# In[242]:


acculist5050mi = []


# In[241]:


result50mik1 = knn(trainnormalised50mi,testnormalised50mi,1, train_index50)
result50mik3 = knn(trainnormalised50mi,testnormalised50mi,3, train_index50)
result50mik5 = knn(trainnormalised50mi,testnormalised50mi,5, train_index50)


# ### Calculate Accuracy for 50-50 split

# In[243]:


acculist5050mi.append(accuracy(test_data50,result50mik1,docsdic))
acculist5050mi.append(accuracy(test_data50,result50mik3,docsdic))
acculist5050mi.append(accuracy(test_data50,result50mik5,docsdic))


# In[244]:


print("K=1")
print("Accuracy with MI feature selection over 50-50 split",acculist5050mi[0])
confusion_matrix(docsdic,result50mik1,test_data50)


# In[247]:


print("K=3")
print("Accuracy with MI feature selection over 50-50 split",acculist5050mi[1])
confusion_matrix(docsdic,result50mik3,test_data50)


# In[246]:


print("K=5")
print("Accuracy with MI feature selection over 50-50 split",acculist5050mi[2])
confusion_matrix(docsdic,result50mik5,test_data50)


# ### K vs Accuracy for 50-50 split , K = 1,3,5

# In[252]:


plt.plot([1,3,5],acculist5050mi)
plt.title("k vs Accuracy 50-50 MI")
plt.show()


# ### K vs Accuracy for all split of data with MI

# In[265]:


plt.plot([1,3,5],acculist7030mi, label = '7030')
plt.plot([1,3,5],acculist8020mi, label = '8020')
plt.plot([1,3,5],acculist5050mi, label = '5050')
plt.legend()
plt.title("k vs Accuracy of all data split MI")
plt.show()


# ### K vs Accuracy for 80-20 set for TF-IDF and MI

# In[267]:


plt.plot([1,3,5],acculist8020, label = '8020-TF')
plt.plot([1,3,5],acculist8020mi, label = '8020-MI')
plt.legend()
plt.title("k vs Accuracy 80-20 MI vs tf")
plt.show()


# ### K vs Accuracy for 70-30 set for TF-IDF and MI

# In[268]:


plt.plot([1,3,5],acculist7030, label = '7030-TF')
plt.plot([1,3,5],acculist7030mi, label = '7030-MI')
plt.legend()
plt.title("k vs Accuracy 70-30 MI vs tf")
plt.show()


# ### K vs Accuracy for 50-50 set for TF-IDF and MI

# In[269]:


plt.plot([1,3,5],acculist5050, label = '5050-TF')
plt.plot([1,3,5],acculist5050mi, label = '5050-MI')
plt.legend()
plt.title("k vs Accuracy 50-50 MI vs tf")
plt.show()

