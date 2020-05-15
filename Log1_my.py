
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
color = sns.color_palette()

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.mode.chained_assignment = None  # default='warn'


# In[2]:


order_products_prior_df = pd.read_csv('order_products__prior.csv')


# In[3]:


order_products_prior_df.head()


# In[4]:


order_products_prior_df.shape


# In[5]:


order_products_train_df = pd.read_csv('order_products__train.csv')


# In[6]:


order_products_train_df.head()


# In[7]:


order_products_train_df.shape


# In[8]:


orders_df = pd.read_csv('orders.csv')
orders_df.head()


# In[9]:


def get_unique_count(x):
    return len(np.unique(x))

cnt_srs = orders_df.groupby("eval_set")["user_id"].aggregate(get_unique_count)
cnt_srs


# In[10]:


aisles_df = pd.read_csv('aisles.csv')
departments_df = pd.read_csv('departments.csv')
products_df = pd.read_csv('products.csv')


# In[11]:


aisles_df.head()


# In[12]:


departments_df.head()


# In[13]:


products_df.head()


# In[14]:


cnt_srs = order_products_prior_df['product_id'].value_counts().reset_index().head(20)
cnt_srs.columns = ['product_id', 'frequency_count']
cnt_srs


# In[15]:


products_selected0 = products_df[products_df.aisle_id.isin(aisles_df.aisle_id)]
banana = products_selected0[products_selected0.product_id=="24852"]
banana


# In[16]:


cnt_srs['product_name'] = products_df[products_df.product_id.isin(cnt_srs.product_id)].product_name
cnt_srs = cnt_srs.iloc[:1,:]
cnt_srs


# In[17]:


order_products_prior_selected = order_products_prior_df[order_products_prior_df.product_id.isin(cnt_srs.product_id)]


# In[18]:


order_products_prior_selected.shape


# In[19]:


order_products_train_selected = order_products_train_df[order_products_train_df.product_id.isin(cnt_srs.product_id)]


# In[20]:


order_products_train_selected.shape


# In[21]:


orders_selected1 = pd.merge(orders_df,order_products_prior_selected, on='order_id')


# In[22]:


orders_selected1.shape


# In[23]:


orders_selected1.head()


# In[24]:


orders_selected2 = pd.merge(orders_df,order_products_train_selected, on='order_id')


# In[25]:


orders_selected2.shape


# In[26]:


orders_selected2.head()


# In[27]:


orders_selected = pd.concat([orders_selected1,orders_selected2]).sort_values('order_id')[['eval_set','order_id','product_id','user_id','order_dow','reordered','days_since_prior_order']]
orders_selected.sort_values('order_id',inplace=True)
orders_selected.head()


# In[28]:


orders_selected.shape


# In[29]:


orders_selected = orders_selected[orders_selected.product_id==24852].fillna(0).sort_values(by="days_since_prior_order")
orders_selected


# In[30]:


orders_selected['day'] = orders_selected.groupby('user_id').days_since_prior_order.cumsum()
unique_users = np.unique(orders_selected.user_id)
unique_users = unique_users[:100]
orders_selected = orders_selected[orders_selected.user_id.isin(unique_users)]
unique_users_N = len(unique_users)
unique_users


# In[31]:


tmp = np.memmap('Memmap', dtype='int32', mode='w+', shape=(int(unique_users_N*orders_selected['day'].max()),2))
user_id_index = 0
for day in range(int(orders_selected['day'].max())):
    tmp[day*unique_users_N:(day+1)*unique_users_N, :] = np.array([unique_users, [day]*unique_users_N]).T
tmp = pd.DataFrame(tmp,columns=['user_id','day'])


# In[32]:


df1 = pd.merge(orders_selected,tmp,on=['day','user_id'],how='outer')
df1


# In[33]:


df1.to_hdf('data.h5',key='df')


# In[251]:


df1 = pd.read_hdf('data.h5')


# In[252]:


df1.sort_values(by=['day','user_id','days_since_prior_order'],inplace=True)


# In[253]:


df1['days_since_prior_order1'] = df1.groupby(by='user_id').days_since_prior_order.fillna(method='ffill')


# In[254]:


df1.days_since_prior_order1.fillna(0,inplace=True)


# In[266]:


df1['ordered'] = (~df1.reordered.isna()).astype(int)


# In[267]:


c


# In[268]:


df1.order_dow.fillna(method='ffill',inplace=True)


# In[286]:


k10 = np.sum(df1.ordered==0)
k11 = np.sum(df1.ordered==1)
prob = k10/(k10+k11)
prob


# In[258]:


X_train = df1[['user_id','order_dow','days_since_prior_order1']]#[:int(len(df1)*0.7)]
y_train= df1['ordered']#[:int(len(df1)*0.7)].astype(int)


# In[241]:


X_train.fillna(0,inplace=True)


# In[125]:


y_train


# In[285]:


k10 = np.sum(y_train==0)
k11 = np.sum(y_train==1)
prob = k10/(k10+k11)
prob


# In[242]:


X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.30, random_state=42)


# In[127]:


X_train.shape, y_train.shape, y_train.sum() 


# In[143]:


clf = LogisticRegression(random_state=1).fit(X_train, y_train)


# In[144]:


y_pred = clf.predict(X_test)
#print(y_pred)
#print(clf.predict_proba(X_test))
print(clf.score(X_test, y_test))


# In[130]:


y_pred.shape


# In[131]:


y_pred


# In[132]:


from sklearn.metrics import f1_score


# In[ ]:


f1_score(y_test, y_pred, average='micro')


# In[146]:


from sklearn.metrics import recall_score
recall_score(y_test, y_pred, average='micro')


# In[147]:


from sklearn.metrics import precision_score
recall_score(y_test, y_pred, average='micro')


# In[117]:


y_train.shape, y_train.sum()


# In[118]:


y_test.shape, y_test.sum()


# In[119]:


y_pred.shape, y_pred.sum()


# In[53]:


X_train


# In[54]:


#LinearRegression


# In[55]:


X_train = df1[['user_id','order_dow','days_since_prior_order1']]#[:int(len(df1)*0.7)]
y_train= df1['ordered']#[:int(len(df1)*0.7)].astype(int)


# In[56]:


X_train.fillna(0,inplace=True)


# In[57]:


X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.30, random_state=42)


# In[58]:


X_train.shape, y_train.shape, y_train.sum() 


# In[59]:


from sklearn.linear_model import LinearRegression


# In[60]:


model = LinearRegression()


# In[61]:


model.fit(X_train, y_train)


# In[62]:


r_sq = model.score(X_train, y_train)
print('coefficient of determination:', r_sq)


# In[63]:


print('intercept:', model.intercept_)


# In[64]:


print('slope:', model.coef_)


# In[65]:


y_pred = model.predict(X_train)
print('predicted response:', y_pred, sep='\n')


# In[ ]:


# random forest


# In[269]:


X_train = df1[['user_id','order_dow','days_since_prior_order1']]#[:int(len(df1)*0.7)]
y_train= df1['ordered']#[:int(len(df1)*0.7)].astype(int)


# In[270]:


X_train.fillna(0,inplace=True)


# In[271]:


X_train, X_test, y_train, y_test = train_test_split(
X_train, y_train, test_size=0.30, random_state=42)


# In[272]:


from sklearn.ensemble import RandomForestClassifier


model = RandomForestClassifier(n_estimators=50, 
                               bootstrap = True,
                               max_features = 'sqrt')

model.fit(X_train, y_train)


# In[273]:


rf_predictions = model.predict(X_test)

rf_probs = model.predict_proba(X_test)[:, 1]


# In[274]:


rf_probs


# In[275]:


from sklearn import metrics

print(metrics.classification_report(y_test.reshape(len(y_test,)).astype(int), (rf_probs.reshape(len(y_test,))>0.2).astype(int)))


# In[280]:


k1 = np.sum(y_test==0)
k1


# In[281]:


k2 = np.sum(y_test==1)
k2


# In[150]:


# random forest_continue
RSEED = 50


# In[151]:


import seaborn as sns


# In[152]:


from sklearn.tree import DecisionTreeClassifier

# Make a decision tree and train
tree = DecisionTreeClassifier(random_state=RSEED)
tree.fit(X_train, y_train)


# In[153]:


print(f'Decision tree has {tree.tree_.node_count} nodes with maximum depth {tree.tree_.max_depth}.')


# In[180]:


print(f'Model Accuracy: {tree.score(X_train, y_train)}')


# In[155]:


# Limit maximum depth and train
short_tree = DecisionTreeClassifier(max_depth = 3, random_state=RSEED)
short_tree.fit(X_train, y_train)

print(f'Model Accuracy: {short_tree.score(X_train, y_train)}')


# In[181]:


from sklearn.metrics import f1_score


# In[182]:


f1_score(y_test, y_pred, average='macro')


# In[184]:


from sklearn.metrics import precision_score
recall_score(y_test, y_pred, average='micro')


# In[ ]:


#graphics


# In[149]:


import seaborn as sns
color = sns.color_palette()


# In[152]:


grouped = orders_df.groupby("order_id")["days_since_prior_order"].aggregate("sum").reset_index()
grouped = grouped.days_since_prior_order.value_counts()

from matplotlib.ticker import FormatStrFormatter
f, ax = plt.subplots(figsize=(15, 10))
sns.barplot(grouped.index, grouped.values)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
plt.ylabel('Number of orders', fontsize=13)
plt.xlabel('Period of reorder', fontsize=13)
plt.show()


# In[ ]:


#for banana graphics


# In[164]:


orders_selected00 = pd.concat([orders_selected1,orders_selected2]).sort_values('order_id')[['eval_set','order_id','product_id','user_id','order_dow','reordered','days_since_prior_order','add_to_cart_order']]
orders_selected00.sort_values('order_id',inplace=True)
orders_selected00.head()


# In[165]:


orders_selected00 = orders_selected00[orders_selected00.product_id==24852].fillna(0).sort_values(by="days_since_prior_order")
orders_selected00


# In[166]:


orders_selected00['day'] = orders_selected00.groupby('user_id').days_since_prior_order.cumsum()
unique_users = np.unique(orders_selected00.user_id)
unique_users = unique_users[:100]
orders_selected00 = orders_selected00[orders_selected00.user_id.isin(unique_users)]
unique_users_N = len(unique_users)
unique_users


# In[167]:


tmp = np.memmap('Memmap', dtype='int32', mode='w+', shape=(int(unique_users_N*orders_selected['day'].max()),2))
user_id_index = 0
for day in range(int(orders_selected['day'].max())):
    tmp[day*unique_users_N:(day+1)*unique_users_N, :] = np.array([unique_users, [day]*unique_users_N]).T
tmp = pd.DataFrame(tmp,columns=['user_id','day'])


# In[171]:


df0 = pd.merge(orders_selected00,tmp,on=['day','user_id'],how='outer')
df0


# In[169]:


df0 = pd.read_hdf('data.h5')


# In[172]:


plt.plot(df0.day.values,df0.add_to_cart_order.cumsum().values,'.')


# In[173]:


plt.plot(df0.day.values,df0.add_to_cart_order.values,'.')


# In[174]:


grouped = df0.groupby("order_id")["days_since_prior_order"].aggregate("sum").reset_index()
grouped = grouped.days_since_prior_order.value_counts()

from matplotlib.ticker import FormatStrFormatter
f, ax = plt.subplots(figsize=(15, 10))
sns.barplot(grouped.index, grouped.values)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
plt.ylabel('Number of orders', fontsize=13)
plt.xlabel('Period of reorder', fontsize=13)
plt.show()

