
# In[1]:
get_ipython().system('pip install --upgrade pandas')


# In[2]:
import pandas as pd
import numpy as np
import boto3
import re
import sagemaker
from sagemaker import get_execution_role
from sklearn.model_selection import train_test_split


# In[3]:

#Additional imports
from sagemaker.predictor import csv_serializer
from sagemaker.amazon.amazon_estimator import get_image_uri


# In[4]:
data = pd.read_csv('final_df.csv')


# In[5]:
data.head()


# In[6]:
data['target'] = data.Label.apply(lambda x: 0 if x=='benign' else 1)


# In[10]:
char_map = {chr(i+97): i+1 for i in range(26)} ##For all the alphabets
char_map['.'] = 27
char_map['_'] = 28
char_map['-'] = 29


# In[12]:


count = 30
##For all the digits
for i in range(10):
    char_map[str(i)] = count
    count+=1


# In[13]:


char_map[' '] = 40


# In[18]:


data['length'] = data.DNS.apply(lambda x: len(x))


# In[19]:


max_length = max(data.length)


# In[22]:


##Creates encoded vector 
def encodeDomain(domain, length):
    vector = np.full(shape= length, fill_value=50) #Creating new array
    for i in range(len(domain)):
        if domain[i] not in char_map:
            print('Invalid key ', domain[i])
        else:
            vector[i] = char_map[domain[i]] 
    return vector


# In[23]:

#Testing
encodeDomain('facebook.com', 40)


# In[62]:


data = data.sample(frac=1).reset_index(drop=True)


# In[63]:


embedding = data.DNS.apply(lambda x : encodeDomain(x, max_length))


# In[64]:


x = np.array(embedding.tolist())
y = np.array(data.Label)


# In[65]:


x


# In[66]:


y


# In[67]:


X = pd.DataFrame(x)
Y = pd.DataFrame(y)


# In[68]:


df = pd.concat([Y,X], axis = 1) #Concat along column


# In[69]:


df.head()


# In[70]:


df.shape


# In[71]:


df.to_csv('train_data.csv', index = False, header = False)


# In[72]:


bucket_name = 'imt575-xgboost'
data_file_key = 'train_data.csv'

s3_model_output_location = r's3://{0}/model'.format(bucket_name)
s3_data_file_location = r's3://{0}/{1}'.format(bucket_name, data_file_key)


# In[73]:


def write_to_s3(filename, bucket, key):
    with open(filename,'rb') as f:
        return boto3.Session().resource('s3').Bucket(bucket).Object(key).upload_fileobj(f)


# In[74]:


write_to_s3('train_data.csv', bucket_name, data_file_key)


# In[51]:



# In[52]:


role = get_execution_role()


# In[53]:

print(role)


# In[54]:

#(insert AWS-ID and secret key)
client = boto3.client('s3', aws_access_key_id = aws_id, 
                      aws_secret_access_key = aws_secret)


# In[75]:


s3_input_train = sagemaker.s3_input(s3_data='s3://{}/train_data.csv'.format(bucket_name), content_type='csv')


# In[78]:


sess = sagemaker.Session()

estimator = sagemaker.estimator.Estimator(containers[boto3.Session().region_name],
                                         role,
                                         train_instance_count = 1,
                                         train_instance_type = 'ml.m5.large',
                                         output_path = s3_model_output_location,
                                         sagemaker_session = sess,
                                         base_job_name = 'xgboost-v1')

estimator.set_hyperparameters(max_depth = 3,
                              eta = .5,
                              gamma = 4,
                              min_child_weight = 6,
                              silent = 0,
                              objective = "binary:logistic", #reg:logistic
                              num_round = 1000) #10

estimator.fit({'train' : s3_input_train})


# predicted = 0.0632267817855
# facebook.com
