from tqdm import tqdm
import logging
import re
import dateparser
import string
import pandas as pd
import configparser
import datetime

import hdbscan

import numpy as np
import torch
import transformers as ppb

config = configparser.ConfigParser()
config.read('clew.conf')

logger = logging.getLogger('joyce-app')

DEBUG = True
LEN_TIMESTAMP = 15

if DEBUG == 'True':
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

# PATH = config['JOYCE']['PATH']

# Load pretrained DistilBERT model/tokenizer
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+=', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\d', '', text)
    return text

class Event2Vec:
    """
    len_timestamp:
        /var/log/messages - 15

    date_time_str:
        '05/12/21 03:00' Example: if problem in 12 Dec 03:24'
    """
    def __init__(self, len_timestamp: int, date_time_str='05/12/21 03:00', \
                 tokenizer=tokenizer, model=model):
        
        self.len_timestamp = len_timestamp
        self.tokenizer = tokenizer
        self.model = model
        self.date_end = dateparser.parse(date_time_str)
        self.date_start = self.date_end - datetime.timedelta(days=2)
        self.features = None
        self.max_len = 40

    
    def look(self, logfile: str = 'andarta5_messages.txt'):
        # egrep -vi "Created slice|Removed slice|Started Session"
        self.raw_data = open(logfile).readlines()
        self.len_hostname = len(self.raw_data[0][self.len_timestamp + 1:].split()[0])
        self.df = pd.DataFrame(self.raw_data, columns=['text'])

        self.df['clean']  = self.df.text.str[self.len_timestamp + self.len_hostname:].apply(lambda x:clean_text(x))
        self.date_event_raw = [x[:self.len_timestamp] for x in self.raw_data]
        self.df['date'] = [dateparser.parse(x) for x in self.date_event_raw]
        self.mask = (self.df['date'] > self.date_start) & (self.df['date'] <= self.date_end)
        self.mask_x = ( (self.df['date'] > self.date_end) & (self.df['date'] <= (self.date_end + datetime.timedelta(hours=1))))
        self.df_previous = self.df.loc[self.mask]
        self.df_x = self.df.loc[self.mask_x]

        self.epochs = [self.df_previous[x:x+500] for x in range(0, len(self.df_previous), 500)]

        for epoch in tqdm(self.epochs):

            self.tokenized = epoch['clean'].apply(lambda x: self.tokenizer.encode(x, add_special_tokens=True))
            input_ids, attention_mask = self._get_attention_mask(self.tokenized)

            with torch.no_grad():
                last_hidden_states = self.model(input_ids, attention_mask=attention_mask)
            
            if self.features is None:
                self.features = last_hidden_states[0][:,0,:].numpy()
            else:
                self.features = np.append(self.features, last_hidden_states[0][:,0,:].numpy(), 0)

        self.cluster_model = hdbscan.HDBSCAN(min_cluster_size=2,
                          metric='euclidean',                      
                          cluster_selection_method='eom', prediction_data=True).fit(self.features)

        self.df_previous['labels'] = self.cluster_model.labels_


        self.tokenized_x = self.df_x['clean'].apply(lambda x: self.tokenizer.encode(x, add_special_tokens=True))
        
        input_ids, attention_mask = self._get_attention_mask(self.tokenized_x)

        with torch.no_grad():
            last_hidden_states = self.model(input_ids, attention_mask=attention_mask)
        
        self.df_x['labels'] = hdbscan.approximate_predict(self.cluster_model, last_hidden_states[0][:,0,:].numpy())[0]
        self.df_x[self.df_x.labels == -1][['date', 'text']].to_csv('anomaly')
        return self.df_x[self.df_x.labels == -1][['date', 'text']]

    def _get_attention_mask(self, tokenized):

        padded = np.array([i + [0]*(self.max_len-len(i)) for i in tokenized.values])

        attention_mask = np.where(padded != 0, 1, 0)
        input_ids = torch.tensor(padded[:])  
        attention_mask = torch.tensor(attention_mask[:])
        
        return input_ids, attention_mask
