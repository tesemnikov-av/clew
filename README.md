<img src="logo.png" width="150"/>

![contributors](https://img.shields.io/github/contributors/tesemnikov-av/pelevin-recomendation-bot) ![last-commit](https://img.shields.io/github/last-commit/tesemnikov-av/Pelevin-recomendation-bot) ![repo-size](https://img.shields.io/github/repo-size/tesemnikov-av/Pelevin-recomendation-bot)

![watch](https://img.shields.io/github/watchers/tesemnikov-av/Pelevin-recomendation-bot?style=social) 

# Clew : event log outlier detection

This application groups all text events up to the specified time and identifies abnormal events in the problem period

## Dependencies

- Python
- Huggingface
- Hdbscan
- Torch

# Usage

```bash
>> egrep -vi "Created slice|Removed slice|Started Session" /var/log/messages > /clew/messages

```
```python
>> event2vec = Event2Vec(len_timestamp=LEN_TIMESTAMP, date_time_str='2021-12-07 13:00:00')
>> event2vec.look(logfile="/clew/messages")
2021-12-07 13:52:19	<abnormal message>
```
