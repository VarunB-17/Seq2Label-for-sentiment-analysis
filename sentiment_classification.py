from dataframe import *

data = DatasetSentiment(val=False)
batches = DynamicBatch(dataset=data,
                       max_tokens=MAX_TOKENS,
                       batches=BATCHES,
                       buckets=BUCKETS)
