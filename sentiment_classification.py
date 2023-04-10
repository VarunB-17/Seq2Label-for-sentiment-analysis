from dataloader import *


data = DatasetSentiment(val=False)
batches = DynamicBatchLoader(dataset=data,
                             max_tokens=MAX_TOKENS,
                             batch_size=BATCHES)

for batch in batches:

    for seq,label in batch:
        print(seq.size())