from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('/gpfs/u/home/LSMC/LSMCshnk/scratch/SparseGPT/SparseGPT-pytorch-1.12/checkpoints/32-5b-openllama-3b-v2-lr-mixed_shuffle_epoch_2/checkpoint/latest') #'/gpfs/u/home/LSMC/LSMCshnk/scratch/yiduo/TinyLlama')
model = AutoModelForCausalLM.from_pretrained('/gpfs/u/home/LSMC/LSMCshnk/scratch/SparseGPT/SparseGPT-pytorch-1.12/checkpoints/32-5b-openllama-3b-v2-lr-mixed_full_epoch_4/checkpoint/latest') #,device_map="cuda:0") #/gpfs/u/home/LSMC/LSMCshnk/scratch/yiduo/TinyLlama') #, device_map="cuda:3") #, torch_dtype=torch.float16)

# Encode input text
input_text = "Write me a poem about Machine Learning."
def get_ppl(input_text):
    input_ids = tokenizer(input_text,truncation=True, max_length=512, return_tensors="pt")

# Avoid running the model on 'cuda' as it depends on your system's configuration and might cause errors in this context
    input_ids = input_ids #.to("cuda")

# Get model outputs
    with torch.no_grad():
        outputs = model(input_ids["input_ids"], labels=input_ids["input_ids"])

    #print(outputs.loss) # Calculate negative log likelihood loss
    neg_log_likelihood = outputs.loss * input_ids["input_ids"].shape[1]

# Calculate perplexity
    return torch.exp(neg_log_likelihood / input_ids["input_ids"].shape[1])
ppl_sum=0 #print(perplexity)
import pandas as pd

# Read the Parquet file
df = pd.read_parquet('wiki_medical_terms.parquet')
import pdb
# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    ppl=get_ppl(row['page_text'])
    ppl_sum+=ppl
    print(index,ppl_sum,ppl)
    if index==199:
        pdb.set_trace()
with open('medical_term.txt','r') as f:
    for line_id,line in enumerate(f):
        #print(line_id,line)
        ppl=get_ppl(line)
        ppl_sum+=ppl
        print(ppl_sum,line_id,ppl)
