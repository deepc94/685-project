import json
import pandas as pd

adj_df = pd.read_csv('adjectives_list.csv')

with open('ai_prompts.txt', 'r') as f:
    ai_prompts = [l.strip() for l in f.readlines()]

with open('human_prompts.txt', 'r') as f:
    human_prompts = [l.strip() for l in f.readlines()]

print(f'Number of adjectives: {len(adj_df)}')
print(f'Number of AI prompts: {len(ai_prompts)}')
print(f'Number of human prompts: {len(human_prompts)}')

all_prompts_dict = dict()
for i, r in adj_df.iterrows():
    adj, abs_n = r['Comparative'], r['Abstract Noun']
    all_prompts_dict[adj] = dict()
    all_prompts_dict[adj]['abstract_noun'] = abs_n
    all_prompts_dict[adj]['ai'] = [p.format(adjective=adj, abs_noun=abs_n) for p in ai_prompts]
    all_prompts_dict[adj]['human'] = [p.format(adjective=adj, abs_noun=abs_n) for p in human_prompts]


with open('all_prompts.json', 'w') as f:
    json.dump(all_prompts_dict, f, indent=4)