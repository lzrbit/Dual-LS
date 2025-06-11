import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
import pandas as pd

buffer_name = "reservoir"
# buffer_name = "gss"


with open(f'./processed_data/8_{buffer_name}_buffer_memory', 'rb') as f:
    loaded_file = pickle.load(f)

# print((loaded_file[1]))




replay_buffer = []

for i in range(len(loaded_file)):
    replay_buffer.append(loaded_file[i])

replay_buffer = np.array(replay_buffer)
print(replay_buffer.shape)




# rank the replay buffer for each row
replay_buffer_ranked = np.zeros(replay_buffer.shape)
for i in tqdm(range(replay_buffer.shape[0]), desc="Ranking rows"):
    buffer_line = replay_buffer[i]
    replay_buffer_ranked[i] = buffer_line[np.argsort(buffer_line)]


# build a dict 
my_dict = {}
frequency_all = []
down_sample = 1
all_num = replay_buffer.shape[0]

for i in tqdm(range(0,all_num, down_sample), desc="Calculating frequency"):
    # first calculate the frequency of each number in the row
    frequency = np.zeros(9)
    for j in range(9):
        frequency[j] = np.sum(replay_buffer_ranked[i]==j)
    frequency_all.append(frequency)
    
# convert the list to a numpy array
frequency_all = np.array(frequency_all)
print(frequency_all.shape)

bar_num = frequency_all.shape[0]
for i in range(bar_num):
    my_dict[i] = frequency_all[i]

# plot the my_dict with sns.barplot

df = pd.DataFrame(my_dict)

# extract 1-8 rows one by one with loop
for i in  range(1, 9):
    df_temp = df.iloc[i].astype(int)
    index = df_temp[df_temp != 0].index[0]
    df_temp = df_temp[index:]
    # print(df_temp.size)
    # print the mean
    # print('Task:', i, ' Max number is ', df_temp.max())
    print('Task:', i, ' Mean number is ', df_temp.mean())
    # save the df_temp to a csv file
    df_temp.to_csv(f'./extracted_csv/{buffer_name}_Task_{i}.csv', index=False)

print("All tasks are saved to csv files.")