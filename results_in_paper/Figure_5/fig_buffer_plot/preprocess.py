import pickle
import numpy as np


# process the data for the final loss
for type_index in range(7):
    list_value_loss = []
    for i in range(1,9):
        with open(f'./record_data/{i}_loss_final', 'rb') as f:
            loaded_file = pickle.load(f)
        if i ==1:
            for j in range(1, len(loaded_file)+1):
                list_value_loss.append(loaded_file[j][type_index])
        else:
            for j in range(0, len(loaded_file)):
                list_value_loss.append(loaded_file[j][type_index])

    with open(f'./processed_data/loss_all_tasks_type_index_{type_index}.pkl', 'wb') as flist:
        pickle.dump(list_value_loss, flist)

#-----------------------------------------------------------------------------------





# # loop for model update
# for model in ["plastic", "stable"]:
#     list_value_random = []
#     for i in range(1,9):
#         with open(f'./record_data/{i}_{model}_model_update_random', 'rb') as f:
#             loaded_file = pickle.load(f)
#         for j in range(len(loaded_file)):
#             list_value_random.append(loaded_file[j].numpy())

#     print(len(list_value_random))
#     with open(f'./processed_data/model_update_random_value_all_tasks_{model}.pkl', 'wb') as flist:
#         pickle.dump(list_value_random, flist)




#---------------------------------------------------------------------------------


# for buffer_name in ["gss", "reservoir"]:
#     list_value_task_labels = []
#     for i in range(1,9):
#         with open(f'./record_data/{i}_{buffer_name}_buffer_task_id', 'rb') as f:
#             loaded_file = pickle.load(f)
#         if i ==1:
#             for j in range(1, len(loaded_file)+1):
#                 list_value_task_labels.append(loaded_file[j])
#         else:
#             for j in range(0, len(loaded_file)):
#                 list_value_task_labels.append(loaded_file[j])


#     print(len(list_value_task_labels))

#     with open(f'./processed_data/buffer_task_id_all_tasks_{buffer_name}.pkl', 'wb') as flist:
#         pickle.dump(list_value_task_labels, flist)