import numpy as np
import pandas as pd

df_original = pd.read_csv('data\pricerunner_aggregate.csv',
                          names=['product_id', 'product_title', 'vendor_id', 'cluster_id',
                                 'cluster_label', 'category_id', 'category_label'])
print("total count:")
print(len(df_original[df_original["category_label"] == "Mobile Phones"]))
df_correct = df_original[['product_title', 'cluster_label', 'category_label']].copy()
df_correct.rename(columns={'product_title': 'title',
                           'cluster_label': 'label'}, inplace=True)

df_correct['match'] = 1

def create_synthetic_data(df, iterations):
    df_output = df
    i = 1
    while i <= iterations:
        df_s = df[['title', 'label', 'category_label']].copy()
        df_s['shuffled_label'] = df_s['label']
        df_s['shuffled_label'] = df_s.groupby('category_label')['label'].transform(np.random.permutation)

        df_s['match'] = np.where(df_s['label'] == df_s['shuffled_label'], 1, 0)

        df_s['label'] = np.where(df_s['shuffled_label'] != '',
                                 df_s['shuffled_label'],
                                 df_s['label'])

        df_output = df_output.append(df_s)
        df_output = df_output.drop(columns=['shuffled_label'])

        i += 1

    return df_output


df_output = create_synthetic_data(df_correct, 4)
df_output = df_output[df_output['category_label'] == 'Mobile Phones']
df_output = df_output.drop(columns='category_label')
print("matching pairs count:")
print(len(df_output[df_output["match"] == 1]))
print("not matching pairs count:")
print(len(df_output[df_output["match"] == 0]))
df_output.rename(columns={'title': 'first',
                           'label': 'second'}, inplace=True)
df_output.to_csv('data\learning_data.csv', index=False)





# def create_synthetic_data(df, iterations):
#     df_output = df
#
#     i = 1
#     while i <= iterations:
#         df_s = df[['title', 'label', 'category_label']].copy()
#         df_s['shuffled_label'] = df_s['label']
#         df_s['shuffled_label'] = df_s.groupby('category_label')['label'].transform(np.random.permutation)
#
#         df_s['match'] = np.where(df_s['label'] == df_s['shuffled_label'], 1, 0)
#
#         df_s['label'] = np.where(df_s['shuffled_label'] != '',
#                                  df_s['shuffled_label'],
#                                  df_s['label'])
#
#         df_output = df_output.append(df_s)
#         df_output = df_output.drop(columns=['shuffled_label'])
#
#         i += 1
#
#     return df_output
#
#
# df_output = create_synthetic_data(df_correct, 5)
# df_output = df_output[df_output['category_label'] == 'Mobile Phones']
# df_output = df_output.drop(columns='category_label')
# df_output.rename(columns={'title': 'first',
#                            'label': 'second'}, inplace=True)
# df_output.to_csv('data\learning_data.csv', index=False)
