import pandas as pd
import numpy as np


def prepare_dataframes(df_info_filename):
    df_left = pd.read_csv("data/tableA.csv", index_col=0)
    df_right = pd.read_csv("data/tableB.csv", index_col=0)
    df_info = pd.read_csv(df_info_filename)

    df_left = df_left[["title", "brand", "price"]].loc[df_info["ltable_id"]]
    df_right = df_right[["title", "brand", "price"]].loc[df_info["rtable_id"]]

    df_left.rename(columns={'title': 'first_title',
                            'brand': 'first_brand',
                            'price': 'first_price'}, inplace=True)
    df_right.rename(columns={'title': 'second_title',
                             'brand': 'second_brand',
                             'price': 'second_price'}, inplace=True)
    df_left.reset_index(inplace=True)
    df_right.reset_index(inplace=True)
    new_df = pd.concat([df_left, df_right], axis=1)
    new_df['label'] = df_info['label'].values
    new_df.drop(columns="id", inplace=True)
    return new_df


df = pd.DataFrame(columns=["first_title", "first_brand", "first_price",
                           "second_title", "second_brand", "second_price", "label"])
df = df.append(prepare_dataframes("data/train.csv"))
df = df.append(prepare_dataframes("data/test.csv"))
df = df.append(prepare_dataframes("data/valid.csv"))
df = df.dropna()
print("Dataframe:")
print(df.head(12))
print(f"\n\nAll pairs count: {len(df)}")
matching_count = len(df[df['label'] == 1])
print(f"Matching pairs count: {matching_count}")
not_matching_count = len(df[df['label'] == 0])
print(f"Not matching pairs count: {not_matching_count}")
print(f"Matching/Not matching pairs ratio: {matching_count/not_matching_count}")
print(df.loc[(df["first_brand"] != df["second_brand"]) & (df["label"] == 1)])
df.to_csv("data/data.csv", index=False)

