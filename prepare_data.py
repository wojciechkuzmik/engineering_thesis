import re
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from IPython.display import display


def prepare_data(filename, print_corr=False, return_as_df=False):
    df = pd.read_csv(filename)

    def matching_numbers(first, second):
        first_numbers = set(re.findall(r'[0-9]+', first))
        second_numbers = set(re.findall(r'[0-9]+', second))
        union = first_numbers.union(second_numbers)
        intersection = first_numbers.intersection(second_numbers)

        if len(first_numbers) == 0 and len(second_numbers) == 0:
            return 1
        else:
            return len(intersection) / len(union)

    def price_similarity(first, second):
        if float(first) > float(second):
            return 1.0 - (float(first) - float(second)) / float(first)
        else:
            return 1.0 - (float(second) - float(first)) / float(second)

    def engineer_features(data_frame):
        data_frame['first_title'] = data_frame['first_title'].str.lower()
        data_frame['second_title'] = data_frame['second_title'].str.lower()

        data_frame['brand_token_set_ratio'] = data_frame.apply(
            lambda x: fuzz.token_set_ratio(str(x['first_brand']).lower(),
                                           str(x['second_brand']).lower()) / 100, axis=1)

        data_frame['title_ratio'] = data_frame.apply(
            lambda x: fuzz.ratio(x['second_title'],
                                 x['first_title']) / 100, axis=1)

        data_frame['title_partial_ratio'] = data_frame.apply(
            lambda x: fuzz.partial_ratio(x['second_title'],
                                         x['first_title']) / 100, axis=1)

        data_frame['title_token_set_ratio'] = data_frame.apply(
            lambda x: fuzz.token_set_ratio(x['second_title'],
                                           x['first_title']) / 100, axis=1)

        data_frame['title_matching_numbers'] = data_frame.apply(
            lambda x: matching_numbers(x['second_title'],
                                       x['first_title']), axis=1)

        data_frame['title_log_fuzz_score'] = ((data_frame['title_ratio'] + data_frame['title_partial_ratio'] +
                                               data_frame['title_token_set_ratio']).apply(np.log)) / 100

        data_frame['price_similarity'] = data_frame.apply(
            lambda x: price_similarity(x['first_price'],
                                       x['second_price']), axis=1)

        data_frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        data_frame.fillna(value=0, inplace=True)

        return data_frame

    df = engineer_features(df)
    if print_corr:
        print("Pearson's correlations:")
        display(df[df.columns[1:]].corr()['label'][:].sort_values(ascending=False))
    if return_as_df:
        return df

    X = df[['title_token_set_ratio', 'title_partial_ratio', 'title_matching_numbers', 'title_ratio',
            'title_log_fuzz_score', 'brand_token_set_ratio', 'price_similarity']].values

    y = df['label'].values
    return X, y
