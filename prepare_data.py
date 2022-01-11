import re
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from IPython.display import display


def prepare_data(filename, print_corr=False):
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

    def engineer_features(data_frame):
        data_frame['first'] = data_frame['first'].str.lower()
        data_frame['second'] = data_frame['second'].str.lower()

        data_frame['ratio'] = data_frame.apply(
            lambda x: fuzz.ratio(x['second'],
                                 x['first']), axis=1)

        data_frame['partial_ratio'] = data_frame.apply(
            lambda x: fuzz.partial_ratio(x['second'],
                                         x['first']), axis=1)

        data_frame['token_sort_ratio'] = data_frame.apply(
            lambda x: fuzz.token_sort_ratio(x['second'],
                                            x['first']), axis=1)

        data_frame['token_set_ratio'] = data_frame.apply(
            lambda x: fuzz.token_set_ratio(x['second'],
                                           x['first']), axis=1)

        data_frame['matching_numbers'] = data_frame.apply(
            lambda x: matching_numbers(x['second'],
                                       x['first']), axis=1)

        data_frame['matching_numbers_log'] = (data_frame['matching_numbers'] + 1).apply(np.log)

        data_frame['log_fuzz_score'] = (data_frame['ratio'] + data_frame['partial_ratio'] +
                                        data_frame['token_sort_ratio'] + data_frame['token_set_ratio']).apply(np.log)

        data_frame['log_fuzz_score_numbers'] = data_frame['log_fuzz_score'] + (data_frame['matching_numbers']).apply(
            np.log)

        data_frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        data_frame.fillna(value=0, inplace=True)

        return data_frame

    df = engineer_features(df)
    if print_corr:
        print("Pearson's correlations:")
        display(df[df.columns[1:]].corr()['match'][:].sort_values(ascending=False))

    X = df[['token_set_ratio', 'log_fuzz_score_numbers', 'partial_ratio', 'matching_numbers_log', 'matching_numbers',
            'ratio', 'log_fuzz_score']].values
    y = df['match'].values

    return X, y
