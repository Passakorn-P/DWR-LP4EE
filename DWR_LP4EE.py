import numpy as np
import pandas as pd
import itertools

from sklearn.preprocessing import MinMaxScaler

from LP4EE_Regularized import LP4EE_Regularized

class DWR_LP4EE:

    @staticmethod
    def predict(X_train, y_train, X_test, alpha = [0.001, 0.01, 0.1, 1, 10, 100, 1000], l1_ratio = [0.5]):

        df_train = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train).rename(columns={0: 'Effort'})], axis=1)
        df_single_test = pd.concat([pd.DataFrame(X_test), pd.DataFrame(np.zeros(X_test.shape[0])).rename(columns={0: 'Effort'})], axis=1)
        df_train.columns = df_train.columns.map(str)
        df_single_test.columns = df_single_test.columns.map(str)

        param_combinations = [(alp, l1) for alp in alpha for l1 in l1_ratio]
        all_evaluate_combinations = list(itertools.product(range(df_train.shape[0]), param_combinations))

        to_expand_results = [DWR_LP4EE.inner_evaluate(df_train, evaluate_combination=evaluate_combination) for evaluate_combination in all_evaluate_combinations]
        expanded_results = [s[0] for s in to_expand_results]
        result_df = pd.concat([pd.DataFrame(expanded_results, columns=['mae']), pd.DataFrame(all_evaluate_combinations, columns=['index', 'alpha'])], axis=1)
        result_df = result_df.pivot(index='index', columns='alpha', values='mae')

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        raw_counts = result_df.eq(result_df.min(axis=1), axis=0).cumsum(axis=1).idxmax(axis=1)
        distances = np.linalg.norm(X_train_scaled - X_test_scaled, axis=1)
        ranks = distances.argsort().argsort() + 1
        n = len(distances)
        polynomial_weights = ((n - ranks + 1) ** 2)
        rank = pd.Series(polynomial_weights, name='weight')
        counts = pd.concat([raw_counts, rank], axis=1)
        counts = counts.groupby(0)['weight'].sum().reindex(result_df.columns, fill_value=0)
        counts /= counts.sum()

        final_combination = list(itertools.product([0], param_combinations))
        to_expand_results = [DWR_LP4EE.inner_evaluate(pd.concat([df_single_test, df_train]), evaluate_combination=evaluate_combination) for evaluate_combination in final_combination]
        preds = [s[1] for s in to_expand_results]
        pred = np.array(preds).T[0].dot(counts)

        return pred

    @staticmethod
    def inner_evaluate(df_train, evaluate_combination):
        sub_train = df_train.drop(evaluate_combination[0], axis=0)
        sub_test = df_train.iloc[[evaluate_combination[0]]]

        X_train, y_train = sub_train.drop('Effort', axis=1).values, sub_train['Effort'].values
        x_test, y_test = sub_test.drop('Effort', axis=1).values, sub_test['Effort'].values

        model = LP4EE_Regularized(regularization='EN', alpha=evaluate_combination[1][0], l1_ratio=evaluate_combination[1][1])
        model.fit(X_train, y_train)
        pred = model.predict(x_test)
        mae = np.abs(pred - y_test)
        return mae, pred
