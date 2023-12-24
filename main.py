import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import statsmodels.api as sm
import numpy as np
from scipy.stats import somersd
import matplotlib.pyplot as plt

columns_of_interest = ['listing_date', 'week_price_change', 'month_price_change',
                       'week_volume_change', 'month_volume_change', 'rank', 'N_of_listings',
                       'month_delta_listings', 'mean_volume', 'class']
columns_decision_score = ['cut_off', 'TP', 'TN', 'FN', 'FP', 'Recall', 'Precision', 'F_Score', 'MISC']
columns_cut_off_best_row_merge = ['n_train', 'cut_off', 'TP', 'TN', 'FN', 'FP', 'Recall', 'Precision', 'F_Score',
                                  'misc']
columns_betas_merge = ['n_train', 'best_cutoff', 'intercept', 'week_price_change', 'month_price_change',
                       'week_volume_change', 'month_volume_change', 'rank', 'N_of_listings', 'mean_volume',
                       'month_delta_listings']
columns_score_test_merge = ['n_train', 'best_cutoff', 'listing_date', 'symbol', 'week_price_change',
                            'month_price_change', 'week_volume_change', 'month_volume_change', 'rank', 'N_of_listings',
                            'month_delta_listings', 'mean_volume', 'P0', 'P1']
columns_real_predictions = ['n_train', 'best_cutoff', 'listing_date', 'symbol', 'P0', 'P1', 'Real_class',
                            'Forecast_class']


def data_quality_analysis(df):
    result_data = []
    for column in columns_of_interest:
        mean_value = df[column].mean()
        std_dev_value = df[column].std()
        min_value = df[column].min()
        max_value = df[column].max()
        n_value = df[column].count()
        n_miss_value = df[column].isnull().sum()

        result_data.append({
            'Variable': column,
            'Mean': mean_value,
            'Std Dev': std_dev_value,
            'Minimum': min_value,
            'Maximum': max_value,
            'N': n_value,
            'N Miss': n_miss_value
        })
    analysis_results = pd.DataFrame(result_data)
    print(analysis_results)

    excel_output_path = 'analysis_results(table).xlsx'
    analysis_results.to_excel(excel_output_path, index=False)
    print(f"Результати аналізу записані у файл: {excel_output_path}")


def task_2(train_df):
    train_df['rank'] = train_df['rank'].fillna(1500).astype(int)
    train_df['week_price_change'] = train_df['week_price_change'].astype(int)
    train_df['month_price_change'] = train_df['month_price_change'].astype(int)
    train_df['week_volume_change'] = train_df['week_volume_change'].astype(int)
    train_df['month_volume_change'] = train_df['month_volume_change'].astype(int)
    train_df['month_volume_change'] = train_df['month_volume_change'].astype(int)
    train_df['month_delta_listings'] = train_df['month_delta_listings'].astype(int)
    train_df['mean_volume'] = train_df['mean_volume'].astype(int)

    train_df = train_df.sort_values(by='listing_date')

    excel_output_path = 'FULL_SET.xlsx'
    train_df.to_excel(excel_output_path, index=False)
    print(f"Пропуски замінені та дані приведені до цілих чисел. Результати записані у файл: {excel_output_path}")
    return train_df


def training_dataset(df):
    X = df.drop(['symbol', 'listing_date', 'class'], axis=1)
    Y = df['class']
    return X[:30], Y[:30]


def MLE(features, target):
    data = sm.add_constant(features)
    model = sm.Logit(target, data).fit()

    print(model.summary())
    print(model.wald_test_terms(scalar=True))


def odds_ratio(features, target):
    data = sm.add_constant(features)
    model = sm.Logit(target, data).fit()
    coefficients = model.params

    confidence_intervals = model.conf_int()
    confidence_intervals['Odds Ratio'] = coefficients

    confidence_intervals.columns = ['5%', '95%', 'Point Estimate']
    print(np.exp(confidence_intervals))


def concordance(features: pd.DataFrame, target: pd.Series):
    data = sm.add_constant(features)
    model = sm.Logit(target, data).fit()

    observed, predicted = target.values, model.predict(sm.add_constant(features)).values

    df = pd.DataFrame({'Observed': observed, 'Predicted': predicted})

    ones = df[df['Observed'] == target.max()]
    zeros = df[df['Observed'] == target.min()]

    concord = (ones['Predicted'].values[:, np.newaxis] > zeros['Predicted'].values).sum()
    discord = (ones['Predicted'].values[:, np.newaxis] < zeros['Predicted'].values).sum()

    total_pairs = len(ones) * len(zeros)

    concordance = concord / total_pairs
    discordance = discord / total_pairs
    ties_percent = 1 - concordance - discordance
    sommers_d = somersd(observed, predicted).statistic

    n = len(observed)
    tau_a = (concord - discord) / (n * (n - 1) / 2)
    gamma = (concord - discord) / (concord + discord)

    result = {
        'Percent Concordant': round(100 * concordance, 2),
        'Percent Discordant': round(100 * discordance, 2),
        'Percent Tied': round(ties_percent, 4),
        'Pairs': total_pairs,
        'Somers D': round(sommers_d, 3),
        'Gamma': round(gamma, 3),
        'Tau-a': round(tau_a, 3),
        'c': round(concordance, 3)
    }

    for key, value in result.items():
        print(f'{key}: {value}')


def apply_cutoff_probability(data, cutoff):
    for i in range(len(data)):
        if data.loc[i, 'F_Probability'] >= cutoff:
            data.at[i, 'F_Probability'] = 1
        else:
            data.at[i, 'F_Probability'] = 0


def func_cut_off():
    cut_off_values = []
    x = 0.05
    while x < 1:
        cut_off_values.append(round(x, 2))
        x += 0.05
    return np.array(cut_off_values)


def decision_score(X, Y):
    cut_off_values = func_cut_off()

    result_df = pd.DataFrame(columns=columns_decision_score)

    for cut_off in cut_off_values:
        cutoff_applied_data = X.copy()

        # Застосування CutOff
        apply_cutoff_probability(cutoff_applied_data, cut_off)

        # Обчислення матриці похибок
        tn, fp, fn, tp = confusion_matrix(Y, cutoff_applied_data['F_Probability']).ravel()

        metrics = {
            'cut_off': cut_off,
            'TP': tp,
            'TN': tn,
            'FN': fn,
            'FP': fp,
            'Recall': round(recall_score(Y, cutoff_applied_data['F_Probability']), 4),
            'Precision': round(precision_score(Y, cutoff_applied_data['F_Probability']), 4),
            'F_Score': round(f1_score(Y, cutoff_applied_data['F_Probability']), 4),
            'MISC': round(1 - accuracy_score(Y, cutoff_applied_data['F_Probability']), 4)
        }
        metrics_data = pd.DataFrame([metrics])
        result_df.loc[len(result_df)] = metrics_data.iloc[0]
    # print(result_df)
    result_df = result_df.sort_values(by=['F_Score', 'MISC', 'cut_off'], ascending=[False, False, False]).reset_index(
        drop=True)
    # print(result_df)
    return result_df


def scoring(X, Y, i):
    X_upd = X[:i]
    Y_upd = Y.iloc[:i]

    data = sm.add_constant(X_upd)
    model = sm.Logit(Y_upd, data).fit(disp=0)

    predicted = model.predict(data)

    predicted_X = X_upd.copy()
    predicted_X['F_Probability'] = predicted

    return predicted_X, Y_upd, model


def task_6(train_df, X, Y):
    CUTOFF_BEST_ROW_MERGE = pd.DataFrame(columns=columns_cut_off_best_row_merge)
    BETAS_MERGE = pd.DataFrame(columns=columns_betas_merge)
    SCORE_TEST_MERGE = pd.DataFrame(columns=columns_score_test_merge)
    REAL_PREDICTIONS = pd.DataFrame(columns=columns_real_predictions)
    for i in range(30, len(train_df)):
        predicted_X, Y_upd, model = scoring(X, Y, i)

        result_df = decision_score(predicted_X, Y_upd)
        X_next = X[:i + 1]

        predicted_X_next = model.predict(sm.add_constant(X_next))

        metrics_1 = {
            'n_train': i,
            'cut_off': result_df.iloc[0]['cut_off'],
            'TP': result_df.iloc[0]['TP'],
            'TN': result_df.iloc[0]['TN'],
            'FN': result_df.iloc[0]['FN'],
            'FP': result_df.iloc[0]['FP'],
            'Recall': result_df.iloc[0]['Recall'],
            'Precision': result_df.iloc[0]['Precision'],
            'F_Score': result_df.iloc[0]['F_Score'],
            'MISC': result_df.iloc[0]['MISC']
        }
        curr_res_1 = pd.DataFrame([metrics_1])
        CUTOFF_BEST_ROW_MERGE = pd.concat([CUTOFF_BEST_ROW_MERGE, curr_res_1], ignore_index=True)

        metrics_2 = {
            'n_train': i,
            'best_cutoff': result_df.iloc[0]['cut_off'],
            'intercept': model.params[0],
            'week_price_change': model.params[1],
            'month_price_change': model.params[2],
            'week_volume_change': model.params[3],
            'month_volume_change': model.params[4],
            'rank': model.params[5],
            'N_of_listings': model.params[6],
            'mean_volume': model.params[8],
            'month_delta_listings': model.params[7]
        }
        curr_res_2 = pd.DataFrame([metrics_2])
        BETAS_MERGE = pd.concat([BETAS_MERGE, curr_res_2], ignore_index=True)

        metrics_3 = {
            'n_train': i,
            'best_cutoff': result_df.iloc[0]['cut_off'],
            'listing_date': train_df.iloc[i]['listing_date'],
            'symbol': train_df.iloc[i]['symbol'],
            'week_price_change': train_df.iloc[i]['week_price_change'],
            'month_price_change': train_df.iloc[i]['month_price_change'],
            'week_volume_change': train_df.iloc[i]['week_volume_change'],
            'month_volume_change': train_df.iloc[i]['month_volume_change'],
            'rank': train_df.iloc[i]['rank'],
            'N_of_listings': train_df.iloc[i]['N_of_listings'],
            'month_delta_listings': train_df.iloc[i]['month_delta_listings'],
            'mean_volume': train_df.iloc[i]['mean_volume'],
            'P0': 1 - predicted_X_next.iloc[-1],
            'P1': predicted_X_next.iloc[-1]
        }
        curr_res_3 = pd.DataFrame([metrics_3])
        SCORE_TEST_MERGE = pd.concat([SCORE_TEST_MERGE, curr_res_3], ignore_index=True)

        metrics_4 = {
            'n_train': i,
            'best_cutoff': result_df.iloc[0]['cut_off'],
            'listing_date': train_df.iloc[i]['listing_date'],
            'symbol': train_df.iloc[i]['symbol'],
            'P0': 1 - predicted_X_next.iloc[-1],
            'P1': predicted_X_next.iloc[-1],
            'Real_class': train_df.iloc[i]['class'],
            'Forecast_class': 0 if predicted_X_next.iloc[-1] < result_df.iloc[0]['cut_off'] else 1
        }
        curr_res_4 = pd.DataFrame([metrics_4])
        REAL_PREDICTIONS = pd.concat([REAL_PREDICTIONS, curr_res_4], ignore_index=True)

    CUTOFF_BEST_ROW_MERGE.to_excel('CUTOFF_BEST_ROW_MERGE.xlsx', index=False)
    BETAS_MERGE.to_excel('BETAS_MERGE.xlsx', index=False)
    SCORE_TEST_MERGE.to_excel('SCORE_TEST_MERGE.xlsx', index=False)
    REAL_PREDICTIONS.to_excel('REAL_PREDICTIONS.xlsx', index=False)

    return CUTOFF_BEST_ROW_MERGE, BETAS_MERGE, SCORE_TEST_MERGE, REAL_PREDICTIONS


def task_7(BETAS_MERGE, REAL_PREDICTIONS):
    real_class = REAL_PREDICTIONS['Real_class'].to_numpy(dtype=int)
    forecast_class = REAL_PREDICTIONS['Forecast_class'].to_numpy(dtype=int)

    tn, fp, fn, tp = confusion_matrix(real_class, forecast_class).ravel()

    result = {
        'TP': tp,
        'TN': tn,
        'FN': fn,
        'FP': fp,
        'Recall': round(recall_score(real_class, forecast_class), 4),
        'Precision': round(precision_score(real_class, forecast_class), 4),
        'F_Score': round(f1_score(real_class, forecast_class), 4),
        'MISC': round(1 - accuracy_score(real_class, forecast_class), 4)
    }

    for key, value in result.items():
        print(f'{key}: {value}')

    for regressor in BETAS_MERGE.columns[1:]:
        plt.figure(figsize=(12, 8))
        plt.plot(BETAS_MERGE['n_train'], BETAS_MERGE[regressor], alpha=0.9, label=regressor)
        plt.axhline(0, color='black', linewidth=1)
        plt.xlabel('n_train')
        plt.ylabel(regressor)
        plt.title(regressor)
        plt.grid(True)
        plt.show()


def main():
    train_file_path = 'TRAIN_DATA.xlsx'
    test_file_path = 'score.xlsx'

    train_df = pd.read_excel(train_file_path, sheet_name='TRAIN_DATA')
    test_df = pd.read_excel(test_file_path, sheet_name='SCORE')

    print("\t1. Аналіз якості даних:")
    data_quality_analysis(train_df)

    print("\n\t2. Заміна пропусків в даних та приведення до цілих чисел:")
    train_df = task_2(train_df)

    print("\n\t3. Формування тренінгового набору даних та Побудова моделі логістичної регресії на тренінговому наборі "
          "даних:")
    X = train_df.drop(['symbol', 'listing_date', 'class'], axis=1)
    Y = train_df['class']
    X_30, Y_30 = training_dataset(train_df)
    MLE(X_30, Y_30)
    odds_ratio(X_30, Y_30)
    concordance(X_30, Y_30)

    print("\n\t4. Скорінгування тренінгового набору даних:")
    data = sm.add_constant(X_30)
    model = sm.Logit(Y_30, data).fit()

    predicted = model.predict(data)

    predicted_X = X_30.copy()
    predicted_X['F_Probability'] = predicted

    print("\n\t5. Визначення балу прийняття рішення:")
    decision_score(predicted_X, Y_30)

    print("\n\t6. Прогнозування на один крок вперед та подальша циклічна обробка даних:")
    CUTOFF_BEST_ROW_MERGE, BETAS_MERGE, SCORE_TEST_MERGE, REAL_PREDICTIONS = task_6(train_df, X, Y)

    print("\n\t7. Аналіз отриманих результатів:")
    task_7(BETAS_MERGE, REAL_PREDICTIONS)


if __name__ == "__main__":
    main()
