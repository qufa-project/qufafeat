import os 
import numpy as np
import pandas as pd

from scipy.stats import shapiro, kstest, skew, kurtosis 

def load_csvfile(path_csvfile: str):
    file = pd.read_csv(path_csvfile, index_col=0)
    return file

def ec_is_norm(input_data):
    """
    [ Shapiro-Wilk Test, Kolmogorov-Smirnov Test ]
    It's the most powerful test to check the normality of a variable.
    IF the p-value <= 0.05 THEN we assume the distribution of our variable is not normal/gaussian.
    IF the p-value > 0.05 THEN we assume the distribution of our variable is normal/gaussian.
    """
    if len(input_data) <= 2000:
        # do Shapiro Test
        statistic, p_value = shapiro(input_data)
        s = skew(input_data)
        k = kurtosis(input_data)

        # passed test
        if p_value > 0.05 or (abs(s) < 2 and abs(k) < 2): 
            df_data = pd.DataFrame(input_data, columns=['data'])
            df_data['result'] = [True for i in range(df_data.shape[0])]
            '''
            [ IQR method ] 
            It's the general method to detect outlier data
            '''
            level_q1 = df_data['data'].quantile(0.25)
            level_q3 = df_data['data'].quantile(0.75)
            iqr = level_q3 - level_q1

            df_data.loc[(df_data['data'] > level_q3 + (1.5 * iqr)) | (df_data['data'] < level_q1 - (1.5 * iqr)), 'result'] = False
            return list(df_data['result'])

        else:  # failed test
            return [False for i in range(len(input_data))]

    else: # input size more than 2000
        statistic, p_value = kstest(input_data, 'norm')
        s = skew(input_data)
        k = kurtosis(input_data)
        
        # passed test
        if p_value > 0.05 or (abs(s) < 2 and abs(k) < 2):
            df_data = pd.DataFrame(input_data, columns=['data'])
            df_data['result'] = [True for i in range(df_data.shape[0])]
            '''
            [ IQR method ] 
            It's the general method to detect outlier data
            '''
            level_q1 = df_data['data'].quantile(0.25)
            level_q3 = df_data['data'].quantile(0.75)
            iqr = level_q3 - level_q1

            df_data.loc[(df_data['data'] > level_q3 + (1.5 * iqr)) | (df_data['data'] < level_q1 - (1.5 * iqr)), 'result'] = False

            return list(df_data['result'])
        else: # failed test
            return [False for i in range(len(input_data))]

def get_normality_violation(filename):
    data = load_csvfile(filename)
    input_data = data['height'].to_list()
            
    result = ec_is_norm(input_data)

    df_result = pd.DataFrame(result, columns={'result'})

    num_of_correct_rows = len(df_result.loc[df_result['result'] == True])
    num_of_except_rows = len(df_result.loc[df_result['result'] == False])

    normality_violation_rate = 100 - (num_of_correct_rows / len(df_result) * 100)

    print('\n**************************************************')
    if num_of_correct_rows == 0:
        print('주어진 데이터는 정규분포를 따르지 않습니다.')
    else:
        print(f'정규성 위배율: {normality_violation_rate:.2f}%')
        print('**************************************************')

def correct_normality_violation(input_filename, output_filename):
    input_data = pd.read_csv(input_filename, index_col=0)
    df_temp = input_data.copy()
    df_temp.rename(columns={input_data.columns[0]: 'data'}, inplace=True)

    """
    [ Shapiro-Wilk Test, Kolmogorov-Smirnov Test ]
    It's the most powerful test to check the normality of a variable.
    IF the p-value <= 0.05 THEN we assume the distribution of our variable is not normal/gaussian.
    IF the p-value > 0.05 THEN we assume the distribution of our variable is normal/gaussian.
    """
    
    if len(input_data) <= 2000:
        # do Shapiro Test
        statistic, p_value = shapiro(input_data)
        s = skew(input_data)
        k = kurtosis(input_data)

        if p_value > 0.05 or (abs(s) < 2 and abs(k) < 2): # passed test
            '''
            [ IQR method ]
            It's the general method to detect outlier data
            '''
            level_q1 = df_temp['data'].quantile(0.25)
            level_q3 = df_temp['data'].quantile(0.75)
            iqr = level_q3 - level_q1

            correction_value = df_temp.loc[(df_temp['data'] <= level_q3 + (1.5 * iqr)) | (df_temp['data'] >= level_q1 - (1.5 * iqr)), 'data'].mean()
            input_data.iloc[(input_data.iloc[:, 0] > level_q3 + (1.5 * iqr)) | (input_data.iloc[:, 0] < level_q1 - (1.5 * iqr)), 0] = correction_value
        
        else:  # failed test
            input_data.iloc[:, 0] = input_data.iloc[:, 0].mean()

    else:  # input size more than 2000
        statistic, p_value = kstest(input_data, 'norm')
        s = skew(input_data)
        k = kurtosis(input_data)

        if p_value > 0.05 or (abs(s) < 2 and abs(k) < 2):  # passed test
            '''
            [ IQR method ]
            It's the general method to detect outlier data
            '''
            level_q1 = df_temp['data'].quantile(0.25)
            level_q3 = df_temp['data'].quantile(0.75)
            iqr = level_q3 - level_q1

            correction_value = df_temp.loc[(df_temp['data'] <= level_q3 + (1.5 * iqr)) | (df_temp['data'] >= level_q1 - (1.5 * iqr)), 'data'].mean()
            input_data.iloc[(input_data.iloc[:, 0] > level_q3 + (1.5 * iqr)) | (input_data.iloc[:, 0] < level_q1 - (1.5 * iqr)), 0] = correction_value

        else:  # failed test
            input_data.iloc[:, 0] = input_data.iloc[:, 0].mean()

    input_data.to_csv(output_filename)
    print('[정규성 위배 오류가 보정된 파일 생성 및 저장]: ' + output_filename)
    print()
