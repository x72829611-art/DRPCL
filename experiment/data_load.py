import pandas as pd
import numpy as np
import os


def load_and_format_covariates_ihdp(style, path=None):
    if path is None:
        if style == "train":
            path = "/home/student1/projects/DRPCL/data/ihdp/ihdp_npci_train_1.csv"
        else:
            path = "/home/student1/projects/DRPCL/data/ihdp/ihdp_npci_test_1.csv"
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    contfeats = [i for i in range(25) if i not in binfeats]
    C =  data[:, 5:]
    perm = binfeats + contfeats
    C = C[:, perm]
    return C


def load_all_other_crap(style, path=None):
    if path is None:
        if style == "train":
            path = "/home/student1/projects/DRPCL/data/ihdp/ihdp_npci_train_1.csv"
        else:
            path = "/home/student1/projects/DRPCL/data/ihdp/ihdp_npci_test_1.csv"
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    A, Y, Y_cf = data[:, 0], data[:, 1][:, None], data[:, 2][:, None]
    mu_0, mu_1, x = data[:, 3][:, None], data[:, 4][:, None], data[:, 5:]
    return A.reshape(-1, 1), Y, Y_cf, mu_0, mu_1


def load_jobs_csv_data(file_path):


        data = pd.read_csv(file_path)
        
        # JOBS数据的实际列名：treatment, y_factual, y_cfactual, mu0, mu1, x_1, x_2, ..., x_n
        A = np.array(data['treatment']).reshape(-1, 1)
        Y = np.array(data['y_factual']).reshape(-1, 1)
        e= np.array(data['e']).reshape(-1, 1)

        
        # 提取协变量列（所有以x_开头的列）
        C_columns = [col for col in data.columns if col.startswith('x_')]
        C= data[C_columns].values
        
        print(f"Jobs data loaded: {len(data)} samples, {len(C_columns)} features")
        
        return C, A, Y, e


def load_twins_csv_data(file_path):
    data = pd.read_csv(file_path)
    
    # Actual CSV format: x_0...x_39, ycf, t, yf
    # Modified to match new CSV format: treatment, y_factual, y_cfactual
    if 'treatment' in data.columns:
        A = np.array(data['treatment']).reshape(-1, 1)
    else:
        A = np.array(data['t']).reshape(-1, 1)
        
    if 'y_factual' in data.columns:
        Y = np.array(data['y_factual']).reshape(-1, 1)
    else:
        Y = np.array(data['yf']).reshape(-1, 1)
        
    if 'y_cfactual' in data.columns:
        Y_cf = np.array(data['y_cfactual']).reshape(-1, 1)
    else:
        Y_cf = np.array(data['ycf']).reshape(-1, 1)
    
    # Calculate mu_0 and mu_1
    # If t=0: yf=mu_0, ycf=mu_1
    # If t=1: yf=mu_1, ycf=mu_0
    mu_0 = np.where(A == 0, Y, Y_cf)
    mu_1 = np.where(A == 1, Y, Y_cf)
    
    # Extract covariates (columns starting with 'x_')
    C_columns = [col for col in data.columns if col.startswith('x_')]
    C = data[C_columns].values

    return C, A, Y, mu_0, mu_1, None # Return None for extra return value if needed, matching signature


def main():
    pass


if __name__ == '__main__':
    main()
