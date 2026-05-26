import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from models import *
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from data_load import *
import gc
import torch
import numpy as np
import csv
import datetime


def _split_output(yt_hat, a, y, y_scaler, index, e=None):
    yt_hat = yt_hat.detach().cpu().numpy()
    h0 = y_scaler.inverse_transform(yt_hat[:, 0].reshape(-1, 1).copy())
    h1 = y_scaler.inverse_transform(yt_hat[:, 1].reshape(-1, 1).copy())
    q = yt_hat[:, 2].copy()
    y = y_scaler.inverse_transform(y.copy())
    e_batch = e[index] if (e is not None and index is not None) else e
    return {'h0': h0, 'h1': h1, 'q': q, 'a': a, 'y': y, 'index': index, 'e': e_batch}


def _train_phase(train_loader, net, optimizer, epoch, loss_fn, beta, phase, clip_grad=None):
    avg_loss = 0
    for i, data in enumerate(train_loader):
        inputs, labels, y_train, t_train = data
        optimizer.zero_grad()
        outputs = net(inputs, y_train, t_train)

        if phase == 1:
            loss_dict = loss_fn(outputs, labels, net, beta=beta)
            loss = loss_dict['total_loss']
        elif phase == 2:
            loss_dict = loss_fn(outputs, labels, net, beta=0.0)
            loss = loss_dict['loss_phase2']
        else:
            loss_dict = loss_fn(outputs, labels, net, beta=beta, cls_weight=0.0, q_weight=0.0)
            loss = loss_dict['loss_phase3']

        if epoch % 100 == 0 and i == 0:
            if phase == 1:
                print(f"Phase1 [Epoch {epoch}] - Reg: {loss_dict['reg_loss']:.4f}, Cls: {loss_dict['cls_loss']:.4f}, Q: {loss_dict['q_loss']:.4f}")
            elif phase == 2:
                print(f"Phase2 [Epoch {epoch}] - Cls: {loss_dict['cls_loss']:.4f}, Q: {loss_dict['q_loss']:.4f}")
            else:
                print(f"Phase3 [Epoch {epoch}] - Reg: {loss_dict['reg_loss']:.4f}")

        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
        optimizer.step()
        avg_loss += loss
    return avg_loss / len(train_loader)


def _compute_pdr_ite(h0, h1, q, a, y_observed):
    h_observed = h1 * a + h0 * (1 - a)
    ite_naive = h1 - h0
    sign_term = (2 * a - 1)
    return sign_term * q * (y_observed - h_observed) + ite_naive


def make_table_ihdp(train_output, train_truth, train_test='train'):
    result_dict = train_output[0]
    h0 = result_dict['h0'].flatten()
    h1 = result_dict['h1'].flatten()
    q = result_dict['q'].flatten()
    a = result_dict['a'].flatten()
    y_observed = result_dict['y'].flatten()
    truth_flat = train_truth.flatten()

    ite_pred = _compute_pdr_ite(h0, h1, q, a, y_observed)

    if len(truth_flat) != len(ite_pred):
        min_len = min(len(truth_flat), len(ite_pred))
        truth_flat, ite_pred = truth_flat[:min_len], ite_pred[:min_len]

    ate_err = abs(truth_flat.mean() - ite_pred.mean())
    pehe_err = np.sqrt(np.mean(np.square(truth_flat - ite_pred)))
    print(f"[{train_test}] ATE Error: {ate_err:.6f}, PEHE: {pehe_err:.6f}")
    return ate_err, pehe_err


def make_table_twins(train_output, train_truth, train_test='train'):
    result_dict = train_output[0]
    h0 = result_dict['h0'].flatten()
    h1 = result_dict['h1'].flatten()
    q = result_dict['q'].flatten()
    a = result_dict['a'].flatten()
    y_observed = result_dict['y'].flatten()
    truth_flat = train_truth.flatten()

    ite_pred = _compute_pdr_ite(h0, h1, q, a, y_observed)
    ate_err = abs(truth_flat.mean() - ite_pred.mean())

    auc = None
    try:
        h_observed = h1 * a + h0 * (1 - a)
        y_binary = np.round(y_observed)
        if len(np.unique(y_binary)) > 1:
            auc = roc_auc_score(y_binary, h_observed)
    except Exception:
        pass

    print(f"[{train_test}] ATE Error: {ate_err:.6f}, AUC: {auc}")
    return ate_err, auc


def make_table_jobs(train_output, train_test='train'):
    result_dict = train_output[0] if isinstance(train_output, list) else train_output
    h0 = result_dict['h0'].flatten()
    h1 = result_dict['h1'].flatten()
    q = result_dict['q'].flatten()
    a = result_dict['a'].flatten()
    y = result_dict['y'].flatten()
    e = result_dict.get('e', np.ones_like(a)).flatten()

    def abs_att(effect_pred, yf, a, e):
        att_true = np.mean(yf[a > 0]) - np.mean(yf[(1 - a + e) > 1])
        att_pred = np.mean(effect_pred[(a + e) > 1])
        return np.abs(att_pred - att_true)

    effect_naive = h1 - h0
    policy = effect_naive > 0
    treat_overlap = (policy == a) * (a > 0)
    control_overlap = (policy == a) * (a < 1)
    treat_value = np.mean(y[treat_overlap]) if np.sum(treat_overlap) > 0 else 0
    control_value = np.mean(y[control_overlap]) if np.sum(control_overlap) > 0 else 0
    pit = np.mean(policy)
    Rpol = 1 - (pit * treat_value + (1 - pit) * control_value)

    effect_pdr = _compute_pdr_ite(h0, h1, q, a, y)
    att_err = abs_att(effect_pdr, y, a, e)

    print(f"[{train_test}] Rpol={Rpol:.4f}, ATT_err={att_err:.4f}")
    return Rpol, att_err


def train_and_predict_unified(data_train_dict, data_test_dict, datasets='IHDP',
                              output_dir='', batch_size=128,
                              lr1=5e-5, lr2=3e-3, lr_q=1e-3, beta=0.08):
    device = torch.device("cpu")
    print(f"Training {datasets} on {device}")

    y_scaler = StandardScaler().fit(data_train_dict["y_train"])
    y_train = y_scaler.transform(data_train_dict["y_train"])
    x_scaler = StandardScaler().fit(data_train_dict["x_train"])
    c_train = x_scaler.transform(data_train_dict["x_train"])
    a_train = data_train_dict["t_train"]
    mu_0_train = data_train_dict.get("mu_0_train", np.zeros_like(data_train_dict["y_train"]))
    mu_1_train = data_train_dict.get("mu_1_train", np.zeros_like(data_train_dict["y_train"]))
    e_train = data_train_dict.get("e_train", None)

    c_test = x_scaler.transform(data_test_dict["x_test"])
    y_test = y_scaler.transform(data_test_dict["y_test"])
    a_test = data_test_dict["t_test"]
    e_test = data_test_dict.get("e_test", None)

    loss=None
    i = 0
    torch.manual_seed(i)
    np.random.seed(i)

    if datasets == 'TWINS':
        net = DRPCLModel(c_train.shape[1], out_features=[400, 200, 1], dropout_rate=0.1)
        loss_fn = drpcl_loss_with_mi
        momentum_phase3 = 0.9
        clip_grad = 1.0
    elif datasets == 'IHDP':
        net = DRPCLModel(c_train.shape[1], out_features=[200, 100, 1])
        loss_fn = drpcl_loss_with_mi
        momentum_phase3 = 0.0
        clip_grad = None
    else:
        net = DRPCLModel(c_train.shape[1], out_features=[200, 100, 1])
        loss_fn = drpcl_loss_with_mi
        momentum_phase3 = 0.9
        clip_grad = None

    labels = np.column_stack([y_train.flatten(), a_train.flatten(), mu_0_train.flatten(), mu_1_train.flatten()])
    tensors = (torch.from_numpy(c_train).float(), torch.from_numpy(labels).float(),
               torch.from_numpy(y_train).float(), torch.from_numpy(a_train).float())
    train_loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size,shuffle=(datasets=='TWINS'))
    
    optimizer_p1 = optim.SGD([
        {'params': net.rep_Z.parameters()}, {'params': net.rep_X.parameters()}, {'params': net.rep_W.parameters()},
        {'params': net.C_Z.parameters()}, {'params': net.C_X.parameters()}, {'params': net.C_W.parameters()},
        {'params': net.h0_head.parameters(), 'weight_decay': 0.01},
        {'params': net.h1_head.parameters(), 'weight_decay': 0.01},
        {'params': net.gw_predictions.parameters(), 'weight_decay': 0.01},
        {'params': net.q_predictions.parameters(), 'weight_decay': 0.01},
    ], lr=lr1, momentum=0.9)

    optimizer_p2 = optim.SGD([
        {'params': net.gw_predictions.parameters(), 'weight_decay': 0.01},
        {'params': net.q_predictions.parameters(), 'weight_decay': 0.01},
    ], lr=lr_q, momentum=0.9)

    optimizer_p3 = optim.SGD([
        {'params': net.rep_Z.parameters()}, {'params': net.rep_X.parameters()}, {'params': net.rep_W.parameters()},
        {'params': net.C_Z.parameters()}, {'params': net.C_X.parameters()}, {'params': net.C_W.parameters()},
        {'params': net.h0_head.parameters(), 'weight_decay': 0.01},
        {'params': net.h1_head.parameters(), 'weight_decay': 0.01},
    ], lr=lr2, momentum=momentum_phase3)

    scheduler_p1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_p1, mode='min', factor=0.5, patience=5, threshold=1e-8, min_lr=0)
    scheduler_p3 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_p3, mode='min', factor=0.5, patience=5, threshold=1e-8, min_lr=0)

    print("=== Phase 1 ===")
    for epoch in range(100):
        loss = _train_phase(train_loader, net, optimizer_p1, epoch, loss_fn, beta, phase=1, clip_grad=clip_grad)
        scheduler_p1.step(loss)

    print("=== Phase 2 ===")
    for epoch in range(50):
        loss = _train_phase(train_loader, net, optimizer_p2, epoch, loss_fn, beta, phase=2, clip_grad=clip_grad)

    print("=== Phase 3 ===")
    for epoch in range(300):
        loss = _train_phase(train_loader, net, optimizer_p3, epoch, loss_fn, beta, phase=3, clip_grad=clip_grad)
        scheduler_p3.step(loss)

    net.eval()
    with torch.no_grad():
        yt_hat_train = net(torch.from_numpy(c_train).float(), torch.from_numpy(y_train).float(), torch.from_numpy(a_train).float())
        yt_hat_test = net(torch.from_numpy(c_test).float(), torch.from_numpy(y_test).float(), torch.from_numpy(a_test).float())

    train_idx = np.arange(c_train.shape[0])
    test_idx = np.arange(c_test.shape[0])
    train_outputs = [_split_output(yt_hat_train, a_train, y_train, y_scaler, train_idx, e=e_train)]
    test_outputs = [_split_output(yt_hat_test, a_test, y_test, y_scaler, test_idx, e=e_test)]
    return test_outputs, train_outputs


def run_ihdp(data_base_dir='./dat/ihdp/csv', output_dir='./result/', datasets='IHDP',
             dragon='DRPCL', reptition=1, lr1=5e-5, lr2=3e-3, lr_q=1e-3,
             batch_size=64, beta=0.08):
    print(f"Dataset: {datasets}")
    data_dir = "/home/student1/projects/DRPCL/data/ihdp"
    train_path = os.path.join(data_dir, f"ihdp_npci_train_{reptition}.csv")
    test_path = os.path.join(data_dir, f"ihdp_npci_test_{reptition}.csv")

    x_train = load_and_format_covariates_ihdp("train", train_path)
    t_train, y_train, y_cf_train, mu_0_train, mu_1_train = load_all_other_crap("train", train_path)
    data_train_dict = {'x_train': x_train, 't_train': t_train, 'y_train': y_train,
                       'mu_0_train': mu_0_train, 'mu_1_train': mu_1_train}
    truth_train = mu_1_train - mu_0_train

    x_test = load_and_format_covariates_ihdp("test", test_path)
    t_test, y_test, y_cf_test, mu_0_test, mu_1_test = load_all_other_crap("test", test_path)
    data_test_dict = {'x_test': x_test, 't_test': t_test, 'y_test': y_test,
                      'mu_0_test': mu_0_test, 'mu_1_test': mu_1_test}
    truth_test = mu_1_test - mu_0_test

    test_outputs, train_output = train_and_predict_unified(
        data_train_dict, data_test_dict, datasets='IHDP', output_dir=output_dir,
        batch_size=batch_size, lr1=lr1, lr2=lr2, lr_q=lr_q, beta=beta)

    ate_train, pehe_train = make_table_ihdp(train_output, truth_train, train_test='train')
    ate_test, pehe_test = make_table_ihdp(test_outputs, truth_test, train_test='test')
    print(f"IHDP Rep {reptition}: ATE_train={ate_train:.4f}, PEHE_train={pehe_train:.4f}, ATE_test={ate_test:.4f}, PEHE_test={pehe_test:.4f}")

    result_file = os.path.join(output_dir, f"ihdp_result_{reptition}.csv")
    os.makedirs(output_dir, exist_ok=True)
    with open(result_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["ate_train", "pehe_train", "ate_test", "pehe_test"])
        writer.writerow([ate_train, pehe_train, ate_test, pehe_test])
    return ate_train, pehe_train, ate_test, pehe_test


def run_jobs(data_base_dir='./dat/jobs/', output_dir='./result/', datasets='JOBS',
             dragon='DRPCL', reptition=1, lr1=5e-5, lr2=1e-4, lr_q=1e-3,
             batch_size=256, beta=1.0):
    print(f"Dataset: {datasets}")
    jobs_data_dir = "/home/student1/projects/DRPCL/data/jobs"
    train_path = os.path.join(jobs_data_dir, f"jobs_train_{reptition}.csv")
    test_path = os.path.join(jobs_data_dir, f"jobs_test_{reptition}.csv")

    x_train, t_train, y_train, e_train = load_jobs_csv_data(train_path)
    data_train_dict = {'x_train': x_train, 't_train': t_train, 'y_train': y_train, 'e_train': e_train}

    x_test, t_test, y_test, e_test = load_jobs_csv_data(test_path)
    data_test_dict = {'x_test': x_test, 't_test': t_test, 'y_test': y_test, 'e_test': e_test}

    test_outputs, train_output = train_and_predict_unified(
        data_train_dict, data_test_dict, datasets='JOBS', output_dir=output_dir,
        batch_size=batch_size, lr1=lr1, lr2=lr2, lr_q=lr_q, beta=beta)

    rpol_train, att_err_train = make_table_jobs(train_output, train_test='train')
    rpol_test, att_err_test = make_table_jobs(test_outputs, train_test='test')
    print(f"JOBS Rep {reptition}: Rpol_train={rpol_train:.4f}, ATT_train={att_err_train:.4f}, Rpol_test={rpol_test:.4f}, ATT_test={att_err_test:.4f}")

    result_file = os.path.join(output_dir, f"jobs_result_{reptition}.csv")
    os.makedirs(output_dir, exist_ok=True)
    with open(result_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Rpol_train", "ATT_err_train", "Rpol_test", "ATT_err_test"])
        writer.writerow([rpol_train, att_err_train, rpol_test, att_err_test])
    return rpol_train, att_err_train, rpol_test, att_err_test


def run_twins(data_base_dir='./dat/twins/', output_dir='./result/', datasets='TWINS',
              dragon='DRPCL', reptition=1, lr1=5e-5, lr2=1e-4, lr_q=1e-3,
              batch_size=128, beta=1.0):
    print(f"Dataset: {datasets}")
    twins_data_dir = "/home/student1/projects/DRPCL/data/twins"
    train_path = os.path.join(twins_data_dir, f"twins_train_{reptition}.csv")
    test_path = os.path.join(twins_data_dir, f"twins_test_{reptition}.csv")

    x_train, t_train, y_train, mu_0_train, mu_1_train, _ = load_twins_csv_data(train_path)
    data_train_dict = {'x_train': x_train, 't_train': t_train, 'y_train': y_train,
                       'mu_0_train': mu_0_train, 'mu_1_train': mu_1_train}
    truth_train = mu_1_train - mu_0_train

    x_test, t_test, y_test, mu_0_test, mu_1_test, _ = load_twins_csv_data(test_path)
    data_test_dict = {'x_test': x_test, 't_test': t_test, 'y_test': y_test,
                      'mu_0_test': mu_0_test, 'mu_1_test': mu_1_test}
    truth_test = mu_1_test - mu_0_test

    test_outputs, train_output = train_and_predict_unified(
        data_train_dict, data_test_dict, datasets='TWINS', output_dir=output_dir,
        batch_size=batch_size, lr1=lr1, lr2=lr2, lr_q=lr_q, beta=beta)

    ate_train, auc_train = make_table_twins(train_output, truth_train, train_test='train')
    ate_test, auc_test = make_table_twins(test_outputs, truth_test, train_test='test')
    print(f"TWINS Rep {reptition}: ATE_train={ate_train:.4f}, AUC_train={auc_train}, ATE_test={ate_test:.4f}, AUC_test={auc_test}")

    result_file = os.path.join(output_dir, f"twins_result_{reptition}.csv")
    os.makedirs(output_dir, exist_ok=True)
    with open(result_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["ate_train", "auc_train", "ate_test", "auc_test"])
        writer.writerow([ate_train, auc_train, ate_test, auc_test])
    return ate_train, auc_train, ate_test, auc_test


def save_summary_results(dataset_name, output_dir, all_results):
    summary_file = os.path.join(output_dir, f"{dataset_name.lower()}_summary_results.csv")
    if not all_results:
        print(f"No results for {dataset_name}")
        return

    with open(summary_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        if dataset_name == 'TWINS':
            keys = ["ATE_Train", "AUC_Train", "ATE_Test", "AUC_Test"]
        elif dataset_name == 'JOBS':
            keys = ["Rpol_Train", "ATT_Train", "Rpol_Test", "ATT_Test"]
        else:
            keys = ["ATE_Train", "PEHE_Train", "ATE_Test", "PEHE_Test"]

        n = len(all_results)
        vals = list(zip(*all_results))
        means = [np.mean([v for v in col if v is not None]) for col in vals]
        stds = [np.std([v for v in col if v is not None]) / np.sqrt(n) for col in vals]

        writer.writerow(["Metric"] + keys)
        writer.writerow(["Mean"] + [f"{m:.6f}" for m in means])
        writer.writerow(["Std"] + [f"{s:.6f}" for s in stds])
        writer.writerow(["N", n])
        writer.writerow([])
        writer.writerow(["ID"] + keys)
        for i, r in enumerate(all_results, 1):
            writer.writerow([i] + list(r))

        print(f"{dataset_name} Summary ({n} reps):")
        for k, m, s in zip(keys, means, stds):
            print(f"  {k}: {m:.4f} +/- {s:.4f}")

    log_path = os.path.join(output_dir, 'log.txt')
    with open(log_path, 'a') as f:
        f.write(f"{dataset_name} Summary ({n} reps):\n")
        for k, m, s in zip(keys, means, stds):
            f.write(f"  {k}: {m:.4f} +/- {s:.4f}\n")
        f.write("-" * 30 + "\n")

    print(f"Saved: {summary_file}")


def turn_knob(data_base_dir, knob, datasets, output_base_dir,
              lr1=5e-5, lr2=1e-4, lr_q=1e-3, batch_size=256, beta=1.0, replications=100):
    output_base_dir = "/home/student1/projects/DRPCL/experiment/results"
    output_dir = os.path.join(output_base_dir, knob)
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, 'log.txt')
    with open(log_path, 'a') as f:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{now}] {datasets}: Beta={beta}, LR1={lr1}, LR2={lr2}, LR_Q={lr_q}, Batch={batch_size}, Reps={replications}\n")

    all_results = []

    if datasets == 'IHDP':
        for i in range(replications):
            print(f"Dataset {i + 1}")
            result = run_ihdp(data_base_dir=data_base_dir, output_dir=output_dir, datasets='IHDP',
                              dragon='DRPCL', reptition=i + 1, lr1=lr1, lr2=lr2, lr_q=lr_q,
                              batch_size=batch_size, beta=beta)
            if result is not None:
                all_results.append(result)
        save_summary_results('IHDP', output_dir, all_results)

    elif datasets == 'JOBS':
        for i in range(replications):
            print(f"Dataset {i + 1}")
            result = run_jobs(data_base_dir=data_base_dir, output_dir=output_dir, datasets='JOBS',
                              dragon='DRPCL', reptition=i + 1, lr1=lr1, lr2=lr2, lr_q=lr_q,
                              batch_size=batch_size, beta=beta)
            if result is not None:
                all_results.append(result)
        save_summary_results('JOBS', output_dir, all_results)

    elif datasets == 'TWINS':
        for i in range(replications):
            print(f"Dataset {i + 1}")
            gc.collect()
            torch.manual_seed(i)
            np.random.seed(i)
            result = run_twins(data_base_dir=data_base_dir, output_dir=output_dir, datasets='TWINS',
                               dragon='DRPCL', reptition=i + 1, lr1=lr1, lr2=lr2, lr_q=lr_q,
                               batch_size=batch_size, beta=beta)
            if result is not None:
                all_results.append(result)
        save_summary_results('TWINS', output_dir, all_results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_base_dir', type=str, default='./dat/')
    parser.add_argument('--knob', type=str, default='DRPCL')
    parser.add_argument('--datasets', type=str, default='IHDP', help="IHDP, JOBS, TWINS")
    parser.add_argument('--output_base_dir', type=str, default='/home/student1/projects/DRPCL/experiment/results')
    parser.add_argument('--lr1', type=float, default=5e-5)
    parser.add_argument('--lr2', type=float, default=1e-4)
    parser.add_argument('--lr_q', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--replications', type=int, default=100)
    args = parser.parse_args()

    turn_knob(args.data_base_dir, args.knob, args.datasets, args.output_base_dir,
              lr1=args.lr1, lr2=args.lr2, lr_q=args.lr_q, batch_size=args.batch_size,
              beta=args.beta, replications=args.replications)

if __name__ == '__main__':
    main()
