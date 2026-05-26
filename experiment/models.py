import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MINet(nn.Module):
    """Mutual information estimation network (variational bound)."""
    def __init__(self, input_dim, output_dim, num_samples, sigma=1.0, mi_min_max='max', name=None):
        super(MINet, self).__init__()
        self.fc_mu1 = nn.Linear(input_dim, input_dim // 2)
        self.fc_mu2 = nn.Linear(input_dim // 2, output_dim)
        self.fc_var1 = nn.Linear(input_dim, input_dim // 2)
        self.fc_var2 = nn.Linear(input_dim // 2, output_dim)
        self.num_samples = num_samples
        self.sigma = sigma
        self.mi_min_max = mi_min_max
        self.name = name

    def forward(self, x_input, y_target, unused=None):
        h_mu = F.elu(self.fc_mu1(x_input))
        mu = self.fc_mu2(h_mu)
        h_var = F.elu(self.fc_var1(x_input))
        logvar = torch.tanh(self.fc_var2(h_var))

        idx = torch.randperm(x_input.size(0))
        y_target_shuffled = y_target[idx]

        loglikeli = -torch.mean(torch.sum(-(y_target - mu)**2 / torch.exp(logvar) - logvar, dim=-1))

        pos = -(mu - y_target)**2 / torch.exp(logvar)
        neg = -(mu - y_target_shuffled)**2 / torch.exp(logvar)
        w_soft = torch.ones_like(pos) / self.num_samples

        pn = -1. if self.mi_min_max == 'max' else 1.
        bound = pn * torch.sum(w_soft * (pos - neg))

        return loglikeli, bound, mu, logvar, w_soft


def weights_init_normal(params):
    if isinstance(params, nn.Linear):
        torch.nn.init.normal_(params.weight, mean=0.0, std=1.0)
        torch.nn.init.zeros_(params.bias)


def weights_init_uniform(params):
    if isinstance(params, nn.Linear):
        limit = math.sqrt(6 / (params.weight.shape[1] + params.weight.shape[0]))
        torch.nn.init.uniform_(params.weight, a=-limit, b=limit)
        torch.nn.init.zeros_(params.bias)


class DRPCLModel(nn.Module):
    """DRPCL: Deep Representation learning for Proximal Causal Learning.
    3 representation networks (Z,X,W) + combined layers + outcome/propensity/bridge heads + 9 MI networks.
    """
    def __init__(self, in_features, out_features=[200, 100, 1], mi_min_max='max',
                 num_samples=100, sigma=1.0, dropout_rate=0.0):
        super(DRPCLModel, self).__init__()
        self.out_features = out_features
        self.mi_min_max = mi_min_max
        hidden_dim = out_features[0]

        # Representation networks
        self.rep_Z = nn.Sequential(
            nn.Linear(in_features, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU())
        self.rep_X = nn.Sequential(
            nn.Linear(in_features, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU())
        self.rep_W = nn.Sequential(
            nn.Linear(in_features, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU())

        # Combined representation layers
        self.C_Z = nn.Sequential(
            nn.Linear(out_features[0]*2, out_features[0]), nn.ReLU(),
            nn.Linear(out_features[0], out_features[0]), nn.ReLU())
        self.C_X = nn.Sequential(
            nn.Linear(out_features[0]*2, out_features[0]), nn.ReLU(),
            nn.Linear(out_features[0], out_features[0]), nn.ReLU())
        self.C_W = nn.Sequential(
            nn.Linear(out_features[0]*2, out_features[0]), nn.ReLU(),
            nn.Linear(out_features[0], out_features[0]), nn.ReLU())

        # Propensity P(A|W,X)
        self.gw_predictions = nn.Sequential(
            nn.Linear(out_features[0]*2, self.out_features[2]), nn.Sigmoid())

        input_dim_combined = out_features[0] * 2 + 1

        # Bridge function q(Z, A, X)
        self.q_predictions = nn.Sequential(
            nn.Linear(input_dim_combined, 100), nn.ReLU(),
            nn.Linear(100, 1))

        # Outcome heads h0(X,W), h1(X,W)
        self.h0_head = nn.Sequential(
            nn.Linear(input_dim_combined, self.out_features[0]), nn.ReLU(), nn.Dropout(p=dropout_rate),
            nn.Linear(self.out_features[0], self.out_features[1]), nn.ReLU(), nn.Dropout(p=dropout_rate),
            nn.Linear(self.out_features[1], self.out_features[2]))
        self.h1_head = nn.Sequential(
            nn.Linear(input_dim_combined, self.out_features[0]), nn.ReLU(), nn.Dropout(p=dropout_rate),
            nn.Linear(self.out_features[0], self.out_features[1]), nn.ReLU(), nn.Dropout(p=dropout_rate),
            nn.Linear(self.out_features[1], self.out_features[2]))

        # MI networks: 6 (rep-variable) + 3 (rep-rep disentanglement)
        self.mi_zx = MINet(self.out_features[0], self.out_features[0], num_samples, sigma, 'min', name='zx')
        self.mi_zw = MINet(self.out_features[0], self.out_features[0], num_samples, sigma, 'min', name='zw')
        self.mi_xw = MINet(self.out_features[0], self.out_features[0], num_samples, sigma, 'min', name='xw')
        self.mi_za = MINet(self.out_features[0], 1, num_samples, sigma, 'max', name='za')
        self.mi_zy = MINet(self.out_features[0], 1, num_samples, sigma, 'min', name='zy')
        self.mi_xa = MINet(self.out_features[0], 1, num_samples, sigma, 'max', name='xa')
        self.mi_xy = MINet(self.out_features[0], 1, num_samples, sigma, 'max', name='xy')
        self.mi_wa = MINet(self.out_features[0], 1, num_samples, sigma, 'min', name='wa')
        self.mi_wy = MINet(self.out_features[0], 1, num_samples, sigma, 'max', name='wy')

    def forward(self, C, Y, A):
        z = self.rep_Z(C)
        x = self.rep_X(C)
        w = self.rep_W(C)

        C_Z = self.C_Z(torch.cat([z, x], dim=1))
        C_X = self.C_X(torch.cat([z, w], dim=1))
        C_W = self.C_W(torch.cat([x, w], dim=1))

        h0_out = self.h0_head(torch.cat([C_X, torch.zeros_like(A), C_W], dim=1))
        h1_out = self.h1_head(torch.cat([C_X, torch.ones_like(A), C_W], dim=1))
        gw_head = self.gw_predictions(torch.cat([C_W, C_X], dim=1))

        A_reshaped = A.view(-1, 1)
        q_head = self.q_predictions(torch.cat([C_Z, A_reshaped, C_X], dim=1))

        # MI constraints
        iid_za, bound_za, *_ = self.mi_za(z, A)
        iid_zy, bound_zy, *_ = self.mi_zy(z, Y)
        iid_xa, bound_xa, *_ = self.mi_xa(x, A)
        iid_xy, bound_xy, *_ = self.mi_xy(x, Y)
        iid_wa, bound_wa, *_ = self.mi_wa(w, A)
        iid_wy, bound_wy, *_ = self.mi_wy(w, Y)
        iid_zx, bound_zx, *_ = self.mi_zx(z, x)
        iid_zw, bound_zw, *_ = self.mi_zw(z, w)
        iid_xw, bound_xw, *_ = self.mi_xw(x, w)

        self.iid_losses = [iid_za, iid_zy, iid_xa, iid_xy, iid_wa, iid_wy, iid_zx, iid_zw, iid_xw]
        self.bound_losses = [bound_za, bound_zy, bound_xa, bound_xy, bound_wa, bound_wy, bound_zx, bound_zw, bound_xw]

        return torch.cat((h0_out, h1_out, q_head, gw_head), 1)

    def get_mi_losses(self):
        return {
            'iid': self.iid_losses if hasattr(self, 'iid_losses') else [],
            'bound': self.bound_losses if hasattr(self, 'bound_losses') else []
        }


def regression_loss(concat_true, concat_pred):
    """Factual outcome regression loss."""
    y_true = concat_true[:, 0]
    a_true = concat_true[:, 1]
    h0_pred = concat_pred[:, 0]
    h1_pred = concat_pred[:, 1]
    loss0 = torch.sum((1. - a_true) * torch.square(y_true - h0_pred))
    loss1 = torch.sum(a_true * torch.square(y_true - h1_pred))
    return loss0 + loss1


# Fixed loss weights
CLS_WEIGHT = 0.1
Q_WEIGHT = 0.1
DISENTANGLE_WEIGHT = 0.5


def _compute_mi_total(model, device):
    """Compute total MI loss from model's stored MI values."""
    mi_losses = model.get_mi_losses()
    if mi_losses['iid'] and mi_losses['bound']:
        loss_iid = sum(mi_losses['iid'])
        loss_bound = sum(mi_losses['bound'][:6]) + DISENTANGLE_WEIGHT * sum(mi_losses['bound'][6:])
        mi_total = loss_iid + loss_bound
        return mi_total, loss_iid, loss_bound
    zero = torch.tensor(0.0, device=device)
    return zero, zero, zero


def _compute_q_target(gw_pred, a_true):
    """Compute bridge function target: A/e + (1-A)/(1-e)."""
    gw_clipped = torch.clamp(gw_pred.detach(), 1e-6, 1 - 1e-6)
    target = a_true / gw_clipped + (1 - a_true) / (1 - gw_clipped)
    return torch.clamp(target, -50.0, 50.0)


def drpcl_loss_with_mi(concat_pred, concat_true, model, beta=0.08,
                       cls_weight=CLS_WEIGHT, q_weight=Q_WEIGHT):
    """DRPCL loss for IHDP/JOBS. beta controls reg_loss weight."""
    q_pred = concat_pred[:, 2].view(-1, 1)
    gw_pred = concat_pred[:, 3].view(-1, 1)
    a_true = concat_true[:, 1].view(-1, 1)

    reg_loss = regression_loss(concat_true, concat_pred)

    if torch.isnan(gw_pred).any():
        gw_pred = torch.nan_to_num(gw_pred, nan=0.5)
    gw_pred = torch.clamp(gw_pred, 1e-6, 1 - 1e-6)
    cls_loss = F.binary_cross_entropy(gw_pred, a_true)

    q_loss = F.mse_loss(q_pred, _compute_q_target(gw_pred, a_true))

    mi_total, loss_iid, loss_bound = _compute_mi_total(model, concat_pred.device)

    loss_phase2 = cls_weight * cls_loss + q_weight * q_loss
    loss_phase3 = beta * reg_loss
    total_loss = loss_phase2 + loss_phase3 + mi_total

    return {
        'total_loss': total_loss, 'loss_phase2': loss_phase2, 'loss_phase3': loss_phase3,
        'reg_loss': reg_loss, 'cls_loss': cls_loss, 'q_loss': q_loss,
        'loss_iid': loss_iid, 'loss_bound': loss_bound,
    }



