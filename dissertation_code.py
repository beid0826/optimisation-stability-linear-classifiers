import numpy as np
import matplotlib.pyplot as plt

# Reproducibility + global settings
SEED = 42
TEST_SIZE = 0.2
N_ITER = 200

# Utilities: split + standardise + label conversion
def train_test_split_np(X, y, test_size=0.2, seed=0, shuffle=True):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    n_test = int(np.round(test_size * n))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def standardise_train_test(X_train, X_test, eps=1e-12):
    """
    Standardise using training mean/std only:
    X_std = (X - mean_train) / std_train.
    """
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma = np.where(sigma < eps, 1.0, sigma)
    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma

def to_pm1_labels(y):
    """
    Convert labels to {-1, +1}. Accepts y in {0,1} or {-1,+1}.
    """
    y = np.asarray(y).astype(int)
    uniq = set(np.unique(y))
    if uniq == {0, 1}:
        return 2*y - 1
    if uniq == {-1, 1}:
        return y
    raise ValueError(f"Unexpected label set {uniq}. Expected {0,1} or {-1,+1}.")

# Datasets: synthetic 2D and breast cancer
def make_synthetic_2d(n=200, sep=1.5, seed=0, feature_scale=(1.0, 1.0)):
    """
    Two Gaussian blobs with labels +/-1.
    feature_scale can intentionally create bad scaling, e.g. (1, 100).
    """
    rng = np.random.default_rng(seed)
    n_pos = n // 2
    n_neg = n - n_pos

    mu_pos = np.array([+sep, +sep])
    mu_neg = np.array([-sep, -sep])

    X_pos = rng.normal(loc=mu_pos, scale=1.0, size=(n_pos, 2))
    X_neg = rng.normal(loc=mu_neg, scale=1.0, size=(n_neg, 2))

    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(n_pos, dtype=int), -np.ones(n_neg, dtype=int)])

    X = X * np.array(feature_scale, dtype=float)
    return X, y

def load_breast_cancer_pm1():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = data.data.astype(float)
    y = to_pm1_labels(data.target)
    return X, y

# Objective: hinge + L2, subgradient, batch (sub)GD
def hinge_losses(z):
    return np.maximum(0.0, 1.0 - z)

def objective_hinge_l2(X, y, theta, theta0, lam):
    scores = X @ theta + theta0
    z = y * scores
    loss = hinge_losses(z).mean()
    reg = 0.5 * lam * (theta @ theta)
    return loss + reg

def subgrad_hinge_l2(X, y, theta, theta0, lam):
    n = X.shape[0]
    scores = X @ theta + theta0
    z = y * scores
    active = (z < 1.0)

    if np.any(active):
        y_active = y[active]
        X_active = X[active]
        sum_yx = (y_active[:, None] * X_active).sum(axis=0)
        sum_y = y_active.sum()
    else:
        sum_yx = np.zeros_like(theta)
        sum_y = 0.0

    grad_theta = -(1.0 / n) * sum_yx + lam * theta
    grad_theta0 = -(1.0 / n) * sum_y
    return grad_theta, grad_theta0

def batch_subgradient_descent(X, y, lam=0.1, eta=0.1, n_iter=200, theta0_init=0.0):
    """
    Batch subgradient descent for hinge + L2.
    Returns (theta, theta0, J_hist, norm_hist).
    """
    n, d = X.shape
    theta = np.zeros(d, dtype=float)
    theta0 = float(theta0_init)

    J_hist = np.zeros(n_iter, dtype=float)
    norm_hist = np.zeros(n_iter, dtype=float)

    for k in range(n_iter):
        J_hist[k] = objective_hinge_l2(X, y, theta, theta0, lam)
        norm_hist[k] = np.linalg.norm(theta)

        g_theta, g_theta0 = subgrad_hinge_l2(X, y, theta, theta0, lam)
        theta -= eta * g_theta
        theta0 -= eta * g_theta0

    return theta, theta0, J_hist, norm_hist

# Plotting helpers
def plot_histories(histories, title, ylabel, savepath=None, logy=False):
    plt.figure(figsize=(7, 4.5))
    for name, h in histories.items():
        plt.plot(h, label=name)
    if logy:
        plt.yscale("log")
        title = title + " (log scale)"
    plt.xlabel(r"Iteration $k$", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=13)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()

def run_eta_sweep(Xtr, ytr, lam, etas, n_iter=200):
    results = {}
    for eta in etas:
        _, _, J_hist, norm_hist = batch_subgradient_descent(Xtr, ytr, lam=lam, eta=eta, n_iter=n_iter)
        results[eta] = (J_hist, norm_hist)
    return results

# Prepare datasets (normal synthetic + breast cancer)
# Synthetic (normal scale)
X_syn, y_syn = make_synthetic_2d(n=200, sep=1.5, seed=SEED, feature_scale=(1.0, 1.0))
Xsyn_tr, Xsyn_te, ysyn_tr, ysyn_te = train_test_split_np(X_syn, y_syn, test_size=TEST_SIZE, seed=SEED)
Xsyn_tr_std, Xsyn_te_std, syn_mu, syn_sigma = standardise_train_test(Xsyn_tr, Xsyn_te)

print("Synthetic dataset:")
print("  X_train shape:", Xsyn_tr.shape, "X_test shape:", Xsyn_te.shape)
print("  labels in train:", np.unique(ysyn_tr), "labels in test:", np.unique(ysyn_te))

# Breast cancer
X_bc, y_bc = load_breast_cancer_pm1()
Xbc_tr, Xbc_te, ybc_tr, ybc_te = train_test_split_np(X_bc, y_bc, test_size=TEST_SIZE, seed=SEED)
Xbc_tr_std, Xbc_te_std, bc_mu, bc_sigma = standardise_train_test(Xbc_tr, Xbc_te)

print("\nBreast cancer dataset:")
print("  X_train shape:", Xbc_tr.shape, "X_test shape:", Xbc_te.shape)
print("  labels in train:", np.unique(ybc_tr), "labels in test:", np.unique(ybc_te))
print("  feature count:", X_bc.shape[1])

# Protocol sanity check
lam_default = 0.1
eta_default = 0.1
_, _, J_hist, norm_hist = batch_subgradient_descent(Xsyn_tr_std, ysyn_tr, lam=lam_default, eta=eta_default, n_iter=100)

print("\nSanity check (synthetic standardised):")
print("  Final objective:", J_hist[-1])
print("  Final ||theta||:", norm_hist[-1])
print("  First 5 J values:", J_hist[:5])

# Learning-rate sweep (synthetic standardised)
etas = [1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0]
eta_results = run_eta_sweep(Xsyn_tr_std, ysyn_tr, lam=lam_default, etas=etas, n_iter=N_ITER)

loss_curves = {rf"$\eta={eta:g}$": eta_results[eta][0] for eta in etas}
norm_curves = {rf"$\eta={eta:g}$": eta_results[eta][1] for eta in etas}

plot_histories(
    loss_curves,
    r"Synthetic (standardised): loss vs iteration",
    r"$J(\theta^{(k)},\theta_0^{(k)})$",
    savepath="fig_syn_std_eta_loss.pdf"
)

plot_histories(
    norm_curves,
    r"Synthetic (standardised): $\|\theta\|$ vs iteration",
    r"$\|\theta^{(k)}\|$",
    savepath="fig_syn_std_eta_norm.pdf"
)


# Scaling experiment (bad scaling: feature 2 x100)
X_bad, y_bad = make_synthetic_2d(n=200, sep=1.5, seed=SEED, feature_scale=(1.0, 100.0))
Xbad_tr, Xbad_te, ybad_tr, ybad_te = train_test_split_np(X_bad, y_bad, test_size=TEST_SIZE, seed=SEED)
Xbad_tr_std, Xbad_te_std, bad_mu, bad_sigma = standardise_train_test(Xbad_tr, Xbad_te)

print("\nBad-scaled synthetic dataset:")
print("  Train feature std (raw):", Xbad_tr.std(axis=0))

etas_wide = [1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0, 2.0, 5.0]

raw_results = run_eta_sweep(Xbad_tr, ybad_tr, lam=lam_default, etas=etas_wide, n_iter=N_ITER)
std_results = run_eta_sweep(Xbad_tr_std, ybad_tr, lam=lam_default, etas=etas_wide, n_iter=N_ITER)

raw_loss = {rf"$\eta={eta:g}$": raw_results[eta][0] for eta in etas_wide}
raw_norm = {rf"$\eta={eta:g}$": raw_results[eta][1] for eta in etas_wide}
std_loss = {rf"$\eta={eta:g}$": std_results[eta][0] for eta in etas_wide}
std_norm = {rf"$\eta={eta:g}$": std_results[eta][1] for eta in etas_wide}

# Use log scale to avoid spikes hiding the stable curves
plot_histories(raw_loss,
               r"Bad-scaled raw: loss vs iteration ($\lambda=0.1$)",
               r"$J(\theta^{(k)},\theta_0^{(k)})$",
               savepath="fig_syn_bad_raw_loss.pdf",
               logy=True)

plot_histories(raw_norm,
               r"Bad-scaled raw : $\|\theta\|$ vs iteration ($\lambda=0.1$)",
               r"$\|\theta^{(k)}\|$",
               savepath="fig_syn_bad_raw_norm.pdf",
               logy=True)

plot_histories(std_loss,
               r"Bad-scaled standardised: loss vs iteration ($\lambda=0.1$)",
               r"$J(\theta^{(k)},\theta_0^{(k)})$",
               savepath="fig_syn_bad_std_loss.pdf")

plot_histories(std_norm,
               r"Bad-scaled standardised: $\|\theta\|$ vs iteration ($\lambda=0.1$)",
               r"$\|\theta^{(k)}\|$",
               savepath="fig_syn_bad_std_norm.pdf")
## Regularisation sweep
def run_lambda_sweep(Xtr, ytr, eta, lambdas, n_iter=200):
    results = {}
    for lam in lambdas:
        _, _, J_hist, norm_hist = batch_subgradient_descent(Xtr, ytr, lam=lam, eta=eta, n_iter=n_iter)
        results[lam] = (J_hist, norm_hist)
    return results

# Regularisation sweep: synthetic (standardised)
eta = 0.1
lambdas = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

lam_results = run_lambda_sweep(Xsyn_tr_std, ysyn_tr, eta=eta, lambdas=lambdas, n_iter=N_ITER)

loss_curves = {rf"$\lambda={lam:g}$": lam_results[lam][0] for lam in lambdas}
norm_curves = {rf"$\lambda={lam:g}$": lam_results[lam][1] for lam in lambdas}

plot_histories(loss_curves,
               r"Synthetic (standardised): loss vs iteration ($\lambda$ sweep)",
               r"$J(\theta^{(k)},\theta_0^{(k)})$",
               savepath="fig_syn_std_lambda_loss.pdf")

plot_histories(norm_curves,
               r"Synthetic (standardised): $\|\theta\|$ vs iteration ($\lambda$ sweep)",
               r"$\|\theta^{(k)}\|$",
               savepath="fig_syn_std_lambda_norm.pdf")

# Breast cancer: raw vs standardised (fixed eta, lambda)
eta = 0.01
lam = 0.1

_, _, J_raw, N_raw = batch_subgradient_descent(Xbc_tr, ybc_tr, lam=lam, eta=eta, n_iter=N_ITER)
_, _, J_std, N_std = batch_subgradient_descent(Xbc_tr_std, ybc_tr, lam=lam, eta=eta, n_iter=N_ITER)

plot_histories({"raw": J_raw, "standardised": J_std},
               r"Breast cancer: raw vs standardised (loss)",
               r"$J(\theta^{(k)},\theta_0^{(k)})$",
               savepath="fig_bc_raw_vs_std_loss.pdf")

plot_histories({"raw": N_raw, "standardised": N_std},
               r"Breast cancer: raw vs standardised ($\|\theta\|$)",
               r"$\|\theta^{(k)}\|$",
               savepath="fig_bc_raw_vs_std_norm.pdf")

# Breast cancer: learning-rate sweep (compare raw vs standardised)
etas_bc = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
lam = 0.1

bc_raw_results = run_eta_sweep(Xbc_tr, ybc_tr, lam=lam, etas=etas_bc, n_iter=N_ITER)
bc_std_results = run_eta_sweep(Xbc_tr_std, ybc_tr, lam=lam, etas=etas_bc, n_iter=N_ITER)

raw_loss = {rf"$\eta={eta:g}$": bc_raw_results[eta][0] for eta in etas_bc}
std_loss = {rf"$\eta={eta:g}$": bc_std_results[eta][0] for eta in etas_bc}

plot_histories(raw_loss,
               r"Breast cancer raw: loss vs iteration ($\eta$ sweep)",
               r"$J(\theta^{(k)},\theta_0^{(k)})$",
               savepath="fig_bc_raw_eta_loss.pdf")

plot_histories(std_loss,
               r"Breast cancer standardised: loss vs iteration ($\eta$ sweep)",
               r"$J(\theta^{(k)},\theta_0^{(k)})$",
               savepath="fig_bc_std_eta_loss.pdf")