import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, LinearOperator, ArpackNoConvergence
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import gc

np.random.seed(42)

# ---------- plotting ----------
LINEWIDTH_BASE = 2.2
LINEWIDTH_ANALYTIC = 2 * LINEWIDTH_BASE
COLOR_ANALYTIC = '#4C72B0'
COLOR_SCATTER_PURE  = '#666666'
COLOR_SCATTER_MIXED = '#999999'
SCATTER_SIZE = 100

plt.rcParams.update({
    "font.size": 24, "lines.linewidth": LINEWIDTH_BASE, "legend.fontsize": 20,
    "axes.labelsize": 26, "xtick.labelsize": 24, "ytick.labelsize": 24,
    "axes.grid": False, "text.usetex": False,
})

def ticks_in(ax):
    ax.tick_params(direction='in', length=8, width=1.6, top=True, right=True)

# ---------- Pauli ----------
sx = sp.csr_matrix([[0,1],[1,0]], dtype=np.complex128)
sz = sp.csr_matrix([[1,0],[0,-1]], dtype=np.complex128)
sm = sp.csr_matrix([[0,0],[1,0]], dtype=np.complex128)
sp_dag = sm.conj().T
I2 = sp.identity(2, dtype=np.complex128, format='csr')

def kron_chain(ops):
    out = ops[0]
    for op in ops[1:]:
        out = sp.kron(out, op, format='csr')
    return out

# ---------- build H, L_ops ----------
def single_site_ops(N):
    sx_sites, sz_sites, sm_sites, sp_sites = [], [], [], []
    for i in range(N):
        ops = [I2]*N
        ops_i = ops.copy(); ops_i[i]=sx;     sx_sites.append(kron_chain(ops_i))
        ops_i = ops.copy(); ops_i[i]=sz;     sz_sites.append(kron_chain(ops_i))
        ops_i = ops.copy(); ops_i[i]=sm;     sm_sites.append(kron_chain(ops_i))
        ops_i = ops.copy(); ops_i[i]=sp_dag; sp_sites.append(kron_chain(ops_i))
    return {'sx':sx_sites,'sz':sz_sites,'sm':sm_sites,'sp':sp_sites}

def build_H_and_jumps(N, J=1.0, g=1.0, gamma=0.5, beta=1.0):
    d = 2**N
    cache = single_site_ops(N)
    H = sp.csr_matrix((d,d), dtype=np.complex128)
    for i in range(N-1):
        H += -J * (cache['sz'][i] @ cache['sz'][i+1])
    for i in range(N):
        H += -g * cache['sx'][i]
    L_ops = []
    for i in range(N):
        L_ops.append(np.sqrt(gamma) * cache['sm'][i])
        L_ops.append(np.sqrt(gamma*np.exp(-beta)) * cache['sp'][i])
    D_ops = [ (L.conj().T @ L).tocsr() for L in L_ops ]
    return H.tocsr(), L_ops, D_ops

# ---------- Liouvillian (matrix-free) ----------
def _vec(mat):
    return np.asarray(mat, dtype=np.complex128, order='F').reshape((-1,), order='F')

def make_L_ops_linear_operators(H, L_ops, D_ops):
    d = H.shape[0]; dd = d*d
    def L_action(v):
        rho = np.array(v, dtype=np.complex128).reshape((d,d), order='F')
        tmp = -1j*(H @ rho - rho @ H)
        for L, D in zip(L_ops, D_ops):
            tmp += (L @ rho) @ (L.conj().T) - 0.5*(D @ rho + rho @ D)
        return _vec(tmp)
    def L_dag_action(v):
        X = np.array(v, dtype=np.complex128).reshape((d,d), order='F')
        tmp = +1j*(H @ X - X @ H)
        for L, D in zip(L_ops, D_ops):
            tmp += (L.conj().T @ X) @ L - 0.5*(D @ X + X @ D)
        return _vec(tmp)
    L = LinearOperator((dd, dd), matvec=L_action, dtype=np.complex128)
    Ldag = LinearOperator((dd, dd), matvec=L_dag_action, dtype=np.complex128)
    return L, Ldag

# ---------- explicit sparse L (for N ≤ 4) ----------
def build_L_sparse_from_ops(H, L_ops, D_ops):
    d = H.shape[0]
    I_d = sp.identity(d, dtype=np.complex128, format='csr')
    Ls = -1j * (sp.kron(I_d, H, format='csr') - sp.kron(H.T.conj(), I_d, format='csr'))
    for L in L_ops:
        D = (L.conj().T @ L).tocsr()
        Ls += sp.kron(L.conj(), L, format='csr') \
              - 0.5*( sp.kron(I_d, D, format='csr') + sp.kron(D.T, I_d, format='csr') )
    return Ls.tocsr()

# ---------- pairing & biorth ----------
def pair_left_to_right(valsR, VR, valsL, WL):
    """
    Pair (valsR, VR) of L with left vectors obtained as right eigenvectors (valsL, WL) of L^†.
    For each λ_R, pick the column of WL whose λ_L^* is closest to λ_R.
    NOTE: valsR MUST be the first argument (not VR).
    """
    kR = VR.shape[1]
    WLm = np.zeros_like(VR, dtype=np.complex128)
    used = set(); valsL_conj = np.conj(valsL)
    for j in range(kR):
        diffs = np.abs(valsL_conj - valsR[j])
        for _ in range(len(valsL)):
            i = int(np.argmin(diffs))
            if i not in used:
                used.add(i); WLm[:, j] = WL[:, i]; break
            diffs[i] = np.inf
    return WLm  # columns are 'column-form' left eigenvectors of L

def biorthonormalize_left_diag(W, V, eps=1e-14):
    """
    Enforce W^H V = I by column-wise scaling of W only (do NOT mix columns).
    Here W are column-form left eigenvectors (i.e., right eigenvectors of L^†).
    """
    s = np.einsum('ij,ij->j', W.conj(), V)  # diag(W^H V)
    s = np.where(np.abs(s) < eps, 1.0, s)
    Wn = W / s
    return Wn, V

# ---------- printing helpers ----------
def print_top_eigs(vals, j2, ss_idx=None, zero_tol=1e-12, top=4, label=""):
    re = np.real(vals); im = np.imag(vals)
    is_zero = np.abs(re) <= zero_tol
    nz = np.where(~is_zero)[0]
    jgap = int(nz[np.argmin(np.abs(re[nz]))]) if nz.size>0 else None

    hdr = f"\n=== Eigenvalues (sorted by Re λ desc){' — '+label if label else ''} ==="
    print(hdr)
    print(f"{'idx':>3}  {'λ (complex)':>26}  {'Re':>12}  {'Im':>12}  {'|Re|':>12}  {'zero?':>7}  {'ss?':>4}  {'L2?':>4}  {'gap?':>5}")
    m = min(top, len(vals))
    for i in range(m):
        mark_ss  = '✓' if (ss_idx is not None and i == ss_idx) else ''
        mark_L2  = '✓' if (j2      is not None and i == j2)      else ''
        mark_gap = '✓' if (jgap    is not None and i == jgap)    else ''
        print(f"{i:3d}  {vals[i]:>26.16e}  {re[i]:>12.3e}  {im[i]:>12.3e}  {abs(re[i]):>12.3e}  {str(is_zero[i]):>7}  {mark_ss:>4}  {mark_L2:>4}  {mark_gap:>5}")

    if ss_idx is not None:
        print(f"-> steady-state index   = {ss_idx}, λ_ss = {vals[ss_idx]:.6e} (Re={re[ss_idx]:.6e})")
    if jgap is not None:
        print(f"-> spectral-gap index   = {jgap},  λ_gap= {vals[jgap]:.6e} (Re={re[jgap]:.6e})")
    if j2 is not None:
        print(f"-> L2 chosen index      = {j2},   λ2   = {vals[j2]:.6e} (Re={re[j2]:.6e})")
    if jgap is not None and j2 is not None and j2 != jgap:
        print("!! WARNING: L2 != spectral-gap (consider increasing K_EIGEN/ncv or relaxing zero_tol).")
    print()

def print_L1_and_L2(W, ss_idx, j2, d, show_mats_up_to=4):
    """
    打印 L1 是否为 identity（按最优缩放/相位对齐），并给出 L2 的简要信息。
    """
    vecI = np.eye(d, dtype=np.complex128).reshape(-1, order='F')
    denom = np.vdot(vecI, vecI)  # = d
    # ----- L1 -----
    W1 = W[:, ss_idx]
    alpha = np.vdot(vecI, W1) / denom       # best-fit scale for α I
    L1_mat = W1.reshape((d,d), order='F') - alpha * np.eye(d)
    resid = np.linalg.norm(L1_mat) / max(np.linalg.norm(W1.reshape((d,d),order='F')), 1e-15)
    print(f"[L1 check] best α such that W1≈αI: |W1-αI|_F / |W1|_F = {resid:.3e}")
    if d <= show_mats_up_to:
        print("  L1 (best-aligned to αI) matrix (rounded):")
        print(np.round((W1.reshape((d,d),order='F')/alpha) if np.abs(alpha)>1e-15 else W1.reshape((d,d),order='F'), 4))

    # ----- L2 (optional small-N preview) -----
    if j2 is not None:
        W2 = W[:, j2]
        if d <= show_mats_up_to:
            print("  L2 matrix (rounded):")
            print(np.round(W2.reshape((d,d), order='F'), 4))

# ---------- selection (exclude identity explicitly) ----------
def select_L2_index_excluding_identity(vals, W, d, zero_tol=1e-10):
    re = np.real(vals)
    vecI = np.eye(d, dtype=np.complex128).reshape(-1, order='F')
    vecI = vecI / np.linalg.norm(vecI)
    overlaps = np.abs(W.conj().T @ vecI)

    near_zero = np.where(np.abs(re) <= 10*zero_tol)[0]
    if near_zero.size > 0:
        ss_idx = int(near_zero[np.argmax(overlaps[near_zero])])
    else:
        ss_idx = int(np.argmax(overlaps))

    cand = np.where((np.arange(len(vals)) != ss_idx) & (np.abs(re) > zero_tol))[0]
    if cand.size == 0:
        return ss_idx, None
    j2 = int(cand[np.argmin(np.abs(re[cand]))])
    return ss_idx, j2

# ---------- hybrid eigensolver ----------
def leading_modes_biorth_hybrid(N, H, L_ops, D_ops, k=12, tol=1e-10, maxiter=50000, ncv=32, seed=42):
    d = H.shape[0]; dd = d*d
    if dd <= 256:
        Ls = build_L_sparse_from_ops(H, L_ops, D_ops)
        Ld = Ls.toarray()
        valsR, VR = np.linalg.eig(Ld)               # right eig of L
        valsL, WL = np.linalg.eig(Ld.conj().T)      # right eig of L^† (column-form LEFT of L)
        idx = np.argsort(-np.real(valsR))
        valsR, VR = valsR[idx], VR[:, idx]
        WL = pair_left_to_right(valsR, VR, valsL, WL)
        k_eff = min(k, VR.shape[1])
        valsR, VR, WL = valsR[:k_eff], VR[:, :k_eff], WL[:, :k_eff]
    else:
        L, Ldag = make_L_ops_linear_operators(H, L_ops, D_ops)
        rng = np.random.default_rng(seed)
        v0 = rng.normal(size=dd) + 1j*rng.normal(size=dd); v0 /= np.linalg.norm(v0)
        k_use = min(k, dd-2)
        try:
            valsR, VR = eigs(L, k=k_use, which='LR', tol=tol, maxiter=maxiter,
                             ncv=max(2*k_use+2, ncv), v0=v0)
        except ArpackNoConvergence as e:
            valsR, VR = e.eigenvalues, e.eigenvectors
            if valsR is None or VR is None: raise
        try:
            valsL, WL = eigs(Ldag, k=VR.shape[1], which='LR', tol=tol, maxiter=maxiter,
                             ncv=max(2*VR.shape[1]+2, ncv), v0=v0)
        except ArpackNoConvergence as e:
            valsL, WL = e.eigenvalues, e.eigenvectors
            if valsL is None or WL is None: raise
        idx = np.argsort(-np.real(valsR))
        valsR, VR = valsR[idx], VR[:, idx]
        WL = pair_left_to_right(valsR, VR, valsL, WL)

    # single-sided biorthonormalization
    WL, VR = biorthonormalize_left_diag(WL, VR)
    return valsR, WL, VR

# ---------- analytics ----------
def _unvec_to_mat(vec, d):
    return np.array(vec, dtype=np.complex128).reshape((d,d), order='F')

def analytic_results_k2(Wcol, Vcol, d):
    """
    解析均值/方差（数值稳定版）：
    Var = || L_tr ||_F^2 / [ d*(d+1) * |O_k|^2 ]  (Haar)
    Var = || L_tr ||_F^2 / [ d*(d^2+1) * |O_k|^2 ] (Hilbert–Schmidt)
    其中 L_tr = L - (Tr L)/d * I ；O_k = <W_k, V_k>.
    """
    # 左本征“算符”矩阵
    Lk = np.array(Wcol, dtype=np.complex128).reshape((d, d), order='F')

    # 归一因子（按理 ~1；为了稳健用模平方）
    Ok = complex(np.vdot(Wcol, Vcol))
    Ok2 = max(1e-30, np.abs(Ok)**2)  # 防止极罕见的数值 0

    # 无迹投影 —— 避免 T1 - |T0|^2/d 的灾难性相消
    T0 = np.trace(Lk)
    Lk_tr = Lk - (T0 / d) * np.eye(d, dtype=np.complex128)
    numer = float(np.vdot(Lk_tr, Lk_tr).real)

    var_haar = numer / (d * (d + 1) * Ok2)
    var_hs   = numer / (d * (d**2 + 1) * Ok2)

    # 解析均值（同样用 |Ok|）
    mean_abs = float(np.abs(T0) / (d * max(np.sqrt(Ok2), 1e-15)))

    return mean_abs, var_haar, var_hs


# ---------- samplers ----------
def sample_haar_pure_vec(d):
    x = (np.random.normal(size=d) + 1j*np.random.normal(size=d))/np.sqrt(2)
    x /= np.linalg.norm(x)
    rho = np.outer(x, x.conj())
    return _vec(rho)

def sample_hs_mixed_vec(d):
    G = (np.random.normal(size=(d,d)) + 1j*np.random.normal(size=(d,d)))/np.sqrt(2)
    A = G @ G.conj().T; A /= np.trace(A)
    return _vec(A)

class ComplexStats:
    def __init__(self):
        self.n = 0; self.sum_z = 0+0j; self.sum_abs2 = 0.0
    def add(self, z):
        self.n += 1; self.sum_z += z; self.sum_abs2 += (np.abs(z)**2)
    def mean(self):
        return self.sum_z / max(self.n, 1)
    def var(self):
        if self.n == 0: return np.nan
        m = self.mean(); return float(self.sum_abs2 / self.n - np.abs(m)**2)

# ---------- experiment ----------
def run_experiment(N_list=range(1, 7), NUM_PURE=2000, NUM_MIXED=2000, K_EIGEN=12,
                   J=1.0, g=1.0, gamma=0.5, beta=0.01,
                   tol=1e-10, maxiter=50000, ncv=32, zero_tol=1e-12):
    results = {'analytic': {'mean_abs': [], 'var_haar': [], 'var_hs': []},
               'pure': {'mean_abs': [], 'var': []},
               'mixed': {'mean_abs': [], 'var': []},
               'N': [], 'd': []}

    for N in tqdm(N_list, desc="N loop"):
        d = 2**N
        try:
            H, L_ops, D_ops = build_H_and_jumps(N, J=J, g=g, gamma=gamma, beta=beta)
            vals, W, V = leading_modes_biorth_hybrid(
                N, H, L_ops, D_ops, k=K_EIGEN, tol=tol, maxiter=maxiter, ncv=ncv, seed=42
            )
        except Exception as e:
            print(f"[eigs failed] N={N}, d^2={d*d}: {e}")
            results['analytic']['mean_abs'].append(np.nan)
            results['analytic']['var_haar'].append(np.nan)
            results['analytic']['var_hs'].append(np.nan)
            results['pure']['mean_abs'].append(np.nan);  results['pure']['var'].append(np.nan)
            results['mixed']['mean_abs'].append(np.nan); results['mixed']['var'].append(np.nan)
            results['N'].append(N); results['d'].append(d); continue

        # ---- identify L1 (steady state) and L2 (spectral gap) ----
        ss_idx, j2 = select_L2_index_excluding_identity(vals, W, d, zero_tol=zero_tol)

        # 打印前 4 个特征值（含 0），并标注 ss / L2 / gap
        print_top_eigs(vals, j2, ss_idx=ss_idx, zero_tol=zero_tol, top=4, label=f"N={N}, d={d}")
        # 打印 L1（应为 identity）及 L2（小 d 时显示矩阵）
        print_L1_and_L2(W, ss_idx, j2, d, show_mats_up_to=8)

        if (j2 is None) or (j2 >= W.shape[1]):
            results['analytic']['mean_abs'].append(np.nan)
            results['analytic']['var_haar'].append(np.nan)
            results['analytic']['var_hs'].append(np.nan)
            results['pure']['mean_abs'].append(np.nan);  results['pure']['var'].append(np.nan)
            results['mixed']['mean_abs'].append(np.nan); results['mixed']['var'].append(np.nan)
            results['N'].append(N); results['d'].append(d); continue

        # ---- target mode columns ----
        W2 = W[:, j2]; V2 = V[:, j2]

        # ---- analytic ----
        mean_abs, var_haar, var_hs = analytic_results_k2(W2, V2, d)
        results['analytic']['mean_abs'].append(mean_abs)
        results['analytic']['var_haar'].append(var_haar)
        results['analytic']['var_hs'].append(var_hs)

        # ---- numeric sampling ----
        cs_p, cs_m = ComplexStats(), ComplexStats()
        for _ in range(NUM_PURE):  cs_p.add(np.vdot(W2, sample_haar_pure_vec(d)))
        for _ in range(NUM_MIXED): cs_m.add(np.vdot(W2, sample_hs_mixed_vec(d)))
        results['pure']['var'].append(cs_p.var())
        results['mixed']['var'].append(cs_m.var())
        results['pure']['mean_abs'].append(abs(cs_p.mean()))
        results['mixed']['mean_abs'].append(abs(cs_m.mean()))

        results['N'].append(N); results['d'].append(d)
        del H, L_ops, D_ops, vals, W, V; gc.collect()

    return results

# ---------- plotting ----------
def _pos_or_nan(x):
    x = np.asarray(x, dtype=float); x[~(x>0)] = np.nan; return x

def plot_all(results, save_prefix='k2_analytic_numeric'):
    Ns = np.array(results['N']); ds = np.array(results['d'], dtype=float)

    fig = plt.figure(figsize=(8,6)); ax = plt.gca()
    ax.plot(Ns, results['analytic']['var_haar'], '-',  color=COLOR_ANALYTIC, linewidth=LINEWIDTH_ANALYTIC, label='Haar (analytic)')
    ax.plot(Ns, results['analytic']['var_hs'],   '--', color=COLOR_ANALYTIC, linewidth=LINEWIDTH_ANALYTIC, label='HS (analytic)')
    ax.scatter(Ns, results['pure']['var'],  s=SCATTER_SIZE, c=COLOR_SCATTER_PURE,  marker='o', label='Haar (numeric)')
    ax.scatter(Ns, results['mixed']['var'], s=SCATTER_SIZE, c=COLOR_SCATTER_MIXED, marker='s', label='HS (numeric)')
    ax.set_xlabel('N'); ax.set_ylabel(r'Var($a_2$)'); ticks_in(ax); ax.legend(frameon=False)
    plt.tight_layout(); plt.savefig(f'{save_prefix}_variance_vsN.png', dpi=200); plt.show()

    fig = plt.figure(figsize=(8,6)); ax = plt.gca()
    ax.plot(ds, results['analytic']['var_haar'], '-',  color=COLOR_ANALYTIC, linewidth=LINEWIDTH_ANALYTIC, label='Haar (analytic)')
    ax.plot(ds, results['analytic']['var_hs'],   '--', color=COLOR_ANALYTIC, linewidth=LINEWIDTH_ANALYTIC, label='HS (analytic)')
    ax.scatter(ds, _pos_or_nan(results['pure']['var']),  s=SCATTER_SIZE, c=COLOR_SCATTER_PURE,  marker='o', label='Haar (numeric)')
    ax.scatter(ds, _pos_or_nan(results['mixed']['var']), s=SCATTER_SIZE, c=COLOR_SCATTER_MIXED, marker='s', label='HS (numeric)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$d=2^N$'); ax.set_ylabel(r'Var($a_2$)'); ticks_in(ax); ax.legend(frameon=False)
    plt.tight_layout(); plt.savefig(f'{save_prefix}_variance_vsD_loglog.png', dpi=200); plt.show()

    fig = plt.figure(figsize=(8,6)); ax = plt.gca()
    ax.plot(Ns, results['analytic']['mean_abs'], '-', color=COLOR_ANALYTIC, linewidth=LINEWIDTH_ANALYTIC, label='analytic')
    ax.scatter(Ns, results['pure']['mean_abs'],  s=SCATTER_SIZE, c=COLOR_SCATTER_PURE,  marker='o', label='Haar (numeric)')
    ax.scatter(Ns, results['mixed']['mean_abs'], s=SCATTER_SIZE, c=COLOR_SCATTER_MIXED, marker='s', label='HS (numeric)')
    ax.set_xlabel('N'); ax.set_ylabel(r'$|\langle a_2 \rangle|$'); ticks_in(ax); ax.legend(frameon=False)
    plt.tight_layout(); plt.savefig(f'{save_prefix}_average_vsN.png', dpi=200); plt.show()

    fig = plt.figure(figsize=(8,6)); ax = plt.gca()
    ax.plot(ds, results['analytic']['mean_abs'], '-', color=COLOR_ANALYTIC, linewidth=LINEWIDTH_ANALYTIC, label='analytic')
    ax.scatter(ds, _pos_or_nan(results['pure']['mean_abs']),  s=SCATTER_SIZE, c=COLOR_SCATTER_PURE,  marker='o', label='Haar (numeric)')
    ax.scatter(ds, _pos_or_nan(results['mixed']['mean_abs']), s=SCATTER_SIZE, c=COLOR_SCATTER_MIXED, marker='s', label='HS (numeric)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$d=2^N$'); ax.set_ylabel(r'$|\langle a_2 \rangle|$'); ticks_in(ax); ax.legend(frameon=False)
    plt.tight_layout(); plt.savefig(f'{save_prefix}_average_vsD_loglog.png', dpi=200); plt.show()


# ---------- example ----------
if __name__ == "__main__":
    # Tuning parameters
    N_LIST = range(1, 8)
    NUM_PURE  = 10000; NUM_MIXED = 10000; K_EIGEN = 12
    J, g = 1.0, 1.0; gamma, beta = 0.5, 0.1

    results = run_experiment(N_list=N_LIST, NUM_PURE=NUM_PURE, NUM_MIXED=NUM_MIXED,
                             K_EIGEN=K_EIGEN, J=J, g=g, gamma=gamma, beta=beta,
                             tol=1e-10, maxiter=50000, ncv=64, zero_tol=1e-12)
    plot_all(results)
    np.savez_compressed('results_k2_high_temp_1.npz', **results)
