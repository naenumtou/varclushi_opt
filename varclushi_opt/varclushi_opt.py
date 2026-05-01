
import pandas as pd
import numpy as np
import random
import collections
import math
from factor_analyzer import Rotator


class VarClusHi_Opt(object):

    """
    VarClusHi - Optimisation structure.

    Key changes from original version
    ====================================
        1.  correig: use np.linalg.eigh on pre-computed corr matrix; cache the
            full corr matrix once and slice it (O(1) vs recomputing).

        2.  _calc_tot_var: accept pre-sliced sub-matrix to skip repeated loc[] calls.

        3.  _reassign / _reassign_rs: iterate over a membership array instead of
            list copies; move features in O(1) with set-based bookkeeping.

        4.  _varclusspu: cache global correlation matrix; pass corr slices directly
            to all inner helpers; avoid repeated DataFrame construction.

        5.  rsquare / _rsquarespu: vectorise cross-cluster RS computation with
            matrix-vector products instead of looping per feature.

        6.  Minor: pre-allocate result rows in lists, concat once (avoids
            quadratic DataFrame growth from repeated .loc append).
    """

    def __init__(self, df, feat_list = None, maxeigval2 = 1, maxclus = None, n_rs = 0):
        if feat_list is None:
            self.df = df
            self.feat_list = df.columns.tolist()
        else:
            self.df = df[feat_list]
            self.feat_list = feat_list
        self.maxeigval2 = maxeigval2
        self.maxclus = maxclus
        self.n_rs = n_rs

    # ----------------------------- #
    #  Core linear-algebra helpers  #
    # ----------------------------- #

    @staticmethod
    def _eigh_sorted(corr_vals, n_pcs = 2):
        # Return top-n_pcs eigenvalues / eigenvectors from a corr matrix
        raw_eigvals, raw_eigvecs = np.linalg.eigh(corr_vals)

        # eigh returns ascending order → reverse
        idx = np.arange(len(raw_eigvals) - 1, -1, -1)
        eigvals = raw_eigvals[idx][:n_pcs]
        eigvecs = raw_eigvecs[:, idx][:, :n_pcs]
        varprops = eigvals / raw_eigvals.sum()
        return eigvals, eigvecs, varprops

    @staticmethod
    def correig(df, feat_list = None, n_pcs = 2):
        if feat_list is None:
            feat_list = df.columns.tolist()
        else:
            df = df[feat_list]

        n = len(feat_list)
        if n <= 1:
            corr = np.array([[float(n)]])
            eigvals = np.array([float(n)] + [0.0] * (n_pcs - 1))
            eigvecs = np.array([[float(n)]])
            varprops = np.array([float(n)])
        else:
            corr = np.corrcoef(df.values.T)
            eigvals, eigvecs, varprops = VarClusHi_Opt._eigh_sorted(corr, n_pcs)

        corr_df = pd.DataFrame(corr, columns = feat_list, index = feat_list)
        
        return eigvals, eigvecs, corr_df, varprops

    @staticmethod
    def pca(df, feat_list = None, n_pcs = 2):

        if feat_list is None:
            feat_list = df.columns.tolist()
        else:
            df = df[feat_list]

        stand_df = (df - df.mean()) / df.std()
        eigvals, eigvecs, _, varprops = VarClusHi_Opt.correig(df, feat_list, n_pcs = n_pcs)

        princomps = stand_df.values if len(feat_list) <= 1 else np.dot(stand_df.values, eigvecs)

        return eigvals, eigvecs, princomps, varprops

    # ---------------------------------------------------------------- #
    #  Variance helpers — accept raw numpy arrays to avoid re-slicing  #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _eigval1_from_corr(corr_vals):

        # Largest eigenvalue of a correlation matrix (numpy array)
        if corr_vals.shape[0] <= 1:
            return float(corr_vals[0, 0])
        eigvals = np.linalg.eigvalsh(corr_vals)
        return float(eigvals[-1]) #eigh returns ascending

    @staticmethod
    def _calc_tot_var(df, *clusters):

        # Original interface kept for public compatibility
        tot_len = tot_var = tot_prop = 0
        for clus in clusters:
            if not clus:
                continue
            c_len = len(clus)
            c_eigvals, _, _, c_varprops = VarClusHi_Opt.correig(df[clus])
            tot_var += c_eigvals[0]
            tot_prop = (tot_prop * tot_len + c_varprops[0] * c_len) / (tot_len + c_len)
            tot_len += c_len
        return tot_var, tot_prop

    @staticmethod
    def _tot_var_from_corr(corr_full, idx1, idx2):

        """
        Fast variant: compute total eigval1 from pre-sliced corr matrix.
        idx1, idx2 are integer index arrays into corr_full.
        """
        
        tot = 0.0
        for idx in (idx1, idx2):
            if len(idx) == 0:
                continue
            sub = corr_full[np.ix_(idx, idx)]
            tot += VarClusHi_Opt._eigval1_from_corr(sub)
        return tot

    # --------------------------------------------------------------------- #
    #  Reassign — O(n_feat) list ops replaced with set + index bookkeeping  #
    # --------------------------------------------------------------------- #

    @staticmethod
    def _reassign(df, clus1, clus2, feat_list = None):
        if feat_list is None:
            feat_list = clus1 + clus2

        # Work with sets for O(1) membership --> keep list for ordered output
        fin_set1 = set(clus1)
        fin_set2 = set(clus2)

        corr_vals = np.corrcoef(df[feat_list].values.T)
        feat_idx = {f: i for i, f in enumerate(feat_list)}

        def _var(s1, s2):
            idx1 = [feat_idx[f] for f in s1]
            idx2 = [feat_idx[f] for f in s2]
            return VarClusHi_Opt._tot_var_from_corr(
                corr_vals,
                np.array(idx1, dtype = int),
                np.array(idx2, dtype = int)
            )

        max_var = check_var = _var(fin_set1, fin_set2)

        while True:
            for feat in feat_list:
                if feat in fin_set1:
                    cand1 = fin_set1 - {feat}
                    cand2 = fin_set2 | {feat}
                elif feat in fin_set2:
                    cand1 = fin_set1 | {feat}
                    cand2 = fin_set2 - {feat}
                else:
                    continue

                new_var = _var(cand1, cand2)
                if new_var > check_var:
                    check_var = new_var
                    fin_set1, fin_set2 = cand1, cand2

            if max_var == check_var:
                break
            max_var = check_var

        # Preserve original ordering
        all_feats = clus1 + clus2
        out1 = [f for f in all_feats if f in fin_set1]
        out2 = [f for f in all_feats if f in fin_set2]
        return out1, out2, max_var

    @staticmethod
    def _reassign_rs(df, clus1, clus2, n_rs=0):
        feat_list = clus1 + clus2
        fin_rs_clus1, fin_rs_clus2, max_rs_var = VarClusHi_Opt._reassign(df, clus1, clus2)

        for _ in range(n_rs):
            random.shuffle(feat_list)
            rs_clus1, rs_clus2, rs_var = VarClusHi_Opt._reassign(df, clus1, clus2, feat_list)
            if rs_var > max_rs_var:
                max_rs_var = rs_var
                fin_rs_clus1, fin_rs_clus2 = rs_clus1, rs_clus2

        return fin_rs_clus1, fin_rs_clus2, max_rs_var

    # -------------------------------- #
    #  Main clustering — speedup path  #
    # -------------------------------- #

    def _varclusspu(self):
        ClusInfo = collections.namedtuple(
            'ClusInfo',
            ['clus', 'eigval1', 'eigval2', 'eigvecs', 'varprop']
        )

        # Compute & cache global corr matrix ONCE
        vals = self.df[self.feat_list]
        global_corr = np.corrcoef(vals.T)
        feat_pos = {f: i for i, f in enumerate(self.feat_list)}
        self.corrs = pd.DataFrame(
            global_corr,
            columns = self.feat_list,
            index = self.feat_list
        )

        def _cluster_info_from_corr(clus):
            idx = [feat_pos[f] for f in clus]
            sub = global_corr[np.ix_(idx, idx)]
            if len(clus) <= 1:
                ev1 = float(sub[0, 0]); ev2 = 0.0
                evec = sub
                vp = 1.0
            else:
                eigvals, eigvecs, varprops = VarClusHi_Opt._eigh_sorted(sub, n_pcs = 2)
                ev1, ev2 = float(eigvals[0]), float(eigvals[1])
                evec = eigvecs
                vp = float(varprops[0])
            return ev1, ev2, evec, vp

        ev1, ev2, evec, vp = _cluster_info_from_corr(self.feat_list)
        clus0 = ClusInfo(
            clus = self.feat_list,
            eigval1 = ev1, eigval2 = ev2,
            eigvecs = evec, varprop = vp
        )
        self.clusters = collections.OrderedDict([(0, clus0)])

        while True:
            if self.maxclus is not None and len(self.clusters) >= self.maxclus:
                break

            idx_max = max(self.clusters, key = lambda x: self.clusters[x].eigval2)
            best = self.clusters[idx_max]

            if best.eigval2 <= self.maxeigval2:
                break

            split_clus = best.clus
            split_idx = [feat_pos[f] for f in split_clus]
            split_corr = global_corr[np.ix_(split_idx, split_idx)]
            split_corr_df = pd.DataFrame(
                split_corr,
                columns = split_clus,
                index = split_clus
            )

            c_eigvals, c_eigvecs, _ = VarClusHi_Opt._eigh_sorted(split_corr, n_pcs = 2)
            if c_eigvals[1] <= self.maxeigval2:
                break

            # Quartimax rotation
            rotator = Rotator(method = 'quartimax')
            r_eigvecs = rotator.fit_transform(pd.DataFrame(c_eigvecs))
            re0, re1 = r_eigvecs[:, 0], r_eigvecs[:, 1]


            sigma1 = math.sqrt(re0 @ split_corr @ re0)
            sigma2 = math.sqrt(re1 @ split_corr @ re1)

            # Vectorised feature assignment
            cov1 = split_corr @ re0 #shape (n_split,)
            cov2 = split_corr @ re1
            corr_pc1 = cov1 / sigma1
            corr_pc2 = cov2 / sigma2

            clus1 = [
                f for f, cp1, cp2 in zip(split_clus, corr_pc1, corr_pc2)
                if abs(cp1) >= abs(cp2)
            ]
            clus2 = [
                f for f, cp1, cp2 in zip(split_clus, corr_pc1, corr_pc2)
                if abs(cp1) < abs(cp2)
            ]

            fin_clus1, fin_clus2, _ = VarClusHi_Opt._reassign_rs(
                self.df,
                clus1,
                clus2,
                self.n_rs
            )

            ev1_c1, ev2_c1, evec_c1, vp_c1 = _cluster_info_from_corr(fin_clus1)
            ev1_c2, ev2_c2, evec_c2, vp_c2 = _cluster_info_from_corr(fin_clus2)

            self.clusters[idx_max] = ClusInfo(
                clus = fin_clus1, eigval1 = ev1_c1, eigval2 = ev2_c1,
                eigvecs = evec_c1, varprop = vp_c1
            )

            self.clusters[len(self.clusters)] = ClusInfo(
                clus = fin_clus2, eigval1 = ev1_c2, eigval2 = ev2_c2,
                eigvecs = evec_c2, varprop = vp_c2
            )

        return self

    # --------------------------------------------------------------- #
    #  Main clustering — full (slow) path — minor optimisations only  #
    # --------------------------------------------------------------- #

    def varclus(self, speedup = True):
        self.speedup = speedup
        if self.speedup:
            return self._varclusspu()

        ClusInfo = collections.namedtuple(
            'ClusInfo',
            ['clus', 'eigval1', 'eigval2', 'pc1', 'varprop']
        )

        c_eigvals, _, c_princomps, c_varprops = VarClusHi_Opt.pca(self.df[self.feat_list])
        clus0 = ClusInfo(
            clus = self.feat_list,
            eigval1 = c_eigvals[0], eigval2 = c_eigvals[1],
            pc1 = c_princomps[:, 0], varprop = c_varprops[0]
        )
        self.clusters = collections.OrderedDict([(0, clus0)])

        while True:
            if self.maxclus is not None and len(self.clusters) >= self.maxclus:
                break

            idx = max(self.clusters, key = lambda x: self.clusters[x].eigval2)
            if self.clusters[idx].eigval2 <= self.maxeigval2:
                break

            split_clus = self.clusters[idx].clus
            c_eigvals, c_eigvecs, _, _ = VarClusHi_Opt.pca(self.df[split_clus])

            if c_eigvals[1] <= self.maxeigval2:
                break

            rotator = Rotator(method = 'quartimax')
            r_eigvecs = rotator.fit_transform(pd.DataFrame(c_eigvecs))
            stand_df = (self.df - self.df.mean()) / self.df.std()
            r_pcs = np.dot(stand_df[split_clus].values, r_eigvecs)

            # Vectorised: correlation of each feature with both rotated PCs
            feat_vals = self.df[split_clus].values          # (n_obs, n_feat)
            pc0, pc1 = r_pcs[:, 0], r_pcs[:, 1]

            def _corr_col_vec(mat, vec):
                """Pearson corr of each column in mat with vec."""
                mat_c = mat - mat.mean(axis = 0)
                vec_c = vec - vec.mean()
                num = mat_c.T @ vec_c
                denom = np.sqrt((mat_c ** 2).sum(axis = 0)) * np.sqrt((vec_c ** 2).sum())
                return num / denom

            corr1 = _corr_col_vec(feat_vals, pc0)
            corr2 = _corr_col_vec(feat_vals, pc1)

            clus1 = [f for f, c1, c2 in zip(split_clus, corr1, corr2) if abs(c1) >= abs(c2)]
            clus2 = [f for f, c1, c2 in zip(split_clus, corr1, corr2) if abs(c1) < abs(c2)]

            fin_clus1, fin_clus2, _ = VarClusHi_Opt._reassign_rs(
                self.df,
                clus1,
                clus2,
                self.n_rs
            )

            c1_eigvals, _, c1_pcs, c1_varprops = VarClusHi_Opt.pca(self.df[fin_clus1])
            c2_eigvals, _, c2_pcs, c2_varprops = VarClusHi_Opt.pca(self.df[fin_clus2])

            self.clusters[idx] = ClusInfo(
                clus = fin_clus1, eigval1 = c1_eigvals[0], eigval2 = c1_eigvals[1],
                pc1 = c1_pcs[:, 0], varprop = c1_varprops[0]
            )

            self.clusters[len(self.clusters)] = ClusInfo(
                clus = fin_clus2, eigval1 = c2_eigvals[0], eigval2 = c2_eigvals[1],
                pc1 = c2_pcs[:, 0], varprop = c2_varprops[0]
            )

        return self

    # ------------ #
    #  Info table  #
    # ------------ #

    @property
    def info(self):
        rows = [
            [repr(i), repr(len(ci.clus)), ci.eigval1, ci.eigval2, ci.varprop]
            for i, ci in self.clusters.items()
        ]
        return pd.DataFrame(
            rows,
            columns = ['Cluster', 'N_Vars', 'Eigval1', 'Eigval2', 'VarProp']
        )

    # ----------------------------- #
    #  R-square table — vectorised  #
    # ----------------------------- #

    def _rsquarespu(self):

        """
        Vectorised RSQ Computation.

        For each cluster i with eigenvector v_i and sigma_i:
            RS_own(feat) = (v_i · corr[feat, clus_i] / sigma_i) ^ 2

        We compute all features x all clusters in matrix form.
        """

        global_corr = self.corrs.values #(p, p)
        feat_list = self.corrs.columns.tolist()
        feat_pos = {f: i for i, f in enumerate(feat_list)}
        n_feat = len(feat_list)
        n_clus = len(self.clusters)

        # Pre-compute (sigma_i, v_i padded to global indices) for each cluster
        clus_items = list(self.clusters.items())
        sigmas = np.empty(n_clus)

        # Store per-cluster: projected score for every feature  (n_feat,)
        proj = np.empty((n_clus, n_feat))   #proj[i, f] = v_i · corr[clus_i, f] / sigma_i

        for k, (_, ci) in enumerate(clus_items):
            c_idx = np.array([feat_pos[f] for f in ci.clus])
            v = ci.eigvecs[:, 0] #eigenvector of PC1
            sub = global_corr[np.ix_(c_idx, c_idx)]
            sigmas[k] = math.sqrt(v @ sub @ v)

            # corr between PC1 of cluster k and every feature
            # (v · corr[clus_k, :]) / sigma_k
            proj[k] = (v @ global_corr[c_idx, :]) / sigmas[k]  #(n_feat,)

        rs_matrix = proj ** 2   #(n_clus, n_feat) — RS of each feat w/ each cluster PC

        rows = []
        for k, (i, ci) in enumerate(clus_items):
            other_mask = np.ones(n_clus, dtype=bool)
            other_mask[k] = False
            for feat in ci.clus:
                fi = feat_pos[feat]
                rs_own = rs_matrix[k, fi]
                rs_nc = rs_matrix[other_mask, fi].max() if n_clus > 1 else 0.0
                rows.append([i, feat, rs_own, rs_nc, (1 - rs_own) / (1 - rs_nc)])

        return pd.DataFrame(
            rows,
            columns = ['Cluster', 'Variable', 'RS_Own', 'RS_NC', 'RS_Ratio']
        )

    @property
    def rsquare(self):
        if self.speedup:
            return self._rsquarespu()

        # ---- slow path (pca-based) ---- #
        pcs = [ci.pc1 for _, ci in self.clusters.items()]
        clus_items = list(self.clusters.items())

        # Pre-compute RS for all features vs all PCs
        all_feat_vals = self.df.values #(n_obs, p)
        all_feat_list = self.df.columns.tolist()
        pc_mat = np.column_stack(pcs) #(n_obs, n_clus)

        def _rs_matrix(feat_vals, pc_mat):

            # Pearson rsq of every column of feat_vals with every column of pc_mat
            f_c = feat_vals - feat_vals.mean(axis = 0)
            p_c = pc_mat - pc_mat.mean(axis = 0)
            cov  = f_c.T @ p_c #(p, n_clus)
            f_sd = np.sqrt((f_c ** 2).sum(axis = 0)) #(p,)
            p_sd = np.sqrt((p_c ** 2).sum(axis = 0)) #(n_clus,)
            r = cov / np.outer(f_sd, p_sd)
            return r ** 2 #(p, n_clus)

        rs_mat = _rs_matrix(all_feat_vals, pc_mat) #(p, n_clus)
        feat_pos = {f: i for i, f in enumerate(all_feat_list)}
        n_clus = len(pcs)

        rows = []
        for k, (i, ci) in enumerate(clus_items):
            other_mask = np.ones(n_clus, dtype = bool)
            other_mask[k] = False
            for feat in ci.clus:
                fi = feat_pos[feat]
                rs_own = rs_mat[fi, k]
                rs_nc = rs_mat[fi, other_mask].max() if n_clus > 1 else 0.0
                rows.append([i, feat, rs_own, rs_nc, (1 - rs_own) / (1 - rs_nc)])

        return pd.DataFrame(
            rows,
            columns = ['Cluster', 'Variable', 'RS_Own', 'RS_NC', 'RS_Ratio']
        )


# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    demo_df = pd.read_csv(
        'https://raw.githubusercontent.com/naenumtou/varclushi_opt/refs/heads/master/data/winequality-red.csv'
    )
    demo_df = demo_df.drop('quality', axis = 1)
    demo_vc = VarClusHi_Opt(demo_df)
    demo_vc.varclus()
    print(demo_vc.info)
    print(demo_vc.rsquare)
