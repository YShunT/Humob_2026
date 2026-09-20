"""EXP03: 状態空間モデルと残差補正を比較し、提出物を生成する。

実行: python3 src/EXP03_run.py
入力: data/raw/humob2026-dataset.tsv、data/processed/の外部特徴CSV
出力: experiments/EXP03/{metrics.json,.log,outputs/,figures/}
依存: common/*.py、numpy、scipy、pandas、matplotlib、catboost
"""

from __future__ import annotations

import ast
import datetime as dt
import json
import os
from pathlib import Path
import time
from contextlib import nullcontext
from dataclasses import dataclass

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/humob-mpl-cache')
import numpy as np
import pandas as pd
from scipy.linalg import eigh

from common.competition_metric import EVAL_X, EVAL_Y, in_eval_box, day_errors, score
from common.experiment_io import (
    check_od_tsv, write_metrics, write_od_tsv, write_records_csv,
)
from common.local_error_map import validation_days, save_local_error_map
from common.run_log import run_log

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments/EXP03'
OUT = EXP / 'outputs'
FIG = EXP / 'figures'
RAW = ROOT / 'data/raw/humob2026-dataset.tsv'
PROCESSED = ROOT / 'data/processed'
PATHS = {
    'cells': PROCESSED / 'events/population_cell_map.csv',
    'evac': PROCESSED / '避難者率/市町別避難者率.csv',
    'water': PROCESSED / '断水率/市町別断水率.csv',
    'shelter': PROCESSED / '避難所/避難所_セル別集計.csv',
    'weather': PROCESSED / 'weather/weather.csv',
}
DATES = pd.date_range('2023-11-01', '2024-10-31')
DAYS = np.array([int(x.strftime('%Y%m%d')) for x in DATES])
CV = set(validation_days())
TEST = [int(x.strftime('%Y%m%d')) for x in pd.date_range('2024-02-01', '2024-03-31')
        if x.strftime('%Y%m%d') not in ('20240202', '20240305')]
QUAKE = int(np.flatnonzero(DAYS == 20240101)[0])
GROUPS = ('evac', 'water', 'rain', 'snow', 'shelter')
SETTINGS = dict(em_max_iter=200, em_tol=1e-5, variance_floor=1e-8,
                weekday_ridge=1.0, deviation_ridge=20.0, global_ridge=1.0,
                initial_variance=1e4, crossfit_folds=5, crossfit_block_days=28,
                inverse_log='median_expm1', seed=2026)
CAT_PARAMS = dict(iterations=300, depth=5, learning_rate=0.04, l2_leaf_reg=10,
                  loss_function='RMSE', random_seed=2026, thread_count=4,
                  verbose=False, allow_writing_files=False, task_type='CPU')
STATIONS = {  # 気象庁の地域気象観測所一覧。既存weather_station_coverage_map.pyと同値。
    '珠洲': (37 + 26.8 / 60, 137 + 17.2 / 60),
    '輪島': (37 + 23.4 / 60, 136 + 53.7 / 60),
    '門前': (37 + 15.7 / 60, 136 + 43.7 / 60),
    '三井': (37 + 17.6 / 60, 136 + 57.7 / 60),
    '志賀': (37 + 8.6 / 60, 136 + 43.5 / 60),
    '七尾': (37 + 1.8 / 60, 136 + 59.5 / 60),
}


def csv_write(name, rows):
    return write_records_csv(OUT / name, rows)


def distance(lat1, lon1, lat2, lon2):
    a, b = np.radians(lat1), np.radians(lat2)
    dlat, dlon = b - a, np.radians(lon2) - np.radians(lon1)
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(
        np.sin(dlat / 2) ** 2 + np.cos(a) * np.cos(b) * np.sin(dlon / 2) ** 2, 0, 1)))


def symmetric_inverse(matrix):
    # macOS Accelerate/NumPy 2の小行列matmulに出る浮動小数点警告を避ける。
    values, vectors = eigh((matrix+matrix.T)/2)
    floor = max(float(np.max(np.abs(values)))*1e-12, 1e-12)
    if values.min() < -floor*10:
        raise np.linalg.LinAlgError('Indefinite regression normal matrix')
    reciprocals = np.divide(1., values, out=np.zeros_like(values), where=values > floor)
    return np.einsum('ik,k,jk->ij', vectors, reciprocals, vectors)


class Dataset:
    """生データは一度だけ読み、外側CV正解と学習用ODを分離する。"""

    def __init__(self, validation=True):
        self.validation = validation
        self.cells = np.array([f'{y}_{x}' for y in range(EVAL_Y[0], EVAL_Y[1] + 1)
                               for x in range(EVAL_X[0], EVAL_X[1] + 1)])
        self.index = {c: i for i, c in enumerate(self.cells)}
        self.observed = np.zeros(len(DATES), bool)
        self.truth = {}
        self.od = {}
        self.outside = [{} for _ in range(7)]
        self.outside_n = np.zeros(7)
        self.mass = []
        for line in RAW.open():
            ds, payload = line.rstrip('\n').split('\t', 1)
            day = int(ds)
            if payload.strip() == 'NA':
                continue
            full = ast.literal_eval(payload)
            small = {o: {d: v for d, v in dests.items() if in_eval_box(d)}
                     for o, dests in full.items() if in_eval_box(o)}
            if validation and day in CV:
                self.truth[day] = small
                continue
            t = int(np.flatnonzero(DAYS == day)[0])
            self.observed[t] = True
            self.od[t] = small
            self.mass.append(sum(sum(z.values()) for z in full.values()))
            if day >= 20240101:
                w = DATES[t].dayofweek
                self.outside_n[w] += 1
                for o, dests in full.items():
                    for d, v in dests.items():
                        if not (in_eval_box(o) and in_eval_box(d)):
                            key = (o, d)
                            self.outside[w][key] = self.outside[w].get(key, 0.) + v
        self.pairs = sorted({(o, d) for od in self.od.values() for o, ds in od.items()
                             for d, v in ds.items() if o != d and v > 0})
        self.pi = {pair: i for i, pair in enumerate(self.pairs)}
        self.origin = np.array([self.index[o] for o, _ in self.pairs], dtype=int)
        self.dest = np.array([self.index[d] for _, d in self.pairs], dtype=int)
        self.diag = np.zeros((len(DATES), len(self.cells)))
        self.flow = np.zeros((len(DATES), len(self.pairs)))
        for t, od in self.od.items():
            for o, ds in od.items():
                for d, v in ds.items():
                    if o == d:
                        self.diag[t, self.index[o]] = v
                    elif (o, d) in self.pi:
                        self.flow[t, self.pi[o, d]] = v
        self.total = np.zeros_like(self.diag)
        for p, o in enumerate(self.origin):
            self.total[:, o] += self.flow[:, p]
        self.ratio = np.divide(self.flow, self.total[:, self.origin],
                               out=np.zeros_like(self.flow), where=self.total[:, self.origin] > 0)
        self.audit = []
        self.external = {}
        self.load_features()

    def load_features(self):
        cells = pd.read_csv(PATHS['cells']).set_index('grid_id').loc[self.cells]
        self.area = cells.area.fillna('対象外').to_numpy(str)
        self.lat = cells.latitude.to_numpy(float)
        self.lon = cells.longitude.to_numpy(float)
        shelter = pd.read_csv(PATHS['shelter']).set_index('grid_id').loc[self.cells]
        self.facility = shelter.facility_count.to_numpy(float)
        assert np.isfinite(self.facility).all() and (self.facility >= 0).all()
        for group, value_col in [('evac', '避難者率'), ('water', '断水世帯率')]:
            table = pd.read_csv(PATHS[group])
            table['date'] = pd.to_datetime(table['日付'])
            if table.duplicated(['date', '市町名']).any():
                raise ValueError(f'{group}: duplicate reports')
            values = np.full_like(self.diag, np.nan)
            for area in np.unique(self.area):
                report = table.loc[table['市町名'] == area].sort_values('date')
                if report.empty:
                    continue
                ix = self.area == area
                rd = report.date.to_numpy('datetime64[ns]')
                pos = np.searchsorted(rd, DATES.to_numpy(), side='right') - 1
                rv = pd.to_numeric(report[value_col], errors='coerce').to_numpy()
                v = np.where(pos >= 0, rv[np.maximum(pos, 0)], np.nan)
                v[:QUAKE] = 0
                values[:, ix] = v[:, None]
                for t in range(len(DATES)):
                    j = pos[t]
                    self.audit.append(dict(kind=group, date=int(DAYS[t]), area=area,
                        value=None if not np.isfinite(v[t]) else float(v[t]),
                        report_date=None if j < 0 else str(report.iloc[j]['date'].date()),
                        report_age=None if j < 0 else int((DATES[t] - report.iloc[j]['date']).days),
                        carried_past_last=bool(j >= 0 and DATES[t] > report.date.iloc[-1])))
            if np.nanmin(values) < 0:
                raise ValueError(f'{group}: negative rate')
            if np.nanmax(values) > 1:
                print(f'[audit] {group} rate maximum={np.nanmax(values):.6f}; not clipped', flush=True)
            self.external[group] = values
        weather = pd.read_csv(PATHS['weather'])
        weather['date'] = pd.to_datetime(weather.date)
        names = list(STATIONS)
        dist = np.stack([distance(self.lat, self.lon, *STATIONS[n]) for n in names], axis=1)
        if weather.duplicated(['date', 'station']).any():
            raise ValueError('Duplicate station/day')
        for group, col in [('rain', 'precipitation_total_mm'), ('snow', 'snow_depth_max_cm')]:
            vals = weather.pivot(index='date', columns='station', values=col).reindex(index=DATES, columns=names)
            quality = weather.pivot(index='date', columns='station', values=col + '_quality').reindex(index=DATES, columns=names)
            result = np.full_like(self.diag, np.nan)
            for t, row in enumerate(vals.to_numpy(float)):
                usable = np.isfinite(row) & (row >= 0)
                if not usable.any():
                    continue
                nearest = np.argmin(np.where(usable[None, :], dist, np.inf), axis=1)
                result[t] = np.log1p(row[nearest])
                for n, station in enumerate(names):
                    assigned = nearest == n
                    if assigned.any():
                        self.audit.append(dict(kind=group, date=int(DAYS[t]), station=station,
                            assigned_cells=int(assigned.sum()), max_distance_km=float(dist[assigned, n].max()),
                            quality=None if pd.isna(quality.iloc[t, n]) else float(quality.iloc[t, n])))
            self.external[group] = result

    def support(self, mask):
        return self.flow[mask].sum(axis=0) > 0

    def fallback(self, mask):
        sums = self.flow[mask].sum(axis=0)
        den = self.total[mask].sum(axis=0)[self.origin]
        return np.divide(sums, den, out=np.zeros_like(sums), where=den > 0)

    def features(self, kind, groups):
        ix = self.origin if kind == 'r' else np.arange(len(self.cells))
        arrays, names = [], []
        for group in groups:
            if group == 'shelter':
                if kind == 'r':
                    f = np.log1p(self.facility[self.dest])
                    arrays.extend([np.broadcast_to(f, self.ratio.shape),
                                   np.broadcast_to(self.facility[self.dest] > 0, self.ratio.shape),
                                   self.external['evac'][:, self.dest] * f])
                    names.extend(['dest_facility_log', 'dest_has_shelter', 'dest_shelter_evac'])
                else:
                    arrays.append(self.external['evac'] * np.log1p(self.facility))
                    names.append('shelter_evac')
            else:
                arrays.append(self.external[group][:, ix])
                names.append('origin_' + group)
                if kind == 'r':
                    arrays.append(self.external[group][:, self.dest])
                    names.append('dest_' + group)
        shape = self.ratio.shape if kind == 'r' else self.diag.shape
        return np.stack(arrays, axis=-1) if arrays else np.empty((*shape, 0)), names

    def reconstruct(self, days, diag, total, ratio, outside=False):
        result = {}
        for row, t in enumerate(days):
            od = {}
            if outside:
                w = DATES[t].dayofweek
                if not self.outside_n[w]:
                    raise ValueError('No outside weekday template')
                for (o, d), v in self.outside[w].items():
                    od.setdefault(o, {})[d] = float(v / self.outside_n[w])
            for i, o in enumerate(self.cells):
                if diag[row, i] > 0:
                    od.setdefault(str(o), {})[str(o)] = float(diag[row, i])
            for p, (o, d) in enumerate(self.pairs):
                v = total[row, self.origin[p]] * ratio[row, p]
                if v > 0:
                    od.setdefault(o, {})[d] = float(v)
            result[int(DAYS[t])] = od
        return result


def prepare_features(raw, mask, names):
    """学習観測行だけで欠損処理と標準化を決める。"""
    values, kept, audit = [], [], []
    for j, name in enumerate(names):
        x = raw[:, :, j]
        finite = np.isfinite(x)
        training = x[mask & finite]
        if training.size == 0:
            audit.append(dict(feature=name, dropped='all_missing'))
            continue
        mean = float(training.mean())
        for label, column in [(name, np.where(finite, x, mean)), (name + '_missing', (~finite).astype(float))]:
            mu = float(column[mask].mean())
            sd = float(column[mask].std())
            audit.append(dict(feature=label, mean=mu, sd=sd, dropped='constant' if sd < 1e-10 else ''))
            if sd >= 1e-10:
                values.append((column - mu) / sd)
                kept.append(label)
    return (np.stack(values, axis=-1) if values else np.empty((*mask.shape, 0))), kept, audit


def calendar(fixed):
    dow = np.asarray(DATES.dayofweek)
    cols = [(dow == w).astype(float) for w in range(1, 7)]
    names = [f'weekday_{w}' for w in range(1, 7)]
    cols.append(np.array([(d.month == 12 and d.day >= 29) or (d.month == 1 and d.day <= 3) for d in DATES], float))
    names.append('year_end')
    if fixed:
        cols.extend([(np.arange(len(DATES)) < QUAKE).astype(float),
                     (np.arange(len(DATES)) >= QUAKE).astype(float)])
        names.extend(['pre_level', 'post_level'])
    return np.stack(cols, axis=-1), names


class Regression:
    """セル別カレンダー・対角外部偏差と共通外部係数の罰則付き同時推定。"""

    def __init__(self, x, mask, fixed, deviations):
        c, self.cn = calendar(fixed)
        self.x = x
        self.mask = mask
        self.z = np.broadcast_to(c[:, None, :], (*mask.shape, c.shape[1]))
        if deviations and x.shape[-1]:
            self.z = np.concatenate([self.z, x], axis=-1)
        z, k = self.z, self.z.shape[-1]
        penalty = np.full(k, SETTINGS['weekday_ridge'])
        if fixed:
            penalty[len(self.cn)-2:len(self.cn)] = 1e-6
        penalty[len(self.cn):] = SETTINGS['deviation_ridge']
        self.gram = np.einsum('tnk,tnl,tn->nkl', z, z, mask, optimize=True)
        self.zx = np.einsum('tnk,tnl,tn->nkl', z, x, mask, optimize=True)
        self.xx = np.einsum('tnk,tnl,tn->kl', x, x, mask, optimize=True)
        self.penalty = penalty

    def fit(self, target, noise_variance=1.):
        # 観測負対数尤度 SSE/(2R) と同じ尺度でL2を適用する。
        inverse = np.linalg.inv(self.gram + noise_variance*np.diag(self.penalty)[None])
        izx = np.einsum('nkl,nlj->nkj', inverse, self.zx)
        rhs = np.einsum('tnk,tn->nk', self.z, np.where(self.mask, target, 0), optimize=True)
        irhs = np.einsum('nkl,nl->nk', inverse, rhs)
        if self.x.shape[-1]:
            schur = self.xx - np.einsum('nki,nkj->ij', self.zx, izx)
            global_inv = symmetric_inverse(schur + np.eye(self.x.shape[-1])*noise_variance*SETTINGS['global_ridge'])
            xy = np.einsum('tnk,tn->k', self.x, np.where(self.mask, target, 0), optimize=True)
            grhs = xy - np.einsum('nki,nk->i', self.zx, irhs)
            self.beta = np.einsum('ij,j->i', global_inv, grhs)
            self.local = irhs - np.einsum('nki,i->nk', izx, self.beta)
        else:
            self.beta = np.empty(0)
            self.local = irhs
        return (np.einsum('tnk,nk->tn', self.z, self.local, optimize=True)
                + np.einsum('tnk,k->tn', self.x, self.beta, optimize=True))

    def loss(self):
        return float(np.sum(self.local**2 * self.penalty) + SETTINGS['global_ridge'] * np.sum(self.beta**2))


def smooth(y, mask, q, r, reset=QUAKE):
    """独立なlocal-level系列のfilter + RTS smoother。欠損では観測更新しない。"""
    nt, ns = y.shape
    mf, pf, pp = np.zeros_like(y), np.zeros_like(y), np.zeros_like(y)
    m, p = np.zeros(ns), np.full(ns, SETTINGS['initial_variance'])
    nll = 0.
    for t in range(nt):
        if t == reset:
            m = np.zeros(ns)
            p = np.full(ns, SETTINGS['initial_variance'])
        if t != 0 and t != reset:
            p = p + q
        pp[t] = p
        v = np.where(mask[t], y[t] - m, 0.)
        f = p + r
        nll += .5 * np.sum(np.where(mask[t], np.log(2 * np.pi * f) + v * v / f, 0.))
        gain = np.where(mask[t], p / f, 0.)
        m = m + gain * v
        p = (1 - gain) * p
        mf[t], pf[t] = m, p
    ms, ps = mf.copy(), pf.copy()
    lag = np.zeros_like(y)
    for t in range(nt - 2, -1, -1):
        if t + 1 == reset:
            continue
        j = pf[t] / pp[t + 1]
        ms[t] += j * (ms[t + 1] - mf[t])
        ps[t] = np.maximum(0, pf[t] + j*j * (ps[t + 1] - pp[t + 1]))
        lag[t + 1] = j * ps[t + 1]
    return ms, ps, lag, float(nll)


@dataclass
class Prediction:
    mean: np.ndarray
    variance: np.ndarray
    mask: np.ndarray
    diagnostics: dict
    coefficients: list
    feature_audit: list


def fit_component(data, kind, groups, fixed, train):
    tic = time.monotonic()
    raw = {'D': data.diag, 'O': data.total, 'r': data.ratio}[kind]
    mask = np.broadcast_to(train[:, None], raw.shape).copy()
    if kind == 'r':
        mask &= data.total[:, data.origin] > 0
        active = data.support(train)
    else:
        active = (raw[train] > 0).any(axis=0)
    mask &= active[None]
    if not mask.any():
        return Prediction(np.zeros_like(raw), np.zeros_like(raw), mask,
                          dict(kind=kind, active=0, seconds=time.monotonic()-tic), [], [])
    # 未観測値は目的変数として参照しない。
    y = np.where(mask, np.log1p(raw) if kind != 'r' else raw, 0.)
    ext, names = data.features(kind, groups)
    x, names, audit = prepare_features(ext, mask, names)
    regression = Regression(x, mask, fixed, deviations=(kind == 'D'))
    initial_level = np.zeros_like(y)
    if not fixed:
        for lo, hi in [(0, QUAKE), (QUAKE, len(DATES))]:
            count = mask[lo:hi].sum(axis=0)
            level = np.divide(y[lo:hi].sum(axis=0), count,
                              out=np.zeros(raw.shape[1]), where=count > 0)
            initial_level[lo:hi] = level
    offset = regression.fit(y-initial_level)
    floor = SETTINGS['variance_floor']
    r = max(float(np.mean((y - initial_level - offset)[mask]**2)), floor)
    q = max(.03 * r, floor)
    converged = fixed
    iterations, last = 0, None
    if fixed:
        for iterations in range(1, SETTINGS['em_max_iter']+1):
            new_offset = regression.fit(y, r)
            rnew = max(float(np.mean((y-new_offset)[mask]**2)), floor)
            delta = float(np.max(np.abs(new_offset-offset)))
            offset, r = new_offset, rnew
            if delta < SETTINGS['em_tol']:
                break
        converged = delta < SETTINGS['em_tol']
        mean = offset
        variance = np.zeros_like(raw)
        objective = float(np.sum((y-offset)[mask]**2)/r + mask.sum()*np.log(r) + regression.loss())
    else:
        for iterations in range(1, SETTINGS['em_max_iter'] + 1):
            state, variance, lag, nll = smooth(y-offset, mask, q, r)
            objective = nll + .5*regression.loss()
            offset_new = regression.fit(y-state, r)
            rnew = max(float(np.mean(((y-state-offset_new)**2 + variance)[mask])), floor)
            delta = np.diff(state, axis=0)**2 + variance[1:] + variance[:-1] - 2*lag[1:]
            edges = np.ones(len(DATES)-1, bool)
            edges[QUAKE-1] = False
            qnew = max(float(delta[edges][:, active].mean()), floor)
            offset, q, r = offset_new, qnew, rnew
            if last is not None and abs(objective-last) <= SETTINGS['em_tol']*(1+abs(last)):
                converged = True
                break
            last = objective
        state, variance, lag, nll = smooth(y-offset, mask, q, r)
        mean = state + offset
        objective = nll + .5*regression.loss()
    coefficients = [dict(kind=kind, series='shared', feature=n, value=float(b))
                    for n, b in zip(names, regression.beta)]
    local_names = regression.cn + (names if kind == 'D' else [])
    series_ids = data.cells if kind != 'r' else [f'{o}>{d}' for o, d in data.pairs]
    for j in np.flatnonzero(active):
        coefficients.extend(dict(kind=kind, series=str(series_ids[j]), feature=n, value=float(b))
                            for n, b in zip(local_names, regression.local[j]))
    if kind != 'r':
        if np.max(mean[:, active]) > 50:
            raise FloatingPointError('Inverse-log mean exploded')
        mean = np.expm1(mean)
    mean[:, ~active] = 0
    variance[:, ~active] = 0
    diag = dict(kind=kind, fixed=fixed, groups=','.join(groups), active=int(active.sum()),
                observations=int(mask.sum()), q=None if fixed else q, r=r,
                iterations=iterations, converged=converged, objective=objective,
                negative_fraction=float(np.mean(mean < 0)),
                max_state_variance=float(variance[:, active].max()),
                seconds=time.monotonic()-tic)
    return Prediction(np.maximum(mean, 0) if kind != 'r' else mean, variance, mask, diag, coefficients, audit)


def normalize(data, ratio, train):
    support = data.support(train)
    pos = np.maximum(ratio, 0) * support[None]
    sums = np.zeros((len(ratio), len(data.cells)))
    for j, o in enumerate(data.origin):
        sums[:, o] += pos[:, j]
    fallback = data.fallback(train)
    result = np.divide(pos, sums[:, data.origin], out=np.zeros_like(pos), where=sums[:, data.origin] > 0)
    result = np.where(sums[:, data.origin] > 0, result, fallback[None])
    check = np.zeros_like(sums)
    for j, o in enumerate(data.origin):
        check[:, o] += result[:, j]
    active_origins = np.bincount(data.origin, weights=support, minlength=len(data.cells)) > 0
    assert np.allclose(check[:, active_origins], 1, atol=1e-10)
    assert np.all(result >= 0) and np.isfinite(result).all()
    audit = dict(negative_fraction=float(np.mean(ratio < 0)),
                 negative_mass=float(-np.minimum(ratio, 0).sum()),
                 fallback_origin_days=int((sums[:, active_origins] <= 0).sum()),
                 max_pre_norm_sum=float(sums.max()),
                 mean_absolute_adjustment=float(np.mean(np.abs(result-ratio))))
    return result, audit


def scores(pred, truth):
    daily = {d: day_errors(pred[d], truth[d]) for d in sorted(truth)}
    result = dict(overall=score(list(daily.values())),
                  jan=score([v for d, v in daily.items() if str(d).startswith('202401')]),
                  apr=score([v for d, v in daily.items() if str(d).startswith('202404')]))
    return result, daily


def metric_summary(result, n_days):
    """内部の評価結果を共通の metrics.json 形式へ変換する。"""
    return dict(n_days=n_days, combined_nrmse=result['combined'],
                nrmse_diag=result['nrmse_diag'], nrmse_offdiag=result['nrmse_off'])


def better(candidates, field):
    best = None
    for c in candidates:
        if best is None or c['score'][field] < best['score'][field]-1e-12:
            best = c
        elif abs(c['score'][field]-best['score'][field]) <= 1e-12:
            if (len(c['groups']), not c['fixed']) < (len(best['groups']), not best['fixed']):
                best = c
    return best


def ablations(data, log):
    times = np.flatnonzero(np.isin(DAYS, list(CV)))
    ds, ns, fits = [], [], []
    cache = {}
    def fit(kind, groups, fixed, name):
        key = (kind, tuple(groups), fixed)
        if key not in cache:
            print(f'[fit] {name} {kind} groups={groups}', flush=True)
            p = fit_component(data, kind, groups, fixed, data.observed)
            fits.append(dict(candidate=name, **p.diagnostics))
            cache[key] = p
            print(f'[fit done] {name} {kind}: {p.diagnostics}', flush=True)
        return cache[key]
    for number in ['-ref', '0', '1', '2', '3', '4', '5', '6', '7', '8']:
        default = {'-ref': (), '0': (), '1': ('evac',), '2': ('water',), '3': ('rain',),
                   '4': ('snow',), '6': GROUPS[:4], '7': ('evac', 'shelter'), '8': GROUPS}
        fixed = number == '-ref'
        if number == '5':
            dg = tuple(GROUPS[i] for i in range(4) if ds[i+2]['score']['nrmse_diag'] < ds[1]['score']['nrmse_diag'])
            ng = tuple(GROUPS[i] for i in range(4) if ns[i+2]['score']['nrmse_off'] < ns[1]['score']['nrmse_off'])
        else:
            dg = ng = default[number]
        dp = fit('D', dg, fixed, 'D'+number)
        op = fit('O', ng, fixed, 'N'+number)
        rp = fit('r', ng, fixed, 'N'+number)
        rr, _ = normalize(data, rp.mean, data.observed)
        pred = data.reconstruct(times, dp.mean[times], op.mean[times], rr[times])
        sc, _ = scores(pred, data.truth)
        ds.append(dict(id='D'+number, groups=dg, fixed=fixed, prediction=dp, score=sc['overall']))
        ns.append(dict(id='N'+number, groups=ng, fixed=fixed, total=op, share=rp,
                       normalized=rr, score=sc['overall']))
        print(f'[CV] D{number}={sc["overall"]["nrmse_diag"]:.6f} N{number}={sc["overall"]["nrmse_off"]:.6f}', flush=True)
    csv_write('fit_diagnostics.csv', fits)
    return ds, ns


def gap_features(mask):
    nt, ns = mask.shape
    prev, nxt = np.full((nt, ns), np.nan), np.full((nt, ns), np.nan)
    for lo, hi in [(0, QUAKE), (QUAKE, nt)]:
        last = np.full(ns, np.nan)
        for t in range(lo, hi):
            prev[t] = t-last
            last = np.where(mask[t], t, last)
        last = np.full(ns, np.nan)
        for t in range(hi-1, lo-1, -1):
            nxt[t] = last-t
            last = np.where(mask[t], t, last)
    return prev, nxt, prev/(prev+nxt)


def cat_frame(data, kind, base, dates):
    ns = base.mean.shape[1]
    ti = np.repeat(np.asarray(dates), ns)
    si = np.tile(np.arange(ns), len(dates))
    oi = data.origin[si] if kind == 'r' else si
    di = data.dest[si] if kind == 'r' else si
    previous, following, position = gap_features(base.mask)
    frame = pd.DataFrame(dict(
        baseline=base.mean[ti, si], state_variance=base.variance[ti, si],
        previous_gap=previous[ti, si], next_gap=following[ti, si], gap_position=position[ti, si],
        weekday=np.asarray(DATES.dayofweek)[ti], month=np.asarray(DATES.month)[ti],
        quake_days=ti-QUAKE, year_end=np.array([(d.month == 12 and d.day >= 29) or
                    (d.month == 1 and d.day <= 3) for d in DATES])[ti].astype(int),
        origin=data.cells[oi], origin_area=data.area[oi],
        origin_facility=np.log1p(data.facility[oi]), origin_has_shelter=(data.facility[oi] > 0).astype(int)))
    for group in GROUPS[:4]:
        frame['origin_'+group] = data.external[group][ti, oi]
    frame['origin_shelter_evac'] = frame.origin_facility * frame.origin_evac
    cats = ['origin', 'origin_area']
    if kind == 'r':
        frame['dest'], frame['dest_area'] = data.cells[di], data.area[di]
        frame['same_area'] = (data.area[oi] == data.area[di]).astype(int)
        frame['distance_km'] = distance(data.lat[oi], data.lon[oi], data.lat[di], data.lon[di])
        frame['dest_facility'] = np.log1p(data.facility[di])
        frame['dest_has_shelter'] = (data.facility[di] > 0).astype(int)
        for group in GROUPS[:4]:
            frame['dest_'+group] = data.external[group][ti, di]
        frame['dest_shelter_evac'] = frame.dest_facility * frame.dest_evac
        cats += ['dest', 'dest_area']
    return frame, cats


def residual_models(data, best_d, best_n, target_times):
    from catboost import CatBoostRegressor
    bases = {'D': best_d['prediction'], 'O': best_n['total'], 'r': best_n['share']}
    # 配分の基準値は本番と同じ非負化・正規化後の値。
    bases['r'] = Prediction(best_n['normalized'], bases['r'].variance, bases['r'].mask,
                             bases['r'].diagnostics, [], [])
    specs = {'D': best_d, 'O': best_n, 'r': best_n}
    xs, ys, ws = {k: [] for k in bases}, {k: [] for k in bases}, {k: [] for k in bases}
    fold_ids = (np.arange(len(DATES)) // SETTINGS['crossfit_block_days']) % SETTINGS['crossfit_folds']
    for fold in range(SETTINGS['crossfit_folds']):
        held = data.observed & (fold_ids == fold)
        train = data.observed & ~held
        dates = np.flatnonzero(held)
        assert not (held & train).any()
        if data.validation:
            assert not np.isin(DAYS[held | train], list(CV)).any()
        for kind in bases:
            spec = specs[kind]
            print(f'[crossfit] fold={fold} {kind}, train={train.sum()} hidden={held.sum()}', flush=True)
            base = fit_component(data, kind, spec['groups'], spec['fixed'], train)
            if kind == 'r':
                base.mean, _ = normalize(data, base.mean, train)
            frame, cats = cat_frame(data, kind, base, dates)
            raw = {'D': data.diag, 'O': data.total, 'r': data.ratio}[kind]
            residual = (raw[dates]-base.mean[dates]).ravel()
            weight = np.ones_like(residual)
            if kind == 'r':
                weight = data.total[dates][:, data.origin].ravel()**2
            keep = weight > 0
            xs[kind].append(frame.loc[keep])
            ys[kind].append(residual[keep])
            ws[kind].append(weight[keep])
    corrected = {}
    for kind, base in bases.items():
        x = pd.concat(xs[kind], ignore_index=True)
        y, weight = np.concatenate(ys[kind]), np.concatenate(ws[kind])
        frame, cats = cat_frame(data, kind, base, target_times)
        model = CatBoostRegressor(**CAT_PARAMS)
        print(f'[CatBoost] {kind} rows={len(x)}', flush=True)
        model.fit(x, y, cat_features=cats, sample_weight=weight/weight.mean())
        residual = model.predict(frame).reshape(len(target_times), -1)
        corrected[kind] = base.mean[target_times] + residual
        # 全学習日で0の系列は状態空間モデルと同じく予測0。
        if kind != 'r':
            active = ({'D': data.diag, 'O': data.total}[kind][data.observed] > 0).any(axis=0)
            corrected[kind] = np.maximum(corrected[kind], 0) * active[None]
        del x, y, weight
        xs[kind].clear()
    corrected['r'], _ = normalize(data, corrected['r'], data.observed)
    return corrected


def candidate_rows(candidates, field):
    return [dict(candidate=c['id'], groups=','.join(c['groups']), fixed=c['fixed'],
                 score=c['score'][field]) for c in candidates]


def plots(ds, ns, predictions, data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for label, candidates, field in [('diag', ds, 'nrmse_diag'), ('off', ns, 'nrmse_off')]:
        ids = [c['id'] for c in candidates]
        values = np.array([c['score'][field] for c in candidates])
        best = int(np.argmin(values))
        span = max(float(values.max() - values.min()), 0.002)
        margin = span * 0.14
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(ids, values, color='#8aa1b4', linewidth=1.2, zorder=1)
        colors = ['#d95f02' if i == best else '#2878b5' for i in range(len(ids))]
        ax.scatter(ids, values, color=colors, s=62, zorder=2)
        for i, value in enumerate(values):
            ax.annotate(f'{value:.4f}', (i, value), xytext=(0, 8),
                        textcoords='offset points', ha='center', fontsize=8,
                        fontweight='bold' if i == best else 'normal')
        ax.set_ylim(float(values.min() - margin), float(values.max() + margin * 1.8))
        ax.set_ylabel('NRMSE_diag' if label == 'diag' else 'NRMSE_off')
        component = '対角' if label == 'diag' else '非対角'
        ax.set_title(f'EXP03 {component}モデルのアブレーション（縦軸拡大・低いほど良い）')
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / f'{label}_ablation_scores.png', dpi=150)
        plt.close(fig)
    from common.pred_dist import save_pred_dist
    save_pred_dist('EXP03', predictions=predictions,
                   out=FIG / 'EXP03_pred_dist.png')


def validate(log=None):
    context = nullcontext(log) if log else run_log(
        'EXP03', 'run:validate', 'python3 src/EXP03_run.py'
    )
    with context as log:
        log.stage('validate-load', '配布データ・外部情報読み込みとCV分離')
        data = Dataset(True)
        OUT.mkdir(parents=True, exist_ok=True)
        FIG.mkdir(parents=True, exist_ok=True)
        log.stage('validate-ablation', '対角・非対角の状態空間アブレーション')
        ds, ns = ablations(data, log)
        best_d, best_n = better(ds, 'nrmse_diag'), better(ns, 'nrmse_off')
        times = np.flatnonzero(np.isin(DAYS, list(CV)))
        log.stage('validate-residual', '内部欠損ブロックで残差教師生成・CatBoost学習')
        hybrid = residual_models(data, best_d, best_n, times)
        hp = data.reconstruct(times, hybrid['D'], hybrid['O'], hybrid['r'])
        hs, _ = scores(hp, data.truth)
        hd = hs['overall']['nrmse_diag'] < best_d['score']['nrmse_diag']-1e-12
        hn = hs['overall']['nrmse_off'] < best_n['score']['nrmse_off']-1e-12
        diag = hybrid['D'] if hd else best_d['prediction'].mean[times]
        total = hybrid['O'] if hn else best_n['total'].mean[times]
        ratio = hybrid['r'] if hn else best_n['normalized'][times]
        selected = dict(diag=dict(base=best_d['id'], groups=list(best_d['groups']), fixed=best_d['fixed'], hybrid=hd),
                        off=dict(base=best_n['id'], groups=list(best_n['groups']), fixed=best_n['fixed'], hybrid=hn))
        log.stage('validate-score', '最終OD復元・公式評価')
        predictions = data.reconstruct(times, diag, total, ratio, outside=True)
        sc, _ = scores(predictions, data.truth)
        ds.append(dict(id='D-H', groups=GROUPS, fixed=False, score=hs['overall']))
        ns.append(dict(id='N-H', groups=GROUPS, fixed=False, score=hs['overall']))
        dr, nr = candidate_rows(ds, 'nrmse_diag'), candidate_rows(ns, 'nrmse_off')
        overall = metric_summary(sc['overall'], len(CV))
        validation = {
            'jan': metric_summary(sc['jan'], sum(str(day).startswith('202401') for day in CV)),
            'apr': metric_summary(sc['apr'], sum(str(day).startswith('202404') for day in CV)),
        }
        metrics = dict(experiment='EXP03', date=dt.datetime.now().astimezone().isoformat(timespec='minutes'),
            compared_to='EXP02', status='評価済み',
            config=dict(settings=SETTINGS, catboost=CAT_PARAMS, selected=selected,
             observed_days=int(data.observed.sum()), candidate_pairs=len(data.pairs)),
            results=dict(overall=overall, validation=validation,
                         diag_candidates=dr, off_candidates=nr))
        write_metrics(EXP / 'metrics.json', metrics)
        log.stage('validate-artifacts', '誤差マップ・市町誤差・比較図')
        from common.municipality_error import write_municipality_errors
        write_municipality_errors(predictions, data.truth, 'EXP03')
        save_local_error_map(predictions, data.truth, FIG / 'local_error_map.png', experiment='EXP03')
        plots(ds, ns, predictions, data)
        log.result(combined=overall['combined_nrmse'], diag=overall['nrmse_diag'], off=overall['nrmse_offdiag'])
        log.note('→ experiments/EXP03/metrics.json')
        log.note('→ experiments/EXP03/outputs/fit_diagnostics.csv')
        log.note('→ experiments/EXP03/figures/')
        print(json.dumps(overall), flush=True)
    return predictions


def submit(validation_predictions, log=None):
    context = nullcontext(log) if log else run_log(
        'EXP03', 'run:submit', 'python3 src/EXP03_run.py'
    )
    with context as log:
        metadata = json.loads((EXP / 'metrics.json').read_text())
        cfg = metadata['config']
        if cfg['settings'] != SETTINGS or cfg['catboost'] != CAT_PARAMS:
            raise ValueError('Configuration changed during the run; rerun EXP03')
        log.stage('submit-load', '全観測日読み込み')
        data = Dataset(False)
        times = np.flatnonzero(np.isin(DAYS, TEST))
        sd, sn = cfg['selected']['diag'], cfg['selected']['off']
        log.stage('submit-fit', '採用状態空間モデルを全観測日で再学習')
        dp = fit_component(data, 'D', sd['groups'], sd['fixed'], data.observed)
        op = fit_component(data, 'O', sn['groups'], sn['fixed'], data.observed)
        rp = fit_component(data, 'r', sn['groups'], sn['fixed'], data.observed)
        rr, _ = normalize(data, rp.mean, data.observed)
        d, o, r = dp.mean[times], op.mean[times], rr[times]
        if sd['hybrid'] or sn['hybrid']:
            log.stage('submit-residual', '採用残差モデルを全観測日で再学習')
            hybrid = residual_models(data, dict(prediction=dp, **sd),
                dict(total=op, share=rp, normalized=rr, **sn), times)
            if sd['hybrid']:
                d = hybrid['D']
            if sn['hybrid']:
                o, r = hybrid['O'], hybrid['r']
        log.stage('submit-write', '提出TSV・形式検査・日次推移')
        predictions = data.reconstruct(times, d, o, r, outside=True)
        write_od_tsv(predictions, OUT / 'submission.tsv')
        check_od_tsv(OUT / 'submission.tsv', TEST)
        from common.pred_dist import save_pred_dist
        save_pred_dist('EXP03', validation_predictions=validation_predictions)
        log.result(days=len(predictions))
        log.note('→ experiments/EXP03/outputs/submission.tsv')
    return predictions


def self_test():
    from types import SimpleNamespace
    # 欠損日の値を変えても予測不変、後側情報の利用、震災境界の切断。
    mask = np.array([[1], [0], [0], [1], [0], [1]], bool)
    y = np.array([[0.], [0.], [0.], [9.], [0.], [2.]])
    m, _, _, _ = smooth(y, mask, 1., .1, reset=4)
    y2 = y.copy(); y2[~mask] = 9999
    assert np.allclose(m, smooth(y2, mask, 1., .1, reset=4)[0])
    assert m[1, 0] > 1 and m[2, 0] > m[1, 0]
    y2[5, 0] = 999
    assert np.allclose(m[:4], smooth(y2, mask, 1., .1, reset=4)[0][:4])
    # 同一市町でも別セルは別状態。
    yy = np.concatenate([y, y+20], axis=1)
    mm = smooth(yy, np.repeat(mask, 2, axis=1), 1, .1, reset=4)[0]
    assert np.all(mm[:, 1]-mm[:, 0] > 19)
    # RTSを独立に計算したガウス条件付き分布と照合する。
    nt = len(y)
    covariance = np.zeros((nt, nt))
    for lo, hi in [(0, 4), (4, nt)]:
        i = np.arange(hi-lo)
        covariance[lo:hi, lo:hi] = SETTINGS['initial_variance'] + np.minimum(i[:, None], i[None, :])
    obs = np.flatnonzero(mask[:, 0])
    observed_cov = covariance[np.ix_(obs, obs)] + np.eye(len(obs))*.1
    expected = np.einsum('ij,j->i', covariance[:, obs], np.linalg.solve(observed_cov, y[obs, 0]))
    actual_m, actual_v, _, _ = smooth(y, mask, 1., .1, reset=4)
    expected_v = np.diag(covariance) - np.einsum('ij,ji->i', covariance[:, obs],
        np.linalg.solve(observed_cov, covariance[obs]))
    assert np.allclose(actual_m[:, 0], expected, atol=1e-6)
    assert np.allclose(actual_v[:, 0], expected_v, atol=1e-6)
    rng = np.random.default_rng(12)
    t = np.arange(len(DATES))
    raw = np.column_stack([10+np.sin(t/8), 30+3*np.sin(t/9), np.zeros(len(t))])
    ext = rng.normal(size=(*raw.shape, 1))
    fake = SimpleNamespace(diag=raw, total=raw, ratio=raw, cells=np.array(['a','b','c']),
                           features=lambda kind, groups: (ext, ['external']))
    train = t % 4 != 0
    before = fit_component(fake, 'D', ('evac',), False, train)
    raw[~train] = 10000
    after = fit_component(fake, 'D', ('evac',), False, train)
    assert np.allclose(before.mean, after.mean, atol=1e-10)
    assert np.allclose(before.variance, after.variance, atol=1e-10)
    assert np.all(before.mean[:, 2] == 0)
    toy = SimpleNamespace(cells=['a','b','c'], origin=np.array([0,0,1]),
                          support=lambda mask: np.array([True, True, True]),
                          fallback=lambda mask: np.array([.6,.4,1.]))
    ratio, _ = normalize(toy, np.array([[-1,2,0.], [0,0,-1.]]), None)
    assert np.allclose(ratio, [[0,1,1], [.6,.4,1]])
    inverse = symmetric_inverse(np.array([[2., 1.], [1., 3.]]))
    assert np.allclose(np.einsum('ij,jk->ik', inverse, [[2,1],[1,3]]), np.eye(2))
    print('[self-test] PASS missing targets / future observations / reset / independent cell states / '
          'full refit target masking / zero series / ratio fallback / regression inverse')


def run():
    """EXP03を自己検査・比較・採用し、提出物まで一度に再生成する。"""
    with run_log('EXP03', 'run', 'python3 src/EXP03_run.py') as log:
        log.stage('self-test', '数値実装の不変条件を検査')
        self_test()
        validation_predictions = validate(log)
        submit(validation_predictions, log)


if __name__ == '__main__':
    run()
