"""Прогнозирование продаж товара из датасета Online Retail (UCI).

Порядок блоков: загрузка и очистка (как в ДЗ 1), ряды трёх частот, модели, бэктест,
метрики, проверка на утечку, графики.

Прогноз из точки T везде строится только по наблюдениям 0..T-1.
"""

from __future__ import annotations

import itertools
import math
import urllib.request
import warnings
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

DATA_URLS = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx",
    "https://archive.ics.uci.edu/static/public/352/online+retail.zip",
)
XLSX_NAME = "Online Retail.xlsx"

TARGET = "85099B"  # JUMBO BAG RED RETROSPOT, как в ДЗ 1
CLOSED_WEEKDAY = 5  # по субботам магазин закрыт

# Границы рядов (берём только полные периоды):
# день - сетка Вс-Пт как в ДЗ 1; до 04.01 магазин 11 дней был закрыт, 09.12 - неполный день;
# неделя (Вс-Сб) - без неполной последней недели;
# месяц - декабрь 2010 полный, декабрь 2011 неполный.
DAILY_RANGE = ("2011-01-04", "2011-12-08")
WEEKLY_RANGE = ("2011-01-08", "2011-12-03")
MONTHLY_RANGE = ("2010-12-01", "2011-11-01")

HORIZONS = {"День": (1, 7, 31, 61), "Неделя": (1, 4, 8), "Месяц": (1, 2)}
# Начальная обучающая часть: у всех частот заканчивается в мае - середине июня 2011.
N_INIT = {"День": 140, "Неделя": 24, "Месяц": 6}

# Товары для бустинга и VAR отбираем только по январю-маю 2011, чтобы не смотреть в тест.
SELECTION_PERIOD = ("2011-01-04", "2011-05-31")


# ---------------------------------------------------------------------------
# 1. Загрузка и очистка (из ДЗ 1)
# ---------------------------------------------------------------------------

def download_dataset(data_dir: str | Path) -> Path:
    """Скачивает Online Retail.xlsx в data_dir, если его там ещё нет."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    xlsx = data_dir / XLSX_NAME
    if xlsx.exists():
        return xlsx
    for url in DATA_URLS:
        target = data_dir / Path(url).name.replace("%20", " ")
        try:
            urllib.request.urlretrieve(url, target)
        except OSError:
            continue
        if target.suffix == ".zip":
            with zipfile.ZipFile(target) as zf:
                zf.extract(XLSX_NAME, data_dir)
        if xlsx.exists():
            return xlsx
    raise RuntimeError(f"Не удалось скачать датасет; положите {XLSX_NAME} в {data_dir} вручную")


def load_transactions(xlsx_path: str | Path) -> pd.DataFrame:
    """Читает Excel (около 10 с) и кэширует его в pickle рядом."""
    xlsx_path = Path(xlsx_path)
    cache = xlsx_path.with_suffix(".pkl")
    if cache.exists() and cache.stat().st_mtime >= xlsx_path.stat().st_mtime:
        return pd.read_pickle(cache)
    df = pd.read_excel(xlsx_path, dtype={"InvoiceNo": str, "StockCode": str})
    df["StockCode"] = df["StockCode"].str.strip().str.upper()
    df["Description"] = df["Description"].astype("string").str.strip()
    df.to_pickle(cache)
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Убирает отменённые счета (номер на C), возвраты (Quantity <= 0) и нулевые цены."""
    drop = df["InvoiceNo"].str.startswith("C") | (df["Quantity"] <= 0) | (df["UnitPrice"] <= 0)
    clean = df.loc[~drop].copy()
    clean["Date"] = clean["InvoiceDate"].dt.normalize()
    return clean


def trading_calendar(clean: pd.DataFrame) -> pd.DatetimeIndex:
    """Дни, когда была хотя бы одна продажа (магазин работал)."""
    return pd.DatetimeIndex(np.sort(clean["Date"].unique()), name="Date")


# ---------------------------------------------------------------------------
# 2. Ряды
# ---------------------------------------------------------------------------

def daily_matrix(clean: pd.DataFrame, calendar: pd.DatetimeIndex, codes) -> pd.DataFrame:
    """Дневные продажи товаров: рабочий день без продаж = 0, магазин закрыт = NaN."""
    codes = list(codes)
    sales = (clean[clean["StockCode"].isin(codes)]
             .groupby(["Date", "StockCode"])["Quantity"].sum().unstack())
    full = pd.date_range(calendar.min(), calendar.max(), freq="D", name="Date")
    out = sales.reindex(index=full, columns=codes)
    open_day = out.index.isin(calendar)
    out.loc[open_day] = out.loc[open_day].fillna(0.0)
    return out.astype(float)


def _active_share(clean: pd.DataFrame, calendar: pd.DatetimeIndex, period) -> pd.Series:
    days = calendar[(calendar >= period[0]) & (calendar <= period[1])]
    return clean[clean["Date"].isin(days)].groupby("StockCode")["Date"].nunique() / len(days)


def select_panel(clean: pd.DataFrame, calendar: pd.DatetimeIndex, target: str = TARGET,
                 period=SELECTION_PERIOD, min_active: float = 0.8) -> list[str]:
    """Товары, которые продавались хотя бы в 80% рабочих дней января-мая. Целевой - первый."""
    sold = _active_share(clean, calendar, period)
    codes = sold[sold >= min_active].sort_values(ascending=False).index.tolist()
    return [target] + [c for c in codes if c != target]


def jumbo_peers(clean: pd.DataFrame, calendar: pd.DatetimeIndex, target: str = TARGET,
                period=SELECTION_PERIOD, min_active: float = 0.5) -> pd.DataFrame:
    """Другие сумки JUMBO BAG, которые регулярно продавались в январе-мае (второй ряд для VAR).
    Новинки второго полугодия не берём, иначе в сумме будет скачок из-за новых товаров."""
    is_jumbo = clean["Description"].str.contains("JUMBO BAG", na=False) & (clean["StockCode"] != target)
    sold = _active_share(clean[is_jumbo], calendar, period)
    codes = sold[sold >= min_active].sort_values(ascending=False).index
    desc = clean[clean["StockCode"].isin(codes)].groupby("StockCode")["Description"].first()
    return pd.DataFrame({"описание": desc.reindex(codes), "доля дней с продажами": sold[codes].round(3)})


@dataclass
class SeriesSet:
    """Ряд одной частоты. values - таблица время x товар, первый столбец - целевой товар."""
    name: str
    values: pd.DataFrame
    horizons: tuple[int, ...]
    n_init: int
    period: int | None = None  # сезонный период в шагах (только у дневного ряда)
    step: str = ""
    peers: pd.Series | None = None  # второй ряд для VAR

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.values.index

    @property
    def y(self) -> np.ndarray:
        return self.values.iloc[:, 0].to_numpy()

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def eval_start(self) -> int:
        """Первая оцениваемая позиция: для неё самый дальний горизонт прогнозируется
        ровно из конца начальной обучающей части."""
        return self.n_init + max(self.horizons) - 1

    @property
    def targets(self) -> range:
        """Оцениваемые позиции, общие для всех горизонтов. Позицию j горизонт h
        прогнозирует по наблюдениям 0..j-h."""
        return range(self.eval_start, self.n)

    @property
    def origins(self) -> range:
        """Точки прогноза T: модель видит 0..T-1 и прогнозирует T-1+h."""
        return range(self.n_init, self.n)

    def horizons_at(self, T: int) -> list[int]:
        """Горизонты, для которых прогноз из T попадает на оцениваемую дату."""
        return [h for h in self.horizons if self.eval_start <= T - 1 + h <= self.n - 1]

    def corrupted(self, T: int, seed: int = 0) -> "SeriesSet":
        """Копия, где всё начиная с позиции T заменено случайными числами (для проверки утечки)."""
        rng = np.random.default_rng(seed)
        vals = self.values.copy()
        vals.iloc[T:] = rng.uniform(0, 1e5, size=vals.iloc[T:].shape)
        peers = None
        if self.peers is not None:
            peers = self.peers.copy()
            peers.iloc[T:] = rng.uniform(0, 1e5, size=len(peers) - T)
        return replace(self, values=vals, peers=peers)


def build_series_sets(daily: pd.DataFrame, peers_daily: pd.Series) -> dict[str, SeriesSet]:
    """Дневной, недельный и месячный ряды из дневной таблицы продаж."""
    def grid(lo, hi, frame):
        part = frame.loc[lo:hi]
        return part[part.index.day_of_week != CLOSED_WEEKDAY]

    d = grid(*DAILY_RANGE, daily)
    sets = {"День": SeriesSet("День", d, HORIZONS["День"], N_INIT["День"], period=6,
                              step="рабочий день, Вс-Пт",
                              peers=grid(*DAILY_RANGE, peers_daily.to_frame()).iloc[:, 0])}
    for name, rule, (lo, hi), step in (("Неделя", "W-SAT", WEEKLY_RANGE, "неделя Вс-Сб"),
                                       ("Месяц", "MS", MONTHLY_RANGE, "месяц")):
        agg = daily.resample(rule).sum(min_count=1).loc[lo:hi]
        sets[name] = SeriesSet(name, agg, HORIZONS[name], N_INIT[name], step=step,
                               peers=peers_daily.resample(rule).sum(min_count=1).loc[lo:hi])
    return sets


def index_report(sets: dict[str, SeriesSet]) -> pd.DataFrame:
    """Проверка индекса: порядок, дубли, шаг, пропуски, размеры обучения и теста."""
    rows = []
    for s in sets.values():
        idx = s.index
        steps = pd.Series(idx[1:] - idx[:-1]).dt.days
        rows.append({
            "ряд": s.name, "шаг": s.step, "начало": idx[0].date(), "конец": idx[-1].date(),
            "N": s.n,
            "индекс возрастает": bool(idx.is_monotonic_increasing),
            "дублей": int(idx.duplicated().sum()),
            "шаг в днях (мин/макс)": f"{steps.min()}/{steps.max()}",
            "NaN (праздники)": int(np.isnan(s.y).sum()),
            "нулей": int((s.y == 0).sum()),
            "начальное обучение": s.n_init,
            "конец начального обучения": idx[s.n_init - 1].date(),
            "оцениваемых дат": len(s.targets),
            "первая оцениваемая дата": idx[s.eval_start].date(),
            "макс. горизонт": max(s.horizons),
        })
    return pd.DataFrame(rows).set_index("ряд")


def split_report(sets: dict[str, SeriesSet]) -> pd.DataFrame:
    """Для каждого горизонта: до какой даты модель видит данные и какие даты прогнозирует."""
    rows = []
    for s in sets.values():
        targets = np.array(s.targets)
        facts = int(np.isfinite(s.y[targets]).sum())
        for h in s.horizons:
            last_obs = targets - h
            rows.append({"ряд": s.name, "h": h,
                         "последнее наблюдение (с)": s.index[last_obs[0]].date(),
                         "последнее наблюдение (по)": s.index[last_obs[-1]].date(),
                         "прогноз на (с)": s.index[targets[0]].date(),
                         "прогноз на (по)": s.index[targets[-1]].date(),
                         "прогнозов": len(targets), "с фактом": facts})
    return pd.DataFrame(rows).set_index(["ряд", "h"])


# ---------------------------------------------------------------------------
# 3. Вспомогательные функции
# ---------------------------------------------------------------------------

def causal_fill(y: np.ndarray, period: int | None = None) -> np.ndarray:
    """Заполняет NaN только прошлым: тем же днём неделю назад, иначе последним значением."""
    y = np.asarray(y, dtype=float).copy()
    for i in range(len(y)):
        if np.isnan(y[i]):
            if period and i >= period and not np.isnan(y[i - period]):
                y[i] = y[i - period]
            elif i > 0:
                y[i] = y[i - 1]
    if np.isnan(y[0]):
        y[0] = y[np.isfinite(y)][0]
    return y


def naive_scale(y: np.ndarray) -> float:
    """Знаменатель MASE: MAE наивного прогноза на шаг вперёд на обучающей части."""
    d = np.abs(np.diff(np.asarray(y, dtype=float)))
    return float(np.nanmean(d))


def to_log(y):
    return np.log1p(np.asarray(y, dtype=float))


def from_log(z):
    """Обратно из log(1 + x); получается медиана прогноза, а не среднее."""
    return np.clip(np.expm1(np.asarray(z, dtype=float)), 0, None)


# ---------------------------------------------------------------------------
# 4. Модели
# ---------------------------------------------------------------------------

class Forecaster:
    """select() выбирает конфигурацию по начальной обучающей части (один раз),
    predict() переобучает модель на 0..T-1 и даёт прогнозы на заданные горизонты."""

    name = "?"

    def applicable(self, s: SeriesSet) -> bool:
        return True

    def select(self, s: SeriesSet) -> None:
        pass

    def describe(self) -> str:
        return ""

    def path(self, s: SeriesSet, T: int, H: int) -> np.ndarray:
        raise NotImplementedError

    def predict(self, s: SeriesSet, T: int, horizons) -> dict[int, float]:
        p = self.path(s, T, max(horizons))
        return {h: float(p[h - 1]) for h in horizons}


class Naive(Forecaster):
    """Последнее значение на все шаги вперёд."""
    name = "Наивная"

    def path(self, s, T, H):
        y = s.y[:T]
        return np.full(H, y[np.isfinite(y)][-1])


class SeasonalNaive(Forecaster):
    """Тот же день недели в последнюю неделю. Только для дневного ряда:
    у недель и месяцев нет даже одного полного года."""
    name = "Сезонная наивная"

    def applicable(self, s):
        return s.period is not None

    def path(self, s, T, H):
        m, y = s.period, s.y[:T]
        out = np.full(H, np.nan)
        for h in range(1, H + 1):
            k = T - 1 + h - m * math.ceil(h / m)  # последний известный тот же день недели
            while k >= 0 and np.isnan(y[k]):
                k -= m
            if k >= 0:
                out[h - 1] = y[k]
        return out


class ETS(Forecaster):
    """ETS на log(1 + x). Варианты: без тренда / затухающий тренд, без сезонности /
    недельная сезонность (только день). Вариант выбирается по AICc."""
    name = "ETS"

    def __init__(self):
        self.config: dict | None = None
        self.table: pd.DataFrame | None = None

    def _candidates(self, s):
        seasonal = [None, "add"] if s.period else [None]
        for trend, seas in itertools.product([None, "add"], seasonal):
            yield {"trend": trend, "damped_trend": trend is not None, "seasonal": seas,
                   "seasonal_periods": s.period if seas else None}

    @staticmethod
    def _fit(z, cfg):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ETSModel(z, error="add", **cfg).fit(disp=False, maxiter=1000)

    @staticmethod
    def label(cfg):
        t = "Ad" if cfg["trend"] else "N"
        return f"ETS(A,{t},{'A' if cfg['seasonal'] else 'N'})"

    def select(self, s):
        z = to_log(causal_fill(s.y[:s.n_init], s.period))
        rows = []
        for cfg in self._candidates(s):
            try:
                aicc = self._fit(z, cfg).aicc
            except Exception:  # мало точек для такого числа параметров
                aicc = np.inf
            rows.append({"модель": self.label(cfg), "AICc": aicc, "cfg": cfg})
        self.table = pd.DataFrame(rows).sort_values("AICc").reset_index(drop=True)
        self.config = self.table.loc[0, "cfg"]

    def describe(self):
        return self.label(self.config)

    def fit_at(self, s, T):
        return self._fit(to_log(causal_fill(s.y[:T], s.period)), self.config)

    def path(self, s, T, H):
        return from_log(self.fit_at(s, T).forecast(H))


def choose_d(z: np.ndarray, alpha: float = 0.05, min_n: int = 20) -> tuple[int, str]:
    """Порядок разности для ARIMA по тестам ADF и KPSS (AIC моделей с разным d сравнивать
    нельзя). Если ADF отвергает единичный корень, а KPSS не отвергает стационарность
    (около уровня или тренда), то d = 0, иначе d = 1. На коротком ряду берём d = 0."""
    z = z[np.isfinite(z)]
    if len(z) < min_n:
        return 0, f"d = 0: всего {len(z)} точек, тесты не применяем"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_p = adfuller(z, regression="ct", autolag="AIC")[1]
        kpss_c = kpss(z, regression="c", nlags="auto")[1]
        kpss_ct = kpss(z, regression="ct", nlags="auto")[1]
    note = f"ADF p = {adf_p:.3f}, KPSS p (уровень/тренд) = {kpss_c:.3f}/{kpss_ct:.3f}"
    if adf_p < alpha and max(kpss_c, kpss_ct) > alpha:
        return 0, "d = 0: " + note
    return 1, "d = 1: " + note


class ARIMA(Forecaster):
    """SARIMA на log(1 + x) через SARIMAX (праздничные NaN обрабатывает фильтр Калмана).
    d - по тестам, p, q, сезонные P, Q и тренд - по AICc."""
    name = "ARIMA"

    def __init__(self):
        self.order = self.seasonal_order = self.trend = None
        self.d_note = ""
        self.table: pd.DataFrame | None = None
        self._start = None

    def _grid(self, s, d):
        pq = range(3) if s.n_init >= 20 else range(2)
        seas = itertools.product(range(2), range(2)) if s.period else [(0, 0)]
        trends = ["c", "ct"] if d == 0 else ["n", "t"]
        for p, q, (P, Q), tr in itertools.product(pq, pq, list(seas), trends):
            yield (p, d, q), (P, 0, Q, s.period or 0), tr

    @staticmethod
    def _fit(z, order, sorder, trend, start=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = SARIMAX(z, order=order, seasonal_order=sorder, trend=trend,
                          enforce_stationarity=True, enforce_invertibility=True)
            return mod.fit(disp=False, maxiter=200, start_params=start)

    @staticmethod
    def label(order, sorder, trend):
        s = f"({sorder[0]},{sorder[1]},{sorder[2]})[{sorder[3]}]" if sorder[3] else ""
        return f"ARIMA{order}{s}, тренд={trend}"

    def select(self, s):
        z = to_log(s.y[:s.n_init])
        d, self.d_note = choose_d(z)
        rows = []
        for order, sorder, tr in self._grid(s, d):
            k = sum(order[::2]) + sorder[0] + sorder[2] + len(tr.replace("n", "")) + 1  # число параметров
            try:
                aicc = self._fit(z, order, sorder, tr).aicc if k < np.isfinite(z).sum() - 1 else np.inf
            except Exception:
                aicc = np.inf
            rows.append({"модель": self.label(order, sorder, tr), "AICc": aicc,
                         "spec": (order, sorder, tr)})
        self.table = pd.DataFrame(rows).sort_values("AICc").reset_index(drop=True)
        self.order, self.seasonal_order, self.trend = self.table.loc[0, "spec"]
        self._start = None

    def describe(self):
        return self.label(self.order, self.seasonal_order, self.trend)

    def fit_at(self, s, T, start=None):
        return self._fit(to_log(s.y[:T]), self.order, self.seasonal_order, self.trend, start=start)

    def path(self, s, T, H):
        res = self.fit_at(s, T, start=self._start)
        self._start = res.params  # стартовые значения для следующей точки, чтобы быстрее сходилось
        return from_log(res.forecast(H))


class VARModel(Forecaster):
    """VAR на log(1 + x) двух рядов: 85099B и другие сумки JUMBO BAG.
    Число лагов и тренд выбираются по AIC."""
    name = "VAR"

    def __init__(self, max_lags=None):
        self.max_lags = max_lags or {"День": 12, "Неделя": 4, "Месяц": 1}
        self.p = self.trend = None
        self.table: pd.DataFrame | None = None

    def _data(self, s, T):
        a = causal_fill(s.y[:T], s.period)
        b = causal_fill(s.peers.to_numpy()[:T], s.period)
        return np.column_stack([to_log(a), to_log(b)])

    def select(self, s):
        Z = self._data(s, s.n_init)
        maxp = self.max_lags[s.name]
        trends = ["c", "ct"] if s.n_init >= 20 else ["c"]
        rows = []
        for tr, p in itertools.product(trends, range(1, maxp + 1)):
            # у всех p одна и та же выборка, иначе AIC несравнимы
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    aic = VAR(Z[maxp - p:]).fit(p, trend=tr).aic
                except Exception:
                    aic = np.inf
            rows.append({"модель": f"VAR({p}), тренд={tr}", "AIC": aic, "spec": (p, tr)})
        self.table = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
        self.p, self.trend = self.table.loc[0, "spec"]

    def describe(self):
        return f"VAR({self.p}), тренд={self.trend}"

    def path(self, s, T, H):
        Z = self._data(s, T)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = VAR(Z).fit(self.p, trend=self.trend)
            f = res.forecast(Z[-self.p:], steps=H)
        return from_log(f[:, 0])


class GlobalLightGBM(Forecaster):
    """LightGBM, одна модель на всю панель товаров, своя модель на каждый горизонт h.

    По признакам в момент t предсказывается значение в t + h. Признаки считаются в
    log(1 + x) и нормируются на текущий уровень ряда, чтобы товары разного масштаба были
    сопоставимы. Для точки прогноза T обучаемся на строках с t + h <= T - 1.
    """
    name = "LightGBM"

    PARAMS = {"objective": "l1", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 20,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
              "lambda_l2": 1.0, "verbose": -1, "seed": 0, "deterministic": True,
              "force_row_wise": True, "num_threads": 1}
    SETTINGS = {  # w - окно для уровня, lags - число лагов, rolls - окна средних
        "День": {"w": 24, "lags": 14, "rolls": (6, 12, 48)},
        "Неделя": {"w": 4, "lags": 8, "rolls": (2, 8)},
        "Месяц": {"w": 3, "lags": 3, "rolls": (2,)},
    }

    def __init__(self, rounds: int = 300):
        self.rounds = rounds
        self._owner: SeriesSet | None = None  # для какого ряда посчитан кэш признаков
        self._cache: dict = {}
        self.boosters: dict[int, lgb.Booster] = {}  # последняя обученная модель для каждого h

    def describe(self):
        return f"LightGBM, своя модель на каждый h, {self.rounds} деревьев"

    def features(self, s: SeriesSet, h: int):
        """Признаки (время, товар, признак) и цель для горизонта h."""
        if self._owner is not s:
            self._owner, self._cache = s, {}
        if h in self._cache:
            return self._cache[h]
        cfg = self.SETTINGS[s.name]
        Z = pd.DataFrame(to_log(s.values.to_numpy()), index=s.index)
        level = Z.rolling(cfg["w"], min_periods=1).mean()
        feats, names = [], []
        for j in range(cfg["lags"]):
            feats.append(Z.shift(j) - level)
            names.append(f"lag_{j}")
        for k in cfg["rolls"]:
            feats.append(Z.rolling(k, min_periods=1).mean() - level)
            names.append(f"mean_{k}")
        feats.append(level)
        names.append("level")
        if s.period:  # тот же день недели, что у прогнозируемой даты, за 4 последние недели
            m = s.period
            for k in range(1, 5):
                lag = m * math.ceil(h / m) + m * (k - 1) - h  # всегда >= 0
                feats.append(Z.shift(lag) - level)
                names.append(f"same_weekday_{k}")
            dow = np.broadcast_to(((np.arange(s.n) + h) % m)[:, None], Z.shape).astype(float)
            feats.append(pd.DataFrame(dow, index=Z.index))
            names.append("weekday_target")
        X = np.stack([f.to_numpy() for f in feats], axis=-1)
        target = (Z.shift(-h) - level).to_numpy()
        self._cache[h] = (X, target, level.to_numpy(), names)
        return self._cache[h]

    def predict(self, s, T, horizons):
        out = {}
        for h in horizons:
            X, target, level, names = self.features(s, h)
            rows = slice(0, T - h)  # t = 0..T-1-h, тогда t + h <= T - 1
            Xtr = X[rows].reshape(-1, X.shape[-1])
            ytr = target[rows].reshape(-1)
            ok = np.isfinite(ytr)
            cat = ["weekday_target"] if "weekday_target" in names else []
            ds = lgb.Dataset(Xtr[ok], ytr[ok], feature_name=names, categorical_feature=cat,
                             free_raw_data=True)
            booster = lgb.train(self.PARAMS, ds, num_boost_round=self.rounds)
            self.boosters[h] = booster
            x_now = X[T - 1, 0][None, :]  # целевой товар в последней известной точке
            out[h] = float(from_log(level[T - 1, 0] + booster.predict(x_now)[0]))
        return out


def make_models() -> list[Forecaster]:
    return [Naive(), SeasonalNaive(), ETS(), ARIMA(), VARModel(), GlobalLightGBM()]


# ---------------------------------------------------------------------------
# 5. Бэктест и метрики
# ---------------------------------------------------------------------------

def backtest(models: list[Forecaster], s: SeriesSet, select: bool = True) -> pd.DataFrame:
    """Расширяющееся окно: в каждой точке T модель переобучается на 0..T-1 и прогнозирует
    оцениваемые даты, до которых ровно h шагов. У всех моделей одни и те же точки и даты."""
    rows = []
    for m in models:
        if not m.applicable(s):
            continue
        if select:
            m.select(s)
        for T in s.origins:
            hs = s.horizons_at(T)
            if not hs:
                continue
            scale = naive_scale(s.y[:T])
            for h, p in m.predict(s, T, hs).items():
                j = T - 1 + h
                rows.append({"модель": m.name, "частота": s.name, "h": h, "T": T,
                             "последнее наблюдение": s.index[T - 1], "дата прогноза": s.index[j],
                             "факт": s.y[j], "прогноз": p, "масштаб MASE": scale})
    return pd.DataFrame(rows)


def _metrics(g: pd.DataFrame) -> pd.Series:
    g = g[np.isfinite(g["факт"])]
    e = g["прогноз"] - g["факт"]
    return pd.Series({
        "n": len(g),
        "MAE": e.abs().mean(),
        "RMSE": math.sqrt((e ** 2).mean()),
        "WAPE, %": 100 * e.abs().sum() / g["факт"].abs().sum(),
        "MASE": (e.abs() / g["масштаб MASE"]).mean(),
        "смещение, %": 100 * e.sum() / g["факт"].sum(),
    })


def metrics_table(bt: pd.DataFrame) -> pd.DataFrame:
    """Сводная таблица метрик по модели, частоте и горизонту."""
    freq_order = {f: i for i, f in enumerate(HORIZONS)}
    out = bt.groupby(["частота", "h", "модель"], sort=False).apply(_metrics, include_groups=False)
    out = out.reset_index()
    out["_f"] = out["частота"].map(freq_order)
    out = out.sort_values(["_f", "h", "MASE"]).drop(columns="_f")
    out["n"] = out["n"].astype(int)
    return out.set_index(["частота", "h", "модель"])


def dm_test(e1: np.ndarray, e2: np.ndarray, h: int) -> tuple[float, float]:
    """Тест Диболда-Мариано по абсолютным ошибкам (с поправкой Харви-Лейборна-Ньюболда).
    Ошибки прогнозов на h шагов коррелированы до лага h-1, это учтено в дисперсии.
    Возвращает статистику и p-value; статистика < 0 значит, что первый прогноз точнее."""
    d = np.abs(e1) - np.abs(e2)
    d = d[np.isfinite(d)]
    n = len(d)
    dc = d - d.mean()
    gamma = [np.dot(dc[k:], dc[:n - k]) / n for k in range(h)]
    var = gamma[0] + 2 * sum(gamma[1:])
    if var <= 0:
        var = gamma[0]
    stat = d.mean() / math.sqrt(var / n) * math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return stat, 2 * stats.t.sf(abs(stat), df=n - 1)


def comparison_table(bt: pd.DataFrame, table: pd.DataFrame, metric: str = "MASE",
                     min_ratio: int = 3) -> pd.DataFrame:
    """Лучшая и вторая модели на каждой частоте и горизонте и p-value теста DM.
    Тест считаем, только если оцениваемых дат хотя бы в 3 раза больше h."""
    rows = []
    for (f, h), g in table.groupby(level=[0, 1], sort=False):
        g = g.droplevel([0, 1])
        best, second = g[metric].nsmallest(2).index
        cell = bt[(bt["частота"] == f) & (bt["h"] == h)]
        err = {m: (c["прогноз"] - c["факт"]).to_numpy() for m, c in cell.groupby("модель")}
        n = int(g.loc[best, "n"])
        row = {"частота": f, "h": h, "лучшая модель": best, metric: g.loc[best, metric],
               "WAPE, %": g.loc[best, "WAPE, %"],
               "MAE лучшей / MAE наивной": g.loc[best, "MAE"] / g.loc["Наивная", "MAE"],
               "вторая модель": second,
               "MAE второй / MAE лучшей": g.loc[second, "MAE"] / g.loc[best, "MAE"]}
        ok = n >= min_ratio * h
        for name, other in (("p (DM) против наивной", "Наивная"), ("p (DM) против второй", second)):
            row[name] = dm_test(err[best], err[other], h)[1] if ok and other != best else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index(["частота", "h"])


def cross_correlation(s: SeriesSet, lags=range(4)) -> pd.DataFrame:
    """Корреляции log(1 + x) целевого ряда со своими лагами и с лагами второго ряда VAR
    на начальной обучающей части."""
    n = s.n_init
    a = pd.Series(to_log(causal_fill(s.y, s.period))[:n])
    b = pd.Series(to_log(causal_fill(s.peers.to_numpy(), s.period))[:n])
    return pd.DataFrame({"с другими JUMBO BAG": [a.corr(b.shift(k)) for k in lags],
                         "с самим собой": [a.corr(a.shift(k)) if k else 1.0 for k in lags]},
                        index=pd.Index(list(lags), name="лаг k"))


def median_bias(s: SeriesSet) -> float:
    """На сколько процентов медиана факта на оцениваемых датах ниже среднего."""
    y = s.y[list(s.targets)]
    y = y[np.isfinite(y)]
    return 100 * (np.median(y) - y.mean()) / y.mean()


def leakage_check(models: dict[str, list[Forecaster]], sets: dict[str, SeriesSet]) -> pd.DataFrame:
    """Прогноз из конца начальной обучающей части на исходных данных и на данных, где всё
    после этой точки заменено случайными числами. Если прогнозы совпали, утечки нет."""
    rows = []
    for s in sets.values():
        T = s.n_init
        bad = s.corrupted(T)
        for m in models[s.name]:
            if not m.applicable(s):
                continue
            if isinstance(m, ARIMA):
                m._start = None
            a = m.predict(s, T, s.horizons)
            if isinstance(m, ARIMA):
                m._start = None
            b = m.predict(bad, T, s.horizons)
            diff = max(abs(a[h] - b[h]) for h in s.horizons)
            rows.append({"частота": s.name, "модель": m.name, "макс. разница прогнозов": diff,
                         "утечка": "нет" if diff < 1e-6 else "ЕСТЬ"})
    return pd.DataFrame(rows).set_index(["частота", "модель"])


# ---------------------------------------------------------------------------
# 6. Графики
# ---------------------------------------------------------------------------

COLORS = {"Факт": "black", "Наивная": "#9e9e9e", "Сезонная наивная": "#8d6e63", "ETS": "#1f77b4",
          "ARIMA": "#d62728", "VAR": "#9467bd", "LightGBM": "#2ca02c"}
UNIT = "Продажи, шт."


def _plural(n: int, one: str, few: str, many: str) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def _finish(ax, title, ylabel=UNIT, xlabel="Дата", legend=True):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    if legend:
        ax.legend(fontsize=8, loc="best")


def plot_split(sets: dict[str, SeriesSet]):
    """Ряды с начальной обучающей частью и оцениваемыми датами."""
    fig, axes = plt.subplots(len(sets), 1, figsize=(13, 3.2 * len(sets)))
    for ax, s in zip(axes, sets.values()):
        y = pd.Series(s.y, index=s.index)
        mk = "." if s.n < 100 else None
        ax.plot(y.iloc[:s.n_init + 1], color="#1f77b4", marker=mk,
                label=f"начальная обучающая часть (n = {s.n_init})")
        ax.plot(y.iloc[s.n_init:], color="#ff7f0e", marker=mk,
                label=f"добавляется в обучение по ходу бэктеста (n = {s.n - s.n_init})")
        ax.axvline(s.index[s.n_init - 1], color="grey", ls="--", lw=1)
        ax.axvspan(s.index[s.eval_start], s.index[-1], color="#ff7f0e", alpha=0.12,
                   label=f"оцениваемые даты (n = {len(s.targets)})")
        _finish(ax, f"{s.name}: шаг - {s.step}, N = {s.n}")
    fig.tight_layout()
    return fig


def plot_paths(models: list[Forecaster], s: SeriesSet, T: int, history: int = 60):
    """Прогнозы всех моделей из одной точки T. LightGBM показан точками только
    на горизонтах из задания, так как на каждый h у него отдельная модель."""
    H = max(s.horizons)
    fig, ax = plt.subplots(figsize=(13, 4))
    lo = max(0, T - history)
    hi = min(s.n, T + H)
    ax.plot(s.index[lo:hi], s.y[lo:hi], color="black", lw=1.3, label="Факт")
    ax.axvline(s.index[T - 1], color="grey", ls="--", lw=1)
    fut = s.index[T:T + H]
    for m in models:
        if not m.applicable(s):
            continue
        if isinstance(m, GlobalLightGBM):
            p = m.predict(s, T, s.horizons)
            ax.scatter([s.index[T - 1 + h] for h in s.horizons], list(p.values()),
                       color=COLORS[m.name], s=45, zorder=5, label=m.name)
        else:
            ax.plot(fut, m.path(s, T, H)[:len(fut)], color=COLORS[m.name], lw=1.4, label=m.name)
    _finish(ax, f"{s.name}: прогноз до h = {H} из точки {s.index[T - 1].date()} "
                f"(обучение на {T} наблюдениях)")
    fig.tight_layout()
    return fig


def plot_h_step(bt: pd.DataFrame, s: SeriesSet, models=None):
    """Для каждого h: факт и прогнозы, сделанные за h шагов до даты."""
    models = models or [m for m in COLORS if m != "Факт" and m in set(bt["модель"])]
    hs = s.horizons
    fig, axes = plt.subplots(len(hs), 1, figsize=(13, 2.9 * len(hs)), sharex=True)
    y = pd.Series(s.y, index=s.index).iloc[s.eval_start:]
    for ax, h in zip(np.atleast_1d(axes), hs):
        ax.plot(y.index, y.to_numpy(), color="black", lw=1.2, label="Факт",
                marker="." if s.n < 100 else None)
        g = bt[(bt["частота"] == s.name) & (bt["h"] == h)]
        for m in models:
            gm = g[g["модель"] == m]
            if len(gm):
                ax.plot(gm["дата прогноза"], gm["прогноз"], color=COLORS[m], lw=1.1, label=m,
                        marker="." if s.n < 100 else None)
        _finish(ax, f"{s.name}, h = {h}: прогнозы, сделанные за {h} "
                    f"{_plural(h, 'шаг', 'шага', 'шагов')} до даты",
                xlabel="Прогнозируемая дата")
    fig.tight_layout()
    return fig


def plot_metric_by_horizon(table: pd.DataFrame, metric: str = "MASE"):
    """Метрика по горизонтам, отдельно для каждой частоты."""
    freqs = list(dict.fromkeys(table.index.get_level_values(0)))
    fig, axes = plt.subplots(1, len(freqs), figsize=(15, 4))
    for ax, f in zip(axes, freqs):
        t = table.loc[f][metric].unstack("модель")
        for m in [c for c in COLORS if c in t.columns]:
            ax.plot(range(len(t)), t[m], marker="o", color=COLORS[m], label=m)
        ax.set_xticks(range(len(t)), [str(h) for h in t.index])
        if metric == "MASE":
            ax.axhline(1, color="grey", ls=":", lw=1)
        _finish(ax, f"{f}: {metric} по горизонтам", ylabel=metric, xlabel="Горизонт h, шагов")
    fig.tight_layout()
    return fig


def plot_importance(model: GlobalLightGBM, s: SeriesSet, T: int, top: int = 12):
    """Важность признаков LightGBM (gain, %) на самом коротком и самом длинном горизонте."""
    hs = (min(s.horizons), max(s.horizons))
    model.predict(s, T, hs)
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.32 * top + 1.5))
    for ax, h in zip(axes, hs):
        b = model.boosters[h]
        imp = pd.Series(b.feature_importance("gain"), index=b.feature_name())
        imp = (100 * imp / imp.sum()).sort_values().tail(top)
        ax.barh(imp.index, imp.to_numpy(), color=COLORS["LightGBM"])
        _finish(ax, f"{s.name}, h = {h}: важность признаков", ylabel="Признак",
                xlabel="Доля gain, %", legend=False)
    fig.tight_layout()
    return fig


def plot_heatmap(table: pd.DataFrame, metric: str = "WAPE, %"):
    """Тепловая карта метрики: модели по строкам, частота и горизонт по столбцам."""
    t = table[metric].unstack(["частота", "h"])
    cols = list(dict.fromkeys((f, h) for f, h, _ in table.index))
    order = [m for m in COLORS if m in t.index]
    t = t.loc[order, cols]
    fig, ax = plt.subplots(figsize=(13, 0.6 * len(order) + 1.8))
    im = ax.imshow(t.to_numpy(dtype=float), cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(t.shape[1]), [f"{f}\nh={h}" for f, h in t.columns])
    ax.set_yticks(range(len(order)), order)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.iat[i, j]
            ax.text(j, i, "-" if np.isnan(v) else f"{v:.0f}" if v >= 10 else f"{v:.2f}",
                    ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label=metric)
    ax.set_title(f"{metric} по всем моделям, частотам и горизонтам (зеленее = точнее)")
    ax.set_xlabel("Частота и горизонт")
    ax.set_ylabel("Модель")
    fig.tight_layout()
    return fig
