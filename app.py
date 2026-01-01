import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from category_encoders import TargetEncoder

#добавлено из ноутбука
class CarFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Делает:
      - *_was_nan для: mileage, engine, max_power, seats, torque_nm, max_torque_rpm
      - brand = first token из name, brand_grp (rare->other), rare, premium
      - age=2021-year, age2
      - log_km_driven/log_max_power/log_torque_nm (log1p)
      - log_km_per_year = log_km_driven / age
      - is_test_drive, owner_n
      - engine_1 = engine/1000
      - is_automatic
      - name_count/log_name_count/name_unseen
      - name_te через TargetEncoder на log1p(y), с OOF для train (5-fold, rs=42)
    """

    def __init__(
        self,
        ref_year: int = 2021,
        min_brand_count: int = 10,
        te_min_samples_leaf: int = 200,
        te_smoothing: float = 50.0,
        te_n_splits: int = 5,
        random_state: int = 42,
        premium_brands=None,
    ):
        self.ref_year = ref_year
        self.min_brand_count = min_brand_count
        self.te_min_samples_leaf = te_min_samples_leaf
        self.te_smoothing = te_smoothing
        self.te_n_splits = te_n_splits
        self.random_state = random_state
        self.premium_brands = premium_brands or {
            "Mercedes-Benz", "BMW", "Audi", "Jaguar", "Lexus", "Volvo", "Land"
        }

        self.na_cols_ = ["mileage", "engine", "max_power", "seats", "torque_nm", "max_torque_rpm"]
        self.owner_map_ = {
            "First Owner": 1,
            "Second Owner": 2,
            "Third Owner": 3,
            "Fourth & Above Owner": 4
        }

    @staticmethod
    def _as_str(s: pd.Series) -> pd.Series:
        return s.fillna("").astype(str)

    @staticmethod
    def _brand_from_name(name: pd.Series) -> pd.Series:
        return (
            name.fillna("")
                .astype(str)
                .str.strip()
                .str.split()
                .str[0]
                .fillna("")
        )

    @staticmethod
    def _to_num(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce")

    def fit(self, X: pd.DataFrame, y):
        X = X.copy()

        # --- brand rarity ---
        name = self._as_str(X.get("name", pd.Series([""] * len(X), index=X.index)))
        brand = self._brand_from_name(name)
        brand_counts = brand.value_counts()
        self.rare_brands_ = set(brand_counts[brand_counts < self.min_brand_count].index.tolist())

        # --- name counts ---
        self.name_counts_ = name.value_counts().to_dict()

        # --- TargetEncoder fitted on ALL train using log1p(y) (для inference) ---
        y_log = np.log1p(np.asarray(y, dtype=float))
        self.te_all_ = TargetEncoder(
            cols=["name"],
            min_samples_leaf=self.te_min_samples_leaf,
            smoothing=self.te_smoothing
        )
        self.te_all_.fit(pd.DataFrame({"name": name}), y_log)

        return self

    def _build_deterministic_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        out = pd.DataFrame(index=X.index)

        # raw cols
        name = self._as_str(X.get("name", pd.Series([""] * len(X), index=X.index)))
        year = self._to_num(X.get("year", np.nan))
        km = self._to_num(X.get("km_driven", np.nan))
        mileage = self._to_num(X.get("mileage", np.nan))
        engine = self._to_num(X.get("engine", np.nan))
        max_power = self._to_num(X.get("max_power", np.nan))
        torque_nm = self._to_num(X.get("torque_nm", np.nan))
        max_torque_rpm = self._to_num(X.get("max_torque_rpm", np.nan))
        seats = self._to_num(X.get("seats", np.nan))

        fuel = self._as_str(X.get("fuel", pd.Series([""] * len(X), index=X.index)))
        seller_type = self._as_str(X.get("seller_type", pd.Series([""] * len(X), index=X.index)))
        transmission = self._as_str(X.get("transmission", pd.Series([""] * len(X), index=X.index)))
        owner = self._as_str(X.get("owner", pd.Series([""] * len(X), index=X.index)))

        # was_nan flags
        for c, s in [
            ("mileage", mileage),
            ("engine", engine),
            ("max_power", max_power),
            ("seats", seats),
            ("torque_nm", torque_nm),
            ("max_torque_rpm", max_torque_rpm),
        ]:
            out[f"{c}_was_nan"] = s.isna().astype(int)

        # brand_grp / rare / premium
        brand = self._brand_from_name(name)
        out["brand_grp"] = brand.where(~brand.isin(self.rare_brands_), "other")
        out["rare"] = (out["brand_grp"] == "other").astype(int)
        out["premium"] = brand.isin(self.premium_brands).astype(int)

        # age / age2
        out["age"] = self.ref_year - year
        out["age2"] = out["age"] ** 2

        # logs (log1p), clip <0 to 0 for safety
        out["log_km_driven"] = np.log1p(km.clip(lower=0))
        out["log_max_power"] = np.log1p(max_power.clip(lower=0))
        out["log_torque_nm"] = np.log1p(torque_nm.clip(lower=0))

        # log_km_per_year = log_km_driven / age
        out["log_km_per_year"] = out["log_km_driven"] / out["age"]

        # owner features
        out["is_test_drive"] = (owner == "Test Drive Car").astype(int)
        out["owner_n"] = owner.map(self.owner_map_).fillna(0).astype(int)

        # engine_1
        out["engine_1"] = engine / 1000.0

        # is_automatic
        out["is_automatic"] = (transmission == "Automatic").astype(int)

        # name_count, log_name_count, name_unseen
        name_count = name.map(self.name_counts_).fillna(0).astype(int)
        out["log_name_count"] = np.log1p(name_count)
        out["name_unseen"] = (name_count == 0).astype(int)

        # raw numeric used directly in model
        out["mileage"] = mileage
        out["max_torque_rpm"] = max_torque_rpm

        # categorical cols used directly
        out["fuel"] = fuel
        out["seller_type"] = seller_type
        out["seats"] = seats

        # убрать inf (например, если year=2021 => age=0)
        out = out.replace([np.inf, -np.inf], np.nan)

        return out

    def fit_transform(self, X: pd.DataFrame, y):
        # 1) fit (учим редкие бренды, name_counts, te_all)
        self.fit(X, y)

        # 2) deterministic features
        out = self._build_deterministic_features(X)

        # 3) OOF Target Encoding для name_te
        name = self._as_str(X.get("name", pd.Series([""] * len(X), index=X.index)))
        y_log = np.log1p(np.asarray(y, dtype=float))

        oof = np.full(len(out), np.nan, dtype=float)
        kf = KFold(n_splits=self.te_n_splits, shuffle=True, random_state=self.random_state)

        name_df = pd.DataFrame({"name": name})
        for tr_idx, val_idx in kf.split(name_df):
            te = TargetEncoder(
                cols=["name"],
                min_samples_leaf=self.te_min_samples_leaf,
                smoothing=self.te_smoothing
            )
            te.fit(name_df.iloc[tr_idx], y_log[tr_idx])
            oof[val_idx] = te.transform(name_df.iloc[val_idx])["name"].astype(float).values

        # если вдруг где-то NaN (редко) — подставим te_all
        if np.isnan(oof).any():
            oof[np.isnan(oof)] = self.te_all_.transform(name_df)["name"].astype(float).values[np.isnan(oof)]

        out["name_te"] = oof
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = self._build_deterministic_features(X)

        name = self._as_str(X.get("name", pd.Series([""] * len(X), index=X.index)))
        name_df = pd.DataFrame({"name": name})
        out["name_te"] = self.te_all_.transform(name_df)["name"].astype(float).values

        return out



st.set_page_config(
    page_title="Car Price Prediction",
    layout="wide"
)

st.title("А этот автомобиль сколько стоит?")

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODELS_DIR / "full_pipe.pkl"
RAW_COLS_PATH = MODELS_DIR / "raw_feature_names.pkl"
FINAL_COLS_PATH = MODELS_DIR / "final_feature_names.pkl"

def format_price(x: float) -> str:
    try:
        return f"₹ {float(x):,.0f}".replace(",", " ")
    except Exception:
        return str(x)


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(RAW_COLS_PATH, "rb") as f:
        raw_cols = pickle.load(f)
    with open(FINAL_COLS_PATH, "rb") as f:
        final_cols = pickle.load(f)
    return model, raw_cols, final_cols

try:
    MODEL, RAW_COLS, FINAL_COLS = load_model()
except Exception as e:
    st.error(f"❌ Ошибка загрузки модели: {e}")
    st.stop()

PIPE = MODEL

def get_categories(pipe) -> dict:
    prep = pipe.named_steps["preparation"]
    ohe = prep.named_transformers_["cat"].named_steps["ohe"]
    cat_cols = prep.transformers_[1][2] 
    return dict(zip(cat_cols, ohe.categories_))


def get_coef_table(pipe) -> pd.DataFrame:
    feature_names = None
    try:
        feature_names = pipe.named_steps["preparation"].get_feature_names_out()
    except Exception:
        feature_names = FINAL_COLS

    # coefficients
    coef = None
    intercept = None
    try:
        ttr = pipe.named_steps["model"]
        ridge = getattr(ttr, "regressor_", None) or getattr(ttr, "regressor", None)
        coef = ridge.coef_
        intercept = ridge.intercept_
    except Exception:
        try:
            ridge = pipe.named_steps["model"]
            coef = ridge.coef_
            intercept = ridge.intercept_
        except Exception:
            pass

    feature_names = list(feature_names)
    coef = np.asarray(coef, dtype=float)

    df = pd.DataFrame({
        "feature": feature_names,
        "coef": coef,
        "abs_coef": np.abs(coef)
    }).sort_values("abs_coef", ascending=False).reset_index(drop=True)
    df.attrs["intercept"] = float(intercept) if intercept is not None else None
    return df


with st.sidebar:
    st.header("Данные")

    uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])

    if uploaded_file is None:
        st.info("👈 Загрузите CSV файл для начала работы")
        st.stop()

    df = pd.read_csv(uploaded_file)

    st.subheader("ℹ️ Информация о файле")
    st.write(f"**Строк:** {df.shape[0]}")
    st.write(f"**Столбцов:** {df.shape[1]}")

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    st.write(f"**Категориальных столбцов:** {len(cat_cols)}")

    if cat_cols:
        # Общее число уникальных категорий (сумма уникальных значений по cat-колонкам)
        total_unique = int(sum(df[c].nunique(dropna=True) for c in cat_cols))
        st.write(f"**Всего категорий (unique):** {total_unique}")

        # Показать детали по каждой категориальной колонке (топ-10)
        with st.expander("Показать категории по столбцам"):
            info = (
                pd.DataFrame({
                    "col": cat_cols,
                    "unique": [df[c].nunique(dropna=True) for c in cat_cols]
                })
                .sort_values("unique", ascending=False)
                .reset_index(drop=True)
            )
            st.dataframe(info, use_container_width=True)


tabs = st.tabs(["EDA", "Определение цены", "Важность признаков"])

with tabs[0]:
    st.subheader("Исследовательский анализ данных")


    st.write("Данные:")
    st.dataframe(df.head(20), use_container_width=True)

    if "name" in df.columns and "brand" not in df.columns:
        df = df.copy()
        df["brand"] = df["name"].astype(str).str.strip().str.split().str[0]

    # 1) Распределение таргета
    if "selling_price" in df.columns:
        st.markdown("### 🎯 Распределение цены автомобиля")
        st.plotly_chart(
            px.histogram(df, x="selling_price", nbins=50, title="Распределение цены"),
            use_container_width=True
        )

        df["_log_selling_price"] = np.log1p(pd.to_numeric(df["selling_price"], errors="coerce"))
        st.plotly_chart(
            px.histogram(df, x="_log_selling_price", nbins=50, title="Логарифм цены"),
            use_container_width=True
        )
    else:
        st.warning("В датасете нет колонки selling_price")

    st.divider()

    # 2) Числовые и категориальные признаки
    drop_cols = {"selling_price", "_log_selling_price"}
    cols = [c for c in df.columns if c not in drop_cols]

    num_cols = df[cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in cols if c not in num_cols]

    st.markdown("### 🔢 Числовые признаки: распределения")
    for c in num_cols:
        st.plotly_chart(
            px.histogram(df, x=c, nbins=50, title=f"Распределение {c}"),
            use_container_width=True
        )

    st.divider()

    st.markdown("### 🏷️ Категориальные признаки: распределения")
    for c in cat_cols:
        vc = df[c].astype(str).value_counts().head(30).reset_index()
        vc.columns = [c, "count"]
        st.plotly_chart(
            px.bar(vc, x=c, y="count", title=f"Количество {c}"),
            use_container_width=True
        )

    st.divider()

    st.markdown("### Boxplot: зависимость цены и марки автомобиля")
    if "brand" in df.columns and "selling_price" in df.columns:
        top_brands = df["brand"].astype(str).value_counts().head(20).index
        d = df[df["brand"].astype(str).isin(top_brands)].copy()
        fig = px.box(d, x="brand", y="selling_price", points="outliers",
                     title="Selling price by brand (top 20 brands)")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Для boxplot нужны колонки: name (или brand) и selling_price.")

    st.divider()

    # 4) Корреляция Пирсона 
    st.markdown("### Корреляция Пирсона")
    if len(num_cols) >= 2:
        corr_pearson = df[num_cols].corr(method="pearson")
        fig = px.imshow(
            corr_pearson,
            title="Pearson correlation (numeric)",
            aspect="auto",
            zmin=-1, zmax=1
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Недостаточно числовых колонок для корреляции Пирсона.")



with tabs[1]:
    st.subheader("Предсказание цены автомобиля")

    cats = get_categories(PIPE)

    # Категориальные: берём из обученного OHE
    fuel = st.selectbox("fuel", options=list(cats.get("fuel", ["Petrol", "Diesel"])))
    seller_type = st.selectbox("seller_type", options=list(cats.get("seller_type", ["Individual", "Dealer"])))
    owner = st.selectbox(
        "owner",
        options=[
            "First Owner", "Second Owner", "Third Owner",
            "Fourth & Above Owner", "Test Drive Car"
        ]
    )
    transmission = st.selectbox("transmission", options=["Manual", "Automatic"])

    # Числовые + name
    name = st.text_input("name (полное название модели)", value="Maruti Swift Dzire VDI")
    year = st.number_input("year", min_value=1980, max_value=2021, value=2016, step=1)
    km_driven = st.number_input("km_driven", min_value=0, value=60000, step=1000)

    mileage = st.number_input("mileage", min_value=0.0, value=18.0, step=0.1)
    engine = st.number_input("engine", min_value=0.0, value=1248.0, step=10.0)
    max_power = st.number_input("max_power", min_value=0.0, value=74.0, step=1.0)
    seats = st.number_input("seats", min_value=2, max_value=20, value=5, step=1)

    torque_nm = st.number_input("torque_nm", min_value=0.0, value=190.0, step=1.0)
    max_torque_rpm = st.number_input("max_torque_rpm", min_value=0.0, value=2000.0, step=50.0)

    if st.button("Оценить", type="primary"):
        row = pd.DataFrame([{
            "name": name,
            "year": year,
            "km_driven": km_driven,
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "owner": owner,
            "mileage": mileage,
            "engine": engine,
            "max_power": max_power,
            "seats": seats,
            "torque_nm": torque_nm,
            "max_torque_rpm": max_torque_rpm,
        }])

        pred = float(PIPE.predict(row)[0])
        st.success(f"Предсказанная цена: {format_price(pred)}")



with tabs[2]:
    st.subheader("Важность признаков в модели (Top-20)")

    coef_df = get_coef_table(PIPE)
    intercept = coef_df.attrs.get("intercept", None)
    if intercept is not None:
        st.caption(f"Intercept: {intercept:.6f}")

    top_k = 30
    view = coef_df.head(top_k).copy()

    st.write("Топ-30 признаков по важности")
    st.dataframe(view, use_container_width=True)

    st.divider()
    st.write("30 наиболее важных признаков")

    fig = px.bar(
        view.iloc[::-1],         
        x="coef",
        y="feature",
        orientation="h",
        height=650,
        title="Top-30 признаков"
    )
    fig.update_layout(xaxis_title="Вес", yaxis_title="Признак")
    st.plotly_chart(fig, use_container_width=True)