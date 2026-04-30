import numpy as np
import pandas as pd

def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum()
    q = q / q.sum()
    return 0.5 * np.abs(p - q).sum()



def compute_metrics_by_breed(wide_df: pd.DataFrame, alpha: float = 1e-6) -> pd.DataFrame:
    out = []
    for breed, g in wide_df.groupby("breed", sort=False):
        p = g["functional"].to_numpy()
        q = g["decorative"].to_numpy()

        tv_p = tv_distance(p, q)

        out.append({
            "breed": breed,
            "tv_functional_decorative": tv_p,
        })
    return pd.DataFrame(out).sort_values("tv_functional_decorative", ascending=False)



def compute_metrics_by_font_name(wide_df: pd.DataFrame, alpha: float = 1e-6) -> pd.DataFrame:
    # font name is the column name
    font_names = wide_df.columns[1:]
    for font_name in font_names:
        p = wide_df[font_name].to_numpy()
        q = wide_df["text"].to_numpy()
        tv_p = tv_distance(p, q)
        tv_q = tv_distance(q, p)
        out.append({
            "font_name": font_name,
            "tv_p": tv_p,
            "tv_q": tv_q
        })
    return pd.DataFrame(out).sort_values("tv_p", ascending=False)



df = pd.read_csv(f"")
metrics_df = compute_metrics_by_breed(df, alpha=1e-6)
metrics_df.to_csv(f"", index=False)
