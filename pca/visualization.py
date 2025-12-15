from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd

from utils import plot_latent_space, plot_variance

EMB_DIR = Path("data/embeddings")

if __name__ == "__main__":
    all_files = list(EMB_DIR.glob("*.csv"))
    for file in all_files:
        parts = file.stem.split("_kpca_")
        if len(parts) != 2:
            continue
        kernel = parts[0]
        try:
            n_components = int(parts[1])
        except ValueError:
            continue
        df = pd.read_csv(file)
        pc_cols = [f"pc{i+1}" for i in range(n_components)]
        if not all(col in df.columns for col in pc_cols):
            continue
        Z = df[pc_cols].values
        y = df["label_uf_over"].values if "label_uf_over" in df.columns else None
        labels = y if y is not None else [0] * Z.shape[0]  # use actual labels if available, otherwise dummy labels
        title = f"{kernel} Kernel PCA with {n_components} components"
        save_path = f"outputs/{kernel}_kpca_{n_components}_latent_space.png"
        plot_latent_space(Z, labels=labels, title=title, save_path=save_path)

        var_file = EMB_DIR / f"{kernel}_kpca_{n_components}_explained_variance.txt"
        with open(var_file, "r") as f:
            var_proxy_str = f.read()
        var_proxy = eval(var_proxy_str)
        var_proxy = pd.Series(var_proxy)
        save_path = f"outputs/{kernel}_kpca_{n_components}_explained_variance.png"
        plot_variance(var_proxy, title=f"Cumulative Explained Variance Proxy - {kernel} Kernel PCA with {n_components} components", save_path=save_path)
