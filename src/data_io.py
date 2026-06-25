import pandas as pd
from pathlib import Path
from urllib.error import URLError, HTTPError

def load_dataset(
    file_path,
    url=None,
    force_download=False,
    verbose=True
):
    """
    Load a dataset from a local cache or download it from a URL.

    Parameters
    ----------
    file_path : str or Path
        Local file path for caching
    url : str, optional
        URL to download the data from
    force_download : bool, default=False
        If True, always re-download the dataset
    verbose : bool, default=True
        Whether to print dataset info

    Returns
    -------
    df : pandas.DataFrame
    """
    file_path = Path(file_path)

    def _read_data(source, suffix):
        """Read dataset based on file extension."""
        if suffix == ".csv":
            return pd.read_csv(source)
        elif suffix in [".xls", ".xlsx"]:
            return pd.read_excel(source)
        elif suffix == ".parquet":
            return pd.read_parquet(source)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    try:
        # ---------------------------
        # Load from local cache
        # ---------------------------
        if file_path.exists() and not force_download:
            if verbose:
                print("Loading data from local cache...")
            df = _read_data(file_path, file_path.suffix.lower())

        # ---------------------------
        # Download from URL
        # ---------------------------
        else:
            if url is None:
                raise ValueError("URL must be provided if file does not exist.")

            if verbose:
                print("Downloading dataset...")

            suffix = file_path.suffix.lower()

            try:
                df = _read_data(url, suffix)
            except Exception:
                # Fallback: try CSV if suffix missing or wrong
                if verbose:
                    print("Fallback: attempting to read as CSV...")
                df = pd.read_csv(url)

            # Save locally
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if suffix == ".csv":
                df.to_csv(file_path, index=False)
            elif suffix in [".xls", ".xlsx"]:
                df.to_excel(file_path, index=False)
            elif suffix == ".parquet":
                df.to_parquet(file_path, index=False)
            else:
                # Default fallback
                df.to_csv(file_path, index=False)

            if verbose:
                print(f"Saved to: {file_path}")

        # ---------------------------
        # Basic validation
        # ---------------------------
        if df.empty:
            raise ValueError("Loaded dataset is empty.")

        # ---------------------------
        # Summary output
        # ---------------------------
        if verbose:
            print(f"\nShape: {df.shape}")
            print(f"\nColumn dtypes:\n{df.dtypes.value_counts()}")
            print(f"\nPreview (first 5 rows, first 5 cols):")
            print(df.iloc[:, :5].head())

        return df

    # ---------------------------
    # Error handling
    # ---------------------------
    except (HTTPError, URLError) as e:
        raise RuntimeError(f"Network error while downloading dataset: {e}")
    except FileNotFoundError:
        raise RuntimeError(f"File not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")