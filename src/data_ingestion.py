import pandas as pd
from pathlib import Path

def load_data_from_url(url, save_path=None, filename=None, verbose=True):
    """
    Download a dataset from a URL and optionally save it locally.

    Parameters
    ----------
    url : str
        URL to download the data from
    save_path : Path or str, optional
        Directory to save the file
    filename : str, optional
        Name of the saved file (default: extracted from URL)
    verbose : bool
        Whether to print dataset info

    Returns
    -------
    df : pandas.DataFrame
    """
    # Load data
    df = pd.read_csv(url)

    # Save locally if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = url.split("/")[-1]

        file_path = save_path / filename
        df.to_csv(file_path, index=False)

        if verbose:
            print(f"Saved to: {file_path}")

    # Print summary
    if verbose:
        print(f"Shape: {df.shape}")
        print(f"\nColumn dtypes:\n{df.dtypes.value_counts()}")
        print(f"\nFirst five columns:\n{df.iloc[:, :5].head()}")

    return df