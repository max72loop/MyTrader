import pandas as pd

# -- Convertit une série de nombres en pourcentages arrondis --
def dataframe_to_percent(df, columns=None, digits=2):
    """
    Transforme certaines colonnes numériques d'un DataFrame en pourcentages formatés.
    """
    if columns is None:
        columns = df.select_dtypes(include=["number"]).columns

    for col in columns:
        df[col] = df[col].apply(lambda x: f"{x * 100:.{digits}f}%" if pd.notnull(x) else "")
    return df


# -- Convertit une série en pourcentage (ex : 0.1234 → 12.34%) --
def to_percent_str(series, digits=2):
    """
    Transforme une série numérique en texte formaté pour l'affichage Streamlit.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    return s.apply(lambda x: f"{x * 100:.{digits}f}%")
