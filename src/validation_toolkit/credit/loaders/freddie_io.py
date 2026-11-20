"""Conversion des fichiers Freddie Mac (txt séparés par '|', sans en-tête) en CSV avec en-têtes."""
import csv
from pathlib import Path

import pandas as pd


def _chain(first, rest):
    yield first
    yield from rest


def _verifier_champs(txt_path, sep):
    """Vérifie que toutes les lignes ont le même nombre de champs ; retourne ce nombre (None si vide).

    Nécessaire car pandas, en lecture par blocs, complète en silence une ligne trop courte
    avec des champs vides et tronque en silence une ligne trop longue.
    """
    sep_b = sep.encode()
    attendu = None
    with open(txt_path, "rb") as f:
        for i, ligne in enumerate(f, start=1):
            ligne = ligne.rstrip(b"\r\n")
            if not ligne:
                continue
            n = ligne.count(sep_b) + 1
            if attendu is None:
                attendu = n
            elif n != attendu:
                raise ValueError(
                    f"{Path(txt_path).name} : ligne {i} : {n} champs au lieu de {attendu}."
                )
    return attendu


def names_from_excel(path, sheet_name=0, column=0, skip_rows=0):
    """Lit la liste ordonnée des noms de colonnes dans le fichier Excel de layout.

    Inspecte d'abord le fichier pour choisir sheet_name, column et skip_rows :
        print(pd.read_excel(path, sheet_name=0, header=None).head(40))
    Retourne une liste de str, dans l'ordre du fichier.
    """
    df = pd.read_excel(path, sheet_name=sheet_name, header=None, skiprows=skip_rows)
    return df.iloc[:, column].dropna().astype(str).str.strip().tolist()


def txt_to_csv(txt_path, columns, csv_path=None, sep="|", chunksize=500_000):
    """Convertit un fichier txt sans en-tête en CSV avec en-têtes, par blocs (mémoire limitée).

    - Tout est lu comme du texte : les codes (statut d'impayés 'RA', 'XX', etc.)
      et les identifiants ne sont jamais convertis en nombres ni en valeurs manquantes.
    - Lève une erreur si le nombre de colonnes du fichier diffère du nombre de noms,
      ou si une ligne n'a pas le même nombre de champs que les autres.
    - Tolère une colonne finale vide (barre verticale en fin de ligne) et la supprime.

    Retourne (chemin du csv, nombre de lignes écrites).
    """
    txt_path = Path(txt_path)
    csv_path = Path(csv_path) if csv_path else txt_path.with_suffix(".csv")
    columns = list(columns)
    n_expected = len(columns)

    drop_last = None  # décidé sur le premier bloc, vérifié sur tous
    first = True
    n_rows = 0

    _verifier_champs(txt_path, sep)

    try:
        reader = pd.read_csv(
            txt_path,
            sep=sep,
            header=None,
            dtype=str,
            keep_default_na=False,
            quoting=csv.QUOTE_NONE,  # un guillemet dans un champ est un caractère comme un autre
            chunksize=chunksize,
        )
        first_chunk = next(reader)
    except (StopIteration, pd.errors.EmptyDataError):
        raise ValueError(f"{txt_path.name} : fichier vide.") from None

    for chunk in _chain(first_chunk, reader):
        n_cols = chunk.shape[1]
        if drop_last is None:
            if n_cols == n_expected:
                drop_last = False
            elif n_cols == n_expected + 1 and (chunk.iloc[:, -1] == "").all():
                drop_last = True
            else:
                raise ValueError(
                    f"{txt_path.name} : {n_cols} colonnes lues, {n_expected} noms fournis."
                )
        expected_cols = n_expected + 1 if drop_last else n_expected
        if n_cols != expected_cols:
            raise ValueError(
                f"{txt_path.name} : nombre de colonnes incohérent ({n_cols} au lieu de {expected_cols})."
            )
        if drop_last:
            if not (chunk.iloc[:, -1] == "").all():
                raise ValueError(f"{txt_path.name} : la dernière colonne n'est pas vide partout.")
            chunk = chunk.iloc[:, :-1]

        chunk.columns = columns
        chunk.to_csv(csv_path, mode="w" if first else "a", header=first, index=False)
        first = False
        n_rows += len(chunk)

    if n_rows == 0:
        raise ValueError(f"{txt_path.name} : fichier vide.")
    return csv_path, n_rows
