# scripts/preprocess_onet.py

from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data/onet_raw")
OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(exist_ok=True)

def load_onet_file(filename):
    return pd.read_csv(
        DATA_DIR / filename,
        sep="\t",
        dtype=str
    )

def clean_numeric(df, col):
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def main():
    print("Loading O*NET files...")

    skills = load_onet_file("Skills.txt")
    knowledge = load_onet_file("Knowledge.txt")
    abilities = load_onet_file("Abilities.txt")
    occupations = load_onet_file("Occupation Data.txt")

    # Keep only importance ratings
    skills = skills[skills["Scale ID"] == "IM"]
    knowledge = knowledge[knowledge["Scale ID"] == "IM"]
    abilities = abilities[abilities["Scale ID"] == "IM"]

    skills = clean_numeric(skills, "Data Value")
    knowledge = clean_numeric(knowledge, "Data Value")
    abilities = clean_numeric(abilities, "Data Value")

    print("Merging skill domains...")

    all_skills = pd.concat([
        skills[["O*NET-SOC Code", "Element Name", "Data Value"]],
        knowledge[["O*NET-SOC Code", "Element Name", "Data Value"]],
        abilities[["O*NET-SOC Code", "Element Name", "Data Value"]],
    ])

    # Pivot to Occupation × Skill matrix
    matrix = all_skills.pivot_table(
        index="O*NET-SOC Code",
        columns="Element Name",
        values="Data Value",
        aggfunc="mean"
    )

    matrix = matrix.fillna(0.0)

    print("Attaching occupation titles...")

    occ_titles = occupations[["O*NET-SOC Code", "Title"]].drop_duplicates()
    matrix = matrix.merge(
        occ_titles,
        left_index=True,
        right_on="O*NET-SOC Code",
        how="left"
    )

    matrix = matrix.set_index("Title")
    matrix = matrix.drop(columns=["O*NET-SOC Code"])

    print("Saving matrix...")

    matrix.to_parquet(OUT_DIR / "occupation_skill_matrix.parquet")

    print("Done.")
    print(f"Shape: {matrix.shape}")

if __name__ == "__main__":
    main()