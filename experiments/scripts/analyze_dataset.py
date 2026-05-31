import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset


def analyze_sts_ultimate():
    print("📥 Pobieranie zbioru danych STS Benchmark...")
    ds = load_dataset("mteb/stsbenchmark-sts")

    # Konwersja każdego podziału na DataFrame i dodanie kolumny 'split'
    all_data = []
    for split in ds.keys():
        df = ds[split].to_pandas()
        df["split"] = split.capitalize()  # Train, Validation, Test

        # Obliczanie długości zdań
        df["word_count_s1"] = df["sentence1"].apply(lambda x: len(str(x).split()))
        df["word_count_s2"] = df["sentence2"].apply(lambda x: len(str(x).split()))
        df["max_word_count"] = df[["word_count_s1", "word_count_s2"]].max(axis=1)

        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)

    print("\n" + "=" * 50)
    print("1. PODSTAWOWE INFORMACJE O PODZIAŁACH")
    print("=" * 50)
    for split in combined_df["split"].unique():
        subset = combined_df[combined_df["split"] == split]
        print(f"\n--- Podział: {split.upper()} ---")
        print(f"Liczba wierszy: {len(subset)}")
        print(
            f"Brakujące wartości (Nulls):\n{subset[['sentence1', 'sentence2', 'score']].isnull().sum()}"
        )

    print("\n" + "=" * 50)
    print("2. ANALIZA METADANYCH (Źródła tekstów, Rok, itp.)")
    print("=" * 50)
    # Wyłapujemy kolumny inne niż tekstowe, liczbowe i te, które sami dodaliśmy
    metadata_cols = [
        col
        for col in combined_df.columns
        if col
        not in [
            "sentence1",
            "sentence2",
            "score",
            "split",
            "word_count_s1",
            "word_count_s2",
            "max_word_count",
        ]
    ]

    if metadata_cols:
        for col in metadata_cols:
            print(f"\nRozkład wartości w kolumnie '{col}':")
            # Grupujemy po podziale i zliczamy kategorie
            dist = combined_df.groupby(["split", col]).size().unstack(fill_value=0)
            print(dist)
    else:
        print("Brak dodatkowych kolumn kategorycznych w tej wersji zbioru.")

    print("\n" + "=" * 50)
    print("3. STATYSTYKI OCEN I DŁUGOŚCI ZDAŃ")
    print("=" * 50)
    for split in combined_df["split"].unique():
        subset = combined_df[combined_df["split"] == split]
        print(f"\n--- Podział: {split.upper()} ---")
        stats_score = subset["score"].describe()
        s1_mean, s2_mean = (
            subset["word_count_s1"].mean(),
            subset["word_count_s2"].mean(),
        )
        max_len = subset["max_word_count"].max()

        print(
            f"Oceny (Score)   -> Średnia: {stats_score['mean']:.2f} | Mediana: {stats_score['50%']:.2f} | Odchylenie std: {stats_score['std']:.2f}"
        )
        print(
            f"Długość (Słowa) -> Średnio S1: {s1_mean:.1f} | Średnio S2: {s2_mean:.1f} | Najdłuższe w zbiorze: {max_len}"
        )

    print("\n" + "=" * 50)
    print("4. PRZYKŁADY ZDAŃ DLA RÓŻNYCH ZAKRESÓW PODOBIEŃSTWA")
    print("=" * 50)

    print("\n🟢 WYSOCE PODOBNE (Score > 4.5) - Zbiór Testowy:")
    high_subset = combined_df[
        (combined_df["score"] >= 4.5) & (combined_df["split"] == "Test")
    ]
    if not high_subset.empty:
        example = high_subset.sample(1).iloc[0]
        print(f"Źródło (Genre): {example.get('genre', 'Brak danych')}")
        print(f"Zdanie 1: {example['sentence1']}")
        print(f"Zdanie 2: {example['sentence2']}")
        print(f"Ocena:    {example['score']:.2f}")

    print("\n🟡 ŚREDNIO PODOBNE (2.0 < Score <= 3.0) - Zbiór Treningowy:")
    mid_subset = combined_df[
        (combined_df["score"] > 2.0)
        & (combined_df["score"] <= 3.0)
        & (combined_df["split"] == "Train")
    ]
    if not mid_subset.empty:
        example = mid_subset.sample(1).iloc[0]
        print(f"Źródło (Genre): {example.get('genre', 'Brak danych')}")
        print(f"Zdanie 1: {example['sentence1']}")
        print(f"Zdanie 2: {example['sentence2']}")
        print(f"Ocena:    {example['score']:.2f}")

    print("\n🔴 CAŁKOWICIE RÓŻNE (Score <= 1.0) - Zbiór Walidacyjny:")
    low_subset = combined_df[
        (combined_df["score"] <= 1.0) & (combined_df["split"] == "Validation")
    ]
    if not low_subset.empty:
        example = low_subset.sample(1).iloc[0]
        print(f"Źródło (Genre): {example.get('genre', 'Brak danych')}")
        print(f"Zdanie 1: {example['sentence1']}")
        print(f"Zdanie 2: {example['sentence2']}")
        print(f"Ocena:    {example['score']:.2f}")

    print("\n" + "=" * 50)
    print("5. GENEROWANIE SPÓJNYCH WYKRESÓW KDE (GĘSTOŚCI)")
    print("=" * 50)

    # Ustawienie "naukowego" stylu wykresów
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams["font.family"] = "serif"

    # Wykres 1: Rozkład Ocen (KDE)
    plt.figure(figsize=(9, 5))
    sns.kdeplot(
        data=combined_df,
        x="score",
        hue="split",
        fill=True,
        common_norm=False,
        alpha=0.3,
        palette="Set1",
    )
    # Wykres 1: Rozkład Ocen (KDE)
    plt.figure(figsize=(9, 5))
    sns.kdeplot(
        data=combined_df,
        x="score",
        hue="split",
        fill=True,
        common_norm=False,
        alpha=0.3,
        palette="Set1",
    )
    plt.title("Comparison of Similarity Score Density (Train vs Val vs Test)", pad=15)
    plt.xlabel("Human-Annotated Score (0.0 - 5.0)")
    plt.ylabel("Probability Density")
    plt.xlim(0, 5)
    plt.tight_layout()
    plt.savefig("sts_score_comparison.png", dpi=300)
    print("✅ Zapisano: sts_score_comparison.png")
    plt.close()

    # Wykres 2: Długości zdań (KDE) - spójny styl
    plt.figure(figsize=(9, 5))
    sns.kdeplot(
        data=combined_df,
        x="max_word_count",
        hue="split",
        fill=True,
        common_norm=False,
        alpha=0.3,
        palette="Set1",
    )
    plt.title("Distribution of Maximum Sentence Length in a Pair", pad=15)
    plt.xlabel("Maximum Number of Words (s1 or s2)")
    plt.ylabel("Probability Density")
    plt.xlim(0, 50)
    plt.tight_layout()
    plt.savefig("sts_length_comparison.png", dpi=300)
    print("✅ Zapisano: sts_length_comparison.png")
    plt.close()

    print("\nAnaliza zakończona sukcesem! Masz teraz pełen obraz danych.")


if __name__ == "__main__":
    analyze_sts_ultimate()
