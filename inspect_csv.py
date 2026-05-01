import pandas as pd

FILE = "lichess_puzzles_reduced.csv"

def main():
    print("📄 File:", FILE)
    print("-" * 50)

    # 1. Column names
    df_head = pd.read_csv(FILE, nrows=5)
    print("🧱 Columns:")
    print(list(df_head.columns))
    print()

    # 2. Sample rows
    print("🔍 Sample rows:")
    print(df_head)
    print()

    # 3. Count total rows (memory efficient)
    print("📊 Counting total rows...")
    with open(FILE, "r") as f:
        total_rows = sum(1 for _ in f) - 1  # minus header
    print("Total rows:", total_rows)
    print()

    # 4. Data types
    print("📌 Data types:")
    print(df_head.dtypes)
    print()

    # 5. Basic stats (only numeric columns)
    print("📈 Basic stats:")
    print(df_head.describe())

if __name__ == "__main__":
    main()  