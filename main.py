import os, sys, subprocess

TEMPLATE_DIR = "/teamspace/uploads"

if __name__ == "__main__":
    os.environ["TEMPLATE_DIR"] = TEMPLATE_DIR

    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"
    ], check=True)

    os.makedirs("output/data", exist_ok=True)
    os.makedirs("output/predictions", exist_ok=True)
    os.makedirs("output/portfolio", exist_ok=True)

    print("=== Step 1: Build lead-4 dataset ===")
    subprocess.run([sys.executable, "lead_ratios.py"], check=True)

    print("=== Step 2: Build reports (corporate filings) ===")
    subprocess.run([sys.executable, "build_reports.py"], check=True)

    print("=== Step 3: NLP engine (text feature extraction) ===")
    subprocess.run([sys.executable, "nlp_engine.py"], check=True)

    print("=== Step 4: Train LightGBM + NLP fusion model ===")
    subprocess.run([sys.executable, "train_dualengine.py"], check=True)

    print("=== Step 5: Portfolio analysis ===")
    subprocess.run([sys.executable, "portfolio_analysis.py"], check=True)

    print("\nCompleted. Results are saved in /output/")
