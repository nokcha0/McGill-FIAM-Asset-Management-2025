import os, sys, subprocess

if __name__ == "__main__":
    print("Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    print("Dependencies installed\n")

    for d in ["output/data", "output/predictions", "output/portfolio"]:
        os.makedirs(d, exist_ok=True)

    print("=== Pipeline: LightGBM + NLP Training ===\n")

    print("Step 1: Build lead-4 dataset")
    #subprocess.run(["python", "lead_ratios.py"], check=True)

    print("Step 2: NLP engine (text feature extraction)")
    subprocess.run(["python", "nlp_engine.py"], check=True)

    print("Step 3: Train LightGBM + NLP fusion model")
    subprocess.run(["python", "train_dualengine.py"], check=True)

    print("Step 4: Portfolio analysis")
    subprocess.run(["python", "portfolio_analysis.py"], check=True)

    print("\nPipeline completed. Results are saved in /output/")
    print("   ├── output/data/          (datasets)")
    print("   ├── output/predictions/   (model predictions)")
    print("   └── output/portfolio/     (portfolio analysis results)\n")
