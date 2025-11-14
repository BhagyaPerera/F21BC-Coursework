# src/__main__.py
from src.ann import ANNConfig
from src.pso import PSOConfig
from src.train.pipeline import run_pipeline

def main():


     ann_config = ANNConfig()
     pso_config = PSOConfig()
     run_pipeline(ann_config,pso_config,1)

if __name__ == "__main__":
    main()
