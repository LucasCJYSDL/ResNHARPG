# Global Convergence Guarantees for Federated Policy Gradient Methods with Adversaries

## Required environments:
- on Ubuntu 20.04
- Python 3.6
- torch 1.7.1
- tensorboard 2.6.0
- gym 0.10.9
- mujoco-py 2.0.2
- tqdm
- scikit-learn
- matplotlib
- ...

## How to run the experiments

```bash
# Enter the code folder:
cd codes
# Run the main file:
python run.py --X --agg 'Y' --env 'Z' --seed W --multiple_run 1 --attack_type 'A'
```
- X can be ResPG or ResNHARPG, corresponding to the two base algorithms: Vanilla PG and N-HARPG, respectively.
- Y can be one of [MDA, CWTM, CWMed, MeaMed, Krum, GM, SimpleMean], where the first six are as listed in Table 1 of our paper, and the last aggregator is to simply average estimates of gradients from all workers.
- Z can be CartPole-v1 or InvertedPendulum-v2.
- W is an integer for the random seed.
- A can be one of [random-action, sign-flipping, random-noise], denoting the attack type.

## For rebuttal

- New results on more challenging benchmarks are provided as the pdf file: 'new result.pdf'.
- Corresponding codes are provided in the 'rebuttal' folder.
