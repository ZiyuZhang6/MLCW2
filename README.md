# TypiClust

TPC-RP algorithm is implemented in tpc_sample_seletion.py, after finishing sampling, it will take some time to visualise selected samples and clusters.
Improved TPC-RP algorithm is implemented in modified_tpc_sample_seletion.py.

## Model Training

All three framework are implemented in fully_supervised.py, self_supervised.py, semi_supervised.py and used improved TPC-RP algorithm.
flexmatch_utils.py and train_utils.py are taken from TorchSSL repository for implementing flexmatch.
Fully-Supervised framework and Fully-Supervised with Self-Supervised Embbeding surpass the result in paper.
I couldn't finish training Self-Supervised framework as running self_supervised.py will take a very long time due to computational expensive,
so it's unguaranteed that the Self-Supervised framework would reproduce paper's result.
