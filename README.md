# beta-VAE

## Latent traversal

Latent traversal replication using the 2D shapes VAE architecture from A.2 Table 1 of beta-VAE by Higgins et al. 
(https://openreview.net/references/pdf?id=Sy2fzU9gl). 

Results after 40 epochs:

<img src="https://github.com/katalinic/betaVAE/blob/master/latent_traversal40.png" width="400">

## Capacity

capacity.py is an attempt at reproducing the Figure 3 (top left) of Understanding disentangling in $\beta$-VAE by Burgess et al. (https://arxiv.org/pdf/1804.03599.pdf). The preset hyperparameters definitely show an increase in latent factor capacity in stages as presented in the paper (i.e. first towards x and y, then scale etc.), though not at the same rate.
