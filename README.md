# C-CHVAE

## Set up
Counterfactual explanations can be obtained by identifying the smallest change made to an input vector to influence a prediction in a positive way. Classic examples can be found in credit scoring or health contexts where one tries to change a classifier's decision from ’loan rejected’ to ’awarded’ or from ’high risk of cardiovascular disease’ to ’low risk’. Our approach ensures that the produced counterfactuals are **proximate** (i.e., not local outliers) and **connected** to regions with substantial data density (i.e., close to correctly classified observations), two requirements known as **counterfactual faithfulness**.

## Intution
We suggest embedding counterfactual search into a data density approximator, here a variational autoencoder (VAE). The idea is to use the VAE as a search device to find counterfactuals that are proximate and connected to the input data. Given the original tabular data, the encoder specifies a lower dimensional, realvalued and dense representation of that data, z. Therefore, it is the encoder that determines which low-dimensional neighbourhood we should look to for potential counterfactuals. Next, we perturb the low dimensional data representation, z + $\delta$, and feed the perturbed representation into the decoder. For small perturbations the decoder gives a potential counterfactual by reconstructing the input data from the perturbed representation. This counterfactualmis likely to occur. Next, the potential counterfactual is passed to the pretrained classifier, which we ask whether the prediction was altered. 

## On running the (C-)HVAE
To run the HVAE you have to predefine each input's type: you can choose one of the following: *real* (for inputs defined on the real line), *pos* (for inputs defined on positive part of R), *count* (for count inputs), *cat* (for categorical inputs) and *ordinal* (for ordinal inputs). To see an example, have a look at the *types*.csv files within the *data* folder.


## Bibtex 
```
@inproceedings{pawelczyk_learning2019,
author = {Pawelczyk, Martin and Broelemann, Klaus and Kasneci, Gjergji},
title = {Learning Model-Agnostic Counterfactual Explanations for Tabular Data},
year = {2020},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
booktitle = {Proceedings of The Web Conference 2020},
pages = {3126–3132},
numpages = {7},
keywords = {Transparency, Counterfactual explanations, Interpretability},
location = {Taipei, Taiwan},
series = {WWW '20}
}
```

