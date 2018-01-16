# BetaBarrelNN

Example of running code:
python load_data_kfold_gmp_2.0_fixedweight.py ${NFLF} ${NF} ${NL} ${HD} ${dataset}

where dataset: TMBB2 or Boctopus
NFLF: Number of first layer filters
NF: number of filters in deeper layers
NL: number of deeper layers

Example of the best configuration for TMBB2 dataset

python load_data_kfold_gmp_2.0_fixedweight.py 128 512 2 32 TMBB2

output:
AVERAGE MCC, sensitivity, specificity: 0.958429790279 0.020635994669 0.955535277493 0.0295890139942 0.99682138944 0.0019037968686

