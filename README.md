### Original Repo: https://github.com/dongji1111/TheFacT

### Run scripts
#### Train model from zero:
` python test_train.py --train_file ../data/yelp_train.txt --test_file ../data/yelp_test.txt --path ../yelp_bpr_1000/ --num_dim 200 --num_run 1 --start 90 --lambda_bpr 1000 --save_rate 1`

#### Train model from previous one:
` python test_train.py --train_file ../data/yelp_train.txt --test_file ../data/yelp_test.txt --path ../yelp_bpr_1000/ --num_dim 200 --num_run 1 --lambda_bpr 1000 --save_rate 1 `

#### Provide recommendation and explanation:
` python newtest.py --path ../new_yelp_result/ --train_file ../data/yelp_train.txt --test_file ../data/yelp_test.txt --version 1 `

```
@inproceedings{,
  title={The FacT: Taming Latent Factor Models for Explainability with Factorization Trees},
  author={Tao, Yiyi and Jia, Yiling and Wang, Nan and Wang, Hongning},
  booktitle={The 42nd International ACM SIGIR Conference on Research \& Development in Information Retrieval},
  year={2019},
  organization={ACM}
}
```
