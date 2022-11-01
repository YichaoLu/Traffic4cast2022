# Traffic4cast2022

## Data Preparation

After cloning this [repo](git@github.com:YichaoLu/Traffic4cast2022.git), download and extract [cache](https://drive.google.com/file/d/1N3bhlPJuLjW-gTwZ3K1Gjt75yrw12YnU/view?usp=sharing) and [output](https://drive.google.com/file/d/11mgqA4dfwvxpwsq0b59tjBmQzOBd0Rre/view?usp=sharing) to the root of the cloned repo.

## Usage

To create a submission for the [Core Challenge](https://www.iarai.ac.at/traffic4cast/challenge/#core-leaderboard), run

```
python preprocess_core_xgb.py --city london
python preprocess_core_xgb.py --city madrid
python preprocess_core_xgb.py --city melbourne
python inference_core_xgb.py --city london
python inference_core_xgb.py --city madrid
python inference_core_xgb.py --city melbourne
python write_submission_core.py
```

This creates a submission file named `cc_submission.zip` under the `output` folder.

To create a submission for the [Extended Challenge](https://www.iarai.ac.at/traffic4cast/challenge/#extended-leaderboard), run

```
python preprocess_extended_lgb.py --city london
python preprocess_extended_lgb.py --city madrid
python preprocess_extended_lgb.py --city melbourne
python inference_extended_lgb.py --city london
python inference_extended_lgb.py --city madrid
python inference_extended_lgb.py --city melbourne
python write_submission_extended.py
```

This creates a submission file named `eta_submission.zip` under the `output` folder.
