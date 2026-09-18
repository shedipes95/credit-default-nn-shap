# Data

No data is committed to this repo. The Home Credit Default Risk dataset is public
but large, and it has its own competition terms, so you download it yourself.

## 1. Get the raw files

The dataset lives on Kaggle:
<https://www.kaggle.com/competitions/home-credit-default-risk/data>

With the Kaggle CLI:

```bash
pip install kaggle
kaggle competitions download -c home-credit-default-risk -p data/raw
unzip 'data/raw/*.zip' -d data/raw
```

You need a Kaggle account and you have to accept the competition rules once,
in the browser, before the download works.

The raw download is seven tables. `application_train.csv` is the main one —
307,511 applications, one row per loan, with `TARGET = 1` meaning the applicant
had payment difficulties. The other six (`bureau`, `bureau_balance`,
`previous_application`, `POS_CASH_balance`, `installments_payments`,
`credit_card_balance`) are one-to-many histories that have to be aggregated back
to applicant level before a model can use them.

## 2. Build the model-ready table

This repo expects a single CSV at `data/train_neural_network_ready.csv` with:

- `SK_ID_CURR` — applicant id, dropped before training
- 499 numeric feature columns — aggregated across the seven tables, imputed,
  skew-corrected and standardised
- `TARGET` — 0 or 1

so 307,511 rows by 501 columns.

**The feature-engineering step that produces this file is not in this repo.** It
was a separate part of the coursework. If you are rebuilding it from scratch, the
shape you are aiming for is: aggregate each secondary table to `SK_ID_CURR` with
mean / min / max / sum / var, add cross-table ratio features, impute, drop
columns that are almost entirely missing, log-transform the heavily skewed
positive columns, and standardise everything (the network needs scaled inputs).

Any table with that structure will run — the code only assumes numeric features
and a binary `TARGET`. If your feature count differs from 499, nothing breaks;
the input dimension is read from the data.
