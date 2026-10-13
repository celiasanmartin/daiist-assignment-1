# Assignment 1 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you — write only what you
can actually explain.*

- **Name**:
- **Student ID**:
- **Email**:
- **Group**: [BBADBA 5A | BBADBA 5B | PPLEDBA 5A | BDBA 3A]

## Dataset
I used the Ames Housing dataset. 
It contains residential home sales in Ames, Iowa, from 2006 to  2010. I downloaded it from OpenML (dataset id 42165, the training set of Kaggle's "House Prices" competition) using scikit-learn's `fetch_openml` and saved it as `data/ames.csv`.

Each row represents one house sale. The dataset has 1,460 rows and 81 columns: 79 describe the house (size, overall quality, neighbourhood, year built, garage, etc.), plus an `Id` column and the target, `SalePrice`.

I picked it because housing is kind of a trending topic nowadays in Spain mainly and lately i have read a lot about it

## Business / real-life framing

**Scenario.** A retail bank uses the model as a second opinion on property valuations when assessing a mortgage application: if the model's estimate and the appraised value differ a lot, the application is flagged for review.

**Target.** The model predicts `SalePrice` directly, in dollars, rather than its logarithm. I chose this so that the RMSE is expressed in dollars, which is easy for the bank to interpret. The trade-off is that `SalePrice` is right-skewed (skewness of about 1.9), so a few expensive houses can pull the linear model; I discuss this in the limitations.

**Split.** Because the bank would value houses sold after the model is built, I split the data by sale year instead of randomly: the model is trained on earlier years and tested on the most recent ones. A random split would be misleading because test houses would be sold in the same years as many training houses, so the model would be evaluated on a market it has already seen, while in practice the bank would value houses sold after the model was built. This would make the reported error lower than the real one.

**Metric.** I assume that overvaluing and undervaluing a house cost the bank a similar amount [your reason: for example, losing customers vs. default risk], so I do not need an asymmetric error. I use RMSE as the main metric because it penalises large errors more heavily, and large valuation mistakes are the ones that matter most for this decisionnnnn

## Data preparation & feature engineering

*What you engineered and why, and any data-quality decisions you made along
the way — e.g. "segment X had defective data, so I excluded it and used a
population-average default for scope Y at inference time; the impact of
that choice is Z."*

## Modeling: three implementations, one model

*Which model (linear or logistic regression) and why. A results table
comparing scikit-learn, the manual PyTorch loop, and the standard
torch.nn.Module/torch.optim workflow, on the same test set, against the
naive baseline. Do the three agree? If not, why not?*

## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data — not generic hedging.*

## Generative AI use disclosure

*Per the syllabus AI Policy: what you used and how, or "no AI content used."*
