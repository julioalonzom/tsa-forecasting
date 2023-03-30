# Time Series Forecasting


[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/julioalonzom/tsaf/main.svg)](https://results.pre-commit.ci/latest/github/julioalonzom/tsaf/main)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This project focuses on using time series analysis methods, in particular Holt-Winters and ARIMA, to forecast unemployment data from the Federal Reserve Economic Data (FRED). The purpose of this project is to explore how different time series models can be applied to real-world data, and to compare the performance of these models on the unemployment data.

## Usage

To get started, clone this repository

```console
$ git clone https://github.com/julioalonzom/tsa-forecasting
```

Then, create and activate the environment with

```console
$ conda/mamba env create
$ conda activate tsaf
```

To build the project, type

```console
$ pytask
```

## Structure of the project

- The two models are fitted on the data.
- Forecasted values are generated.
- Different accuracy measures are produced to evaluate performance of both methods.
- The results are plotted and a table with the measures is produced.

## Credits

This project was created with [cookiecutter](https://github.com/audreyr/cookiecutter)
and the
[econ-project-templates](https://github.com/OpenSourceEconomics/econ-project-templates).
