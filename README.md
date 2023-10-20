<p align="center">
<img width="230" src="./figures/Symbol_logo.png"/>
</p>

<h1 align="center">d_symb playground</h1>
<h2 align="center">A fast interactive exploration of multivariate time series datasets</h2>

<div align="center">
<p>
<img alt="GitHub" src="https://img.shields.io/github/license/boniolp/symbol"> <img alt="GitHub issues" src="https://img.shields.io/github/issues/boniolp/symbol">
</p>
</div>

<p align="center"><a href="https://symbol.streamlit.app/">Try our demo</a></p>
<p align="center">
<img width="500" src="./figures/demo_capture.gif"/>
</p>

$d_{symb}$ playground is a Python-based web interactive tool to interpet and
compare large multivariate time series data sets.
It is based on a novel symbolic representation, called $d_{symb}$, for
multivariate time series.
$d_{symb}$ allows to visualize a data set of multivariate time series with
a single glance, thus to quickly gain insights on your data.
$d_{symb}$ also comes with a compatible distance measure to compare the
obtained symbolic sequences.
Apart from its relevance on data mining tasks, this distance measure is also
fast.
Indeed, comparing a dataset of 80 time series (with 80 dimensions
and 5,000 timestamps) requires 20 seconds instead of 2,000 seconds for DTW-based
analysis.

### Reference

> "An Interpretable Distance Measure for Multivariate Non-Stationary Physiological Signals"<br/>
> Authors: Sylvain W. Combettes, Charles Truong and Laurent Oudre<br/>
> _Proceedings of the International Conference on Data Mining (AI4TS Workshop) (to appear)_, Shanghai, China, 2023.<br/>

```bibtex
@inproceedings{2023_combettes_dsymb_icdm,
  title={An Interpretable Distance Measure for Multivariate Non-Stationary Physiological Signals},
  author={Sylvain W. Combettes and Charles Truong and Laurent Oudre},
  booktitle={Proceedings of the International Conference on Data Mining (AI4TS Workshop) (to appear)},
  year={2023},
  url={http://www.laurentoudre.fr/publis/ICDM2023.pdf}
}
```

## Contributors

* [Sylvain W. Combettes](https://sylvaincom.github.io/) (Centre Borelli, ENS Paris-Saclay)
* [Paul Boniol](https://boniolp.github.io/) (Centre Borelli, ENS Paris-Saclay)
* [Charles Truong](https://charles.doffy.net/) (Centre Borelli, ENS Paris-Saclay)
* [Laurent Oudre](http://www.laurentoudre.fr/) (Centre Borelli, ENS Paris-Saclay)


## Usage

**Step 1:** Clone this repository using `git` and change into its root directory.

```bash
git clone https://github.com/boniolp/symbol.git
cd symbol/
```

**Step 2:** Create and activate a `conda` environment and install the dependencies.

```bash
conda env create -n symbol python=3.9
conda activate symbol
pip install -r requirements.txt
```

**Step 3:** You can use our tool in two different ways: 

- Access online: https://symbol.streamlit.app/
- Run locally (preferable for large time series datasets). To do so, run the following command:

```bash
streamlit run app.py
```

You can then open the app using your web browser. You can upload any kind of time series (one file per time series) with the shape `(n_timestamps, n_dims)`.
A preprocessed version of the dataset [JIGSAWS dataset](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/) can be found [here](https://kiwi.cmla.ens-cachan.fr/index.php/s/ctEdTsz6sxPBxxX).

<p align="center">
<img width="700" src="./figures/cebo_logos.png"/>
</p>
