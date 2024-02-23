<p align="center">
<img width="230" src="./figures/Symbol_logo.png"/>
</p>

<h1 align="center">d_{symb} playground</h1>
<h2 align="center">A fast interactive exploration of multivariate time series datasets</h2>

<div align="center">
<p>
<img alt="GitHub" src="https://img.shields.io/github/license/boniolp/dsymb-playground"> <img alt="GitHub issues" src="https://img.shields.io/github/issues/boniolp/dsymb-playground">
</p>
</div>

<p align="center"><a href="https://dsymb-playground.streamlit.app/">Try our demo</a></p>
<p align="center">
<img width="500" src="./figures/demo_capture.gif"/>
</p>

$d_{symb}$ playground is a Python-based web interactive tool to interpet and
compare large multivariate time series datasets.
It is based on a novel symbolic representation, called $d_{symb}$, for
multivariate time series.
$d_{symb}$ allows to visualize a dataset of multivariate time series with
a single glance, thus to quickly gain insights on your data.
$d_{symb}$ also comes with a compatible distance measure to compare the
obtained symbolic sequences.
Apart from its relevance on data mining tasks, this distance measure is also
fast.
Indeed, comparing a dataset of 80 time series (with 80 dimensions
and 5,000 timestamps) requires 20 seconds instead of 2,000 seconds for DTW-based
analysis.

### Reference

This repository contains the code that supports the following publication on the $d_{symb}$ playground.

Demo paper of the $d_{symb}$ playground [[paper](https://icde2024.github.io/demos.html) / [PDF](http://www.laurentoudre.fr/publis/dsymb_demo.pdf) / [Streamlit app](https://dsymb-playground.streamlit.app/)]:
> S. W. Combettes, P. Boniol, C. Truong, and L. Oudre. d_{symb} playground: an interactive tool to explore large multivariate time series datasets. In _Proceedings of the International Conference on Data Engineering (ICDE)_ (to appear), Utrecht, Netherlands, 2024.

```bibtex
@inproceedings{2024_combettes_dsymb_playground_icde,
  title={d_{symb} playground: an interactive tool to explore large multivariate time series datasets},
  author={Sylvain W. Combettes and Paul Boniol and Charles Truong and Laurent Oudre},
  booktitle={Proceedings of the International Conference on Data Engineering (ICDE) (to appear)},
  year={2024},
  location={Utrecht, Netherlands},
}
```

Method paper of $d_{symb}$ [[paper](https://ieeexplore.ieee.org/abstract/document/10411636) / [PDF](http://www.laurentoudre.fr/publis/ICDM2023.pdf) / [code](https://github.com/sylvaincom/d-symb)]:
> S. W. Combettes, C. Truong, and L. Oudre. An Interpretable Distance Measure for Multivariate Non-Stationary Physiological Signals. In _Proceedings of the International Conference on Data Mining Workshops (ICDMW)_, Shanghai, China, 2023.

```bibtex
@inproceedings{2023_combettes_dsymb_icdm,
  author={Combettes, Sylvain W. and Truong, Charles and Oudre, Laurent},
  booktitle={2023 IEEE International Conference on Data Mining Workshops (ICDMW)}, 
  title={An Interpretable Distance Measure for Multivariate Non-Stationary Physiological Signals}, 
  year={2023},
  pages={533-539},
  doi={10.1109/ICDMW60847.2023.00076},
  location={Shanghai, China},
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
git clone https://github.com/boniolp/dsymb-playground.git
cd dsymb-playground/
```

**Step 2:** Create and activate a `conda` environment and install the dependencies.

```bash
conda env create -n dsymb-playground python=3.9
conda activate dsymb-playground
pip install -r requirements.txt
```

**Step 3:** You can use our tool in two different ways: 

- Access online: https://dsymb-playground.streamlit.app/
- Run locally (preferable for large time series datasets). To do so, run the following command:

```bash
streamlit run app.py
```

You can then open the app using your web browser. You can upload any kind of time series (one file per time series) with the shape `(n_timestamps, n_dims)`.
A preprocessed version of the dataset [JIGSAWS dataset](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/) can be found [here](https://kiwi.cmla.ens-cachan.fr/index.php/s/ctEdTsz6sxPBxxX).

## Acknowledgments

Sylvain W. Combettes is supported by the IDAML chair (ENS Paris-Saclay) and UDOPIA (ANR-20-THIA-0013-01).
Charles Truong is funded by the PhLAMES chair (ENS Paris-Saclay).
Part of the computations has been executed on Atos Edge computer, funded by the IDAML chair (ENS Paris-Saclay).

<p align="center">
<img width="700" src="./figures/cebo_logos.png"/>
</p>
