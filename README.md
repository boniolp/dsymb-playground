# Symbol

<p align="center">
<img width="130" src="./figures/Symbol_logo.png"/>
</p>

<h1 align="center">Symbol</h1>
<h2 align="center">A fast interactive vizualisation of multivariate time series dataset</h2>

<div align="center">
<p>
<img alt="GitHub" src="https://img.shields.io/github/license/boniolp/symbol"> <img alt="GitHub issues" src="https://img.shields.io/github/issues/boniolp/symbol">
</p>
</div>

<p align="center"><a href="https://symbol.streamlit.app/">Try our demo</a></p>
<p align="center">
<img width="500" src="./figures/demo_capture.gif"/>
</p>

Symbol is a python-based web interactive tool to visualize, navigate and explore large multivariate time series datasets. It is based on a new symbolic reprensetaiton, **dsymb**, for multivariate time series. With our tool, exploring a dataset of 80 time series (with 80 dimnensions and 5000 timestamps) require 20 seconds, instead of 2000 seconds for DTW-based analysis.

### Reference

> "dsymb: "<br/>
> Authors<br/>
> Proceedings of XXX, pages XXX-XXX, 2023<br/>

```bibtex
@article{symbol,
  title={},
  author={},
  journal={},
  volume={},
  number={},
  pages={},
  year={},
  publisher={}
}
```

## Contributors

* Paul Boniol (ENS Paris Saclay)
* Sylvain Combettes (ENS Paris Saclay)
* Charles Truong (ENS Paris Saclay)
* Laurent Oudre (ENS Paris Saclay)


## Usage

**Step 1:** Clone this repository using `git` and change into its root directory.

```bash
git clone https://github.com/boniolp/MSAD.git
cd symbol/
```

**Step 2:** Create and activate a `conda` environment and install the dependencies.

```bash
conda env create -n symbol python=3.9
conda activate symbol
pip install -r requirements.txt
```

**Step 3:** You can use our tool in two differnt way: (i) Access online: https://symbol.streamlit.app/ (ii) Run locally (preferable for large time series datasets). To do so, run the following commands:

```bash
streamlit run app.py
```

You can then open the app using your web browser. You can upload any kind of time series (one file per time series) with the shape (n_timestamps,n_dims).



