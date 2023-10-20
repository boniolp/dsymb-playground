about_text = """ 
## Fast interactive exploration of multivariate time series datasets

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

## Contributors

* [Sylvain W. Combettes](https://sylvaincom.github.io/) (Centre Borelli, ENS Paris-Saclay)
* [Paul Boniol](https://boniolp.github.io/) (Centre Borelli, ENS Paris-Saclay)
* [Charles Truong](https://charles.doffy.net/) (Centre Borelli, ENS Paris-Saclay)
* [Laurent Oudre](http://www.laurentoudre.fr/) (Centre Borelli, ENS Paris-Saclay)
"""


data_JIGSAW = f"""
The data series are recorded from the DaVinciSurgicalSystem.
The multivariate data series are composed of ***76 dimensions***.
Each dimension corresponds to a sensor (with an acquisition rate of 30 Hz).
The sensors are divided into four groups: patient-side manipulators
(left and right PSMs), and left and right master tool
manipulators (left and right MTMs).
Each group contains 19 sensors.
These sensors are: 3 variables for the Cartesian position of the manipulator,
9 variables for the rotation matrix, 6 variables for the linear and angular
velocity of the manipulator, and 1 variable for the gripper angle.

A preprocessed version of the dataset can be found
[here](https://kiwi.cmla.ens-cachan.fr/index.php/s/ctEdTsz6sxPBxxX). 
Note that these time series can be uploaded in the `Explore` tab.
"""

Baseline_desc = """
In this experiment, we compare $d_{symb}$ with 9 existing distance measures. 
These distances are specifically dedicated to time series.
Their implementation can be found in the
[aeon](https://www.aeon-toolkit.org/en/latest/api_reference/distances.html) Python package.
All these distances are elastic as they can compare time series of different lengths.

- [DTW (Dynamic Time-Warping)](https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.dtw_distance.html#aeon.distances.dtw_distance):
DTW is the most popular elastic distance.
DTW can perform warping, meaning one-to-many alignment between samples of two
time series.
- [DDTW (Derivative DTW)](https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.ddtw_distance.html):
DDTW applies DTW, not directly on the raw signals, but on their first derivative
in order to prevent unnatural warpings when there is variability in the signals.
- [WDTW (Weighted DTW)](https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.wdtw_distance.html#aeon.distances.wdtw_distance):
Compated to DTW, WDTW aims at avoiding large warpings by penalizing them using a non-linear multiplicative weight.
- [WDDTW (Weighted Derivative DTW)](https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.wddtw_distance.html#aeon.distances.wddtw_distance):
WDDTW combines DDTW and WDTW.
- [LCSS (Longest Common SubSequence)](https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.lcss_distance.html#aeon.distances.lcss_distance):
Initially, LCSS is an edit distance defined on strings, allowing only insertions
and deletions of characters.
It measures the length of the longest pairing of characters that can be between
both strings, so that the pairings respect the order of the letters.
It has been extended to the real-valued case thanks to a threshold.
- [EDR (Edit Distance on Real sequence)](https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.edr_distance.html#aeon.distances.edr_distance):
EDR is based on the Levenshtein distance on strings that allows substitution,
insertions and deletions.
Rather than using a delete operation, EDR considers a deletion in a time series
as a special symbol in another series.
- [ERP (Edit distance with Real Penalty)](https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.erp_distance.html#aeon.distances.erp_distance):
ERP is close to EDR, except that ERP called the delete operation a gap element
and has penalty parameter.
- [MSM (Move-Split-Merge)](https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.msm_distance.html#aeon.distances.msm_distance):
MSM is inspired by edit distances on strings.
MSM states that, contrary to ERP, it has the particularity of being invariant to translations.
It allows three operations: move (substitution), split (duplication), and
merge (contraction).
- [TWE (Time Warp Edit distances)](https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.twe_distance.html#aeon.distances.twe_distance):
TWE is based on the edit distance on strings, however it has no straightforward
equivalent on strings.
Indeed, TWE combines (non-elastic) $L_p$ norms with the (elastic) edit distance.
- [$d_{symb}$](https://github.com/sylvaincom/d-symb): $d_{symb}$ first symbolizes
each multivariate time series into a symbolic sequence, then uses
a distance measure defined on strings, inspired from the general edit distance,
to compare the obtained symbolic sequences.
"""

compare_text_1 = """

## Benchmark $d_{symb}$ distance

We now illustrate the relevance of the $d_{symb}$ distance measure, compared
to other existing ones, on a real-world use case.
We apply our benchmark to the
[JIGSAWS dataset](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/) 
with the goal of identifying surgeons' gestures based on kinematic time series.
These signals are generated when using robotic arms and grippers to perform surgical tasks.
All results are pre-computed (in order to save you the computing time).
"""

compare_text_2 = """

### Explore the distance measures' results

In this dataset, we consider two surgical gestures: ***Knot Tying*** (39 multivariate time series) 
and ***Needle Passing*** (40 multivariate time series).
The goal is to cluster (using an agglomerative clustering approach with complete
linkage) and identify these two gestures, each time for several distance measures.
In the following, we display the pairwise distance matrices corresponding to several
distance measures, as well as the clustering performances and the execution time (in seconds).
In total, 9 distance measures are used in the benchmark, apart from $d_{symb}$.
Distance measures are described in the `Baselines description` tab.
"""
