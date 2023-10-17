about_text = f""" 
## A fast interactive exploration of multivariate time series datasets
Symbol is a Python-based web interactive tool to visualize, navigate, and explore 
large multivariate time series datasets. It is based on a new symbolic representation, 
**dsymb**, for multivariate time series. With our tool, exploring a dataset of 80 time 
series (with 80 dimensions and 5,000 timestamps) requires 20 seconds instead of 2,000 
seconds for DTW-based analysis.

## Contributors

* Paul Boniol (ENS Paris-Saclay)
* Sylvain W. Combettes (ENS Paris-Saclay)
* Charles Truong (ENS Paris-Saclay)
* Laurent Oudre (ENS Paris-Saclay)
"""


data_JIGSAW=f"""
The data series are recorded from the
DaVinciSurgicalSystem. The multivariate data series are
composed of ***76 dimensions***. Each dimension corresponds to a sensor
(with an acquisition rate of 30 Hz). The sensors are divided into
four groups: patient-side manipulators (left and right PSMs), and left and right master tool
manipulators (left and right MTMs). Each group contains 19 sensors. These sensors are: 3
variables for the Cartesian position of the manipulator, 9 variables
for the rotation matrix, 6 variables for the linear and angular
velocity of the manipulator, and 1 variable for the gripper angle.

A preprocesssed version of the dataset can be found [here](https://kiwi.cmla.ens-cachan.fr/index.php/s/ctEdTsz6sxPBxxX). 
These time series can be uploaded in the `Explore` tab.
"""

Baseline_desc =f"""
In this experiment, we compare ***d_symb*** with 9 distance measures. 
These distances are specifically dedicated to time series. Their implementation 
can be found in the [aeon](https://www.aeon-toolkit.org/en/latest/api_reference/distances.html) Python package.

- [DTW](https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.dtw_distance.html#aeon.distances.dtw_distance): Dynamic Time-Warping distance is the most widely used distance for time series. It compensates for misalignment issues by considering the best match for each pair of time series points.
- [DDTW](https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.ddtw_distance.html): Derivative DTW distance is similar to DTW distance, but consider the first order derivative of the original time series.
- [WDTW](https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.wdtw_distance.html#aeon.distances.wdtw_distance): Weighted DTW distance is similar to DTW distance but weights nearer neighbors more heavily depending on the time difference.
- [WDDTW](https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.wddtw_distance.html#aeon.distances.wddtw_distance): Weighted Derivative DTW distance is computing the WDTW on the first order derivative of the time series.
- [MSM](https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.msm_distance.html#aeon.distances.msm_distance): Move-Split-Merge computes similarity by measuring the cost to transform one series into another using a predefined set of operations.
- [LCSS](https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.lcss_distance.html#aeon.distances.lcss_distance): LCSS distance is using the Longest common subsequence of the two time series to measure their similarity.
- [ERP](https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.erp_distance.html#aeon.distances.erp_distance): Edit Real Penalty distance is similar to DTW distance, but allows gaps of points with no match that are then penalized based on their distance to a predefined value.
- [EDR](https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.edr_distance.html#aeon.distances.edr_distance): Edit Distance on Real Sequences computes the minimum number of points that have to be deleted from the time series such that the sum of the distance between the remaining time series points is smaller than a predefined threshold.
- [TWE](https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.twe_distance.html#aeon.distances.twe_distance): Time Warp Edit distance is combining both DTW and Edit distance.

- [d_symb](): TODO
"""

compare_text_1 = f"""

## Compare with other distance

We now illustrate the relevance of d_symb on a real-world
use case. In this use case, we use distance measures on the [JIGSAWS dataset](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/) 
to identify surgeons' gestures based on kinematic
time series when using robotic arms and grippers to perform surgical tasks.


"""

compare_text_2 = f"""

### Explore the results

In this dataset, we consider two surgical gestures: ***Knot Tying*** (39 multivariate time series) 
and ***Needle Passing*** (40 multivariate time series). The objective is to use different distance measures
to cluster (using an agglomerative clustering approach with complete linkage) and identify these two gestures.
In the following section,
the similarity matrices corresponding to several distance measures, as well as the clustering performances 
(using 8 evaluation measures) and the execution time (in seconds) can be visualized.


"""