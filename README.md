# Transfer-Learning-Based Autotuning Using Gaussian Copula

Authors: Thomas Randall, Jaehoon Koo, Brice Videau, Michael Kruse, Xingfu Wu, Paul Hovland, Mary Hall, Rong Ge, Prasanna Balaprakash

This repository is provided for transparency and ease-of-replication for our ICS'23 paper, ["Transfer-Learning-Based Autotuning Using Gaussian Copula"](https://dl.acm.org/doi/10.1145/3577193.3593712).

## Contact: Thomas Randall (tlranda@clemson.edu)
### May 15, 2023

## License: [BSD 2-Clause License](LICENSE.md)

# Repository Organization

* [Benchmarks](Benchmarks): Provides source code and tuning space definitions (problem.py) for all benchmarks.
Organized by benchmark suite, then benchmark.
* [ConditionalSampling](ConditionalSampling): Greater depth presentation and exploration of mathematics and mechanisms of Conditional Sampling with Gaussian Copulas.
Provided as an additional resource referenced by the paper.
* [Data](Data): Raw experimental data provided for transparency.
Files are organized by benchmark suite, benchmark, then each tuning technique.
* [GC\_TLA](GC_TLA): Our technique as well as all scripts necessary to replicate our experiments and analyses.

# Setup and Installation

* After cloning, run `python3 softLinkDataDirs.py` to establish symbolic links between respective entries of the Benchmarks and Data directories.

# Related Materials

Slides and recorded talks may not be uploaded to GitHub, but relevant URLs will be made accessible here.

# Paper Copyright

Publication rights licensed to ACM.
ACM acknowledges that this contribution was authored or co-authored by an employee, contractor or affiliate of the United States government.
As such, the Government retains a nonexclusive, royalty-free right to publish or reproduce this article, or to allow others to do so, for Government purposes only.
ICS'23, June 21-23, 2023, Orlando, FL, USA
© 2023 Copyright held by the owner/author(s). Publication rights licensed to ACM
ACM ISBN 979-8-4007-0056-9/23/06
https://doi.org/10.1145/3577193.3593712

# Funding

This research was partially supported by the Exascale Computing Project (17-SC-20-SC), a collaborative effort of the U.S. Department of Energy Office of Science and the National Nuclear Security Administration, and by U.S. National Science Foundation under Grant CCF-1942182.
This material is based upon work supported by the U.S. Department of Energy, Office of Science, under contract number DE-AC02-06CH11357.

