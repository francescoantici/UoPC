# UoPC 

UoPC is a user-based online framework for predicting job power consumption in HPC systems. UoPC leverages ML-based predictive models tailored for individual users, eliminating the need for voluminous data and training. It offers a user-friendly Python implementation suitable for both end-user usage and integration into workload management systems.
It achieves only a 10% prediction error on more than 700k job data extracted from Supercomputer Fugaku, with minimal overhead on the system operations. 

Our approach can predict average and maximum job power consumption, and it can be used to estimate the whole system power consumption with an error of only 4\%.
By employing a k-nearest neighbours (KNN) prediction model augmented with Natural Language Processing, UoPC streamlines prediction processes for newly submitted jobs. It requires only limited historical data, making it practical for diverse high-performance computing environments and workloads.

## How to use


