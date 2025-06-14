# This is a repository for the masters thesis:
- Analysis of the benefits and limitations of seismic attribute classification using a 3D facies model

# Author
- Thomas Gingstad

This thesis was done with the supervision of Nestor Cardozo and Lothar Schulte

# Abstract
This thesis aims to construct a synthetic 3D seismic model and explore the application of machine learning for facies classification 
using seismic attributes, post-stack inversion, AVO inversion and rock moduli based on this model. The machine learning method used is 
the Random Forests algorithm. The synthetic 3D model is based on a provided synthetic 3D facies model that is considered as the ground 
truth for testing and validation purposes. Characteristic mineral components and porosity distribution were assigned to each facies, 
allowing the calculation of P- and S- impedance and density. These rock properties are regarded as ground truth. They enable the 
derivation of post-stack and angle stack seismic inversion cubes, which were in turn used to estimate the facies and compared them to 
the ground truth facies.

Classification results, based on three training wells, showed that seismic attributes alone achieve a classification accuracy of 57%, 
with relative acoustic impedance identified as the most influential attribute. Post-stack inversion improved the accuracy to 61%, while 
the AVO inversion achieved an accuracy of 65%. The use of rock moduli yielded a comparable result of 64%. All input types exhibit 
limitation due to seismic resolution constrains in thin beds and overlapping impedance values of different facies, particularly between 
the sand and coarse sand facies. These limitations were mitigated by introducing a simplified facies model, where all sand facies were 
merged into a single facies class.

A comparison between the Random Forest classifier and a classical Bayesian Framework, using ground truth impedance values, showed that 
Random Forest achieved slightly higher accuracy (81%) compared to the Bayesian approach (75%). This highlights the potential for Random 
Forest algorithm for facies classification.

# Required Python libraries:

SEGYIO	1.9.22
SEGYSAK	0.5.4
NUMPY	1.23.5
PANDAS	1.3.4
MATPLOTLIB	3.4.3
SEABORN	0.11.2
PLOTLY	5.5.0
SKLEARN	0.24.2
TENSORFLOW	2.12.0
