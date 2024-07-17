# MEG-Analysis-Pipeline
This repository contains code for processing MEG data, such as performing source reconstruction, and conducting Temporal Response Function (TRF) analysis. It includes example scripts demonstrating how to use the provided code.
The methods and tools used are implemented in python, using mne-python.

## Usage
src contains a src.py file that performs source reconstruction using the fsaverage MRI template by FreeSurfer. The Coregistration is done using the example script provided by mne-python and source reconstruction is performed with an LCMV beamformer for specified regions of interest (ROI).

The src_demo.py file is an example of usage.

For TRF analysis you can install sPyEEG which is a package that has been developed in the Sensory Neuroengineering Lab @ Imperial College London and FAU Erlangen: https://github.com/phg17/sPyEEG.

```
git clone --recurse-submodules https://github.com/Al2606/MEG-Analysis-Pipeline.git
cd MEG-Analysis-Pipeline
git submodule update --init --recursive
```

Then follow the instructions in https://github.com/phg17/sPyEEG/blob/efd442f3d206103adfb2e5db701138e3a5002a40/README.md to install sPyEEG.

trf_demo.py provides an example usage for TRF calculation with MEG data.

## Support
For questions please contact alina.schueller@fau.de

