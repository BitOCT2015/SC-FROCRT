# SC-FROCRT
SC-FROCRT Reconstruction
This repository contains the implementation of the SC-FROCRT (Sparse Continuous Full-Range Optical Coherence Refraction Tomography) reconstruction framework. This method is designed for high-resolution, full-range, and isotropic OCT imaging. The repository will be in continued development as we work to refine and expand the functionality of the method.

Method Overview
SC-FROCRT incorporates three key modules:

Super-Resolution Iterative Module: Based on sample priors, this module enhances resolution by iteratively reconstructing the image.
Conjugate Artifact Removal Module: Effectively removes conjugate artifacts that commonly arise in traditional OCT systems.
Isotropic Reconstruction Module: Aims to achieve isotropic resolution by combining multi-angle information using Fourier synthesis.
Current Version
The current version of this repository provides Matlab demo code for the following two modules:
Conjugate Artifact Removal Module
Super-Resolution Iterative Module

The Isotropic Reconstruction Module code can be adapted from K. C. Zhou’s implementation of optical coherence refraction tomography (OCRT). You can find the relevant code at this GitHub repository: K. C. Zhou, "Computational 3D microscopy with optical coherence refraction tomography (OCRT)," Github (2022).