# pyolaf - A Python-based 3D reconstruction framework for light field microscopy

pyolaf is a Python port of the [oLaF](https://gitlab.lrz.de/IP/olaf/) 3D reconstruction framework for light field microscopy (LFM). 

## Overview
  
The light field microscope (LFM) allows for 3D imaging of fluorescent specimens using an array of micro-lenses (MLA) that capture both spatial and directional light field information in a single shot. oLaF is a Matlab framework for 3D reconstruction of LFM data with a deconvolution algorithm that reduces aliasing artifacts.

pyolaf brings these same features to the Python ecosystem, using GPU acceleration and some further code optimizations to **speed up deconvolution by 20x**. 

## Limitations

pyolaf only supports regular grids and single-focus conventional light-field microscopes.
In particular Fourier LFM, hexagonal grids, and multi-focus lenslets are currently not supported.
Pull requests to add these are welcome!

## Demos

<p align="center">
<img src="https://raw.githubusercontent.com/lambdaloop/pyolaf/main/img/fly_raw.png" width="40%" >
<img src="https://raw.githubusercontent.com/lambdaloop/pyolaf/main/img/fly_volume.gif" width="40%">
</p>
<p align="left">
(Left) Raw image a fly from light field microscope, showing the 10k+ lenslets in the microlens array, acquired within <a href=http://faculty.washington.edu/tuthill/>Tuthill Lab</a>
</p>
<p align="right">
(Right) Volume reconstructed from the image, shown as successive slices in a gif. See also the <a href="https://raw.githubusercontent.com/lambdaloop/pyolaf/main/img/fly_volume.mp4">higher resolution video</a>.
</p>


## Copyright

Copyright (c) 2017-2020 Anca Stefanoiu, Josue Page, and Tobias Lasser -- original oLaF code  
Copyright (c) 2023 Lili Karashchuk -- pyolaf

## Citation

When using pyolaf in academic publications, please reference the following citation:

- A. Stefanoiu et. al., "Artifact-free deconvolution in light field microscopy", Opt. Express, 27(22):31644, (2019).

