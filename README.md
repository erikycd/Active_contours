# Active_contours

## Vector-valued active contours for image segmentation

This repository shows an approach for multichannel image segmentation using the vector-valued version of active contours. [[1]](#1)

The theory of active contours (AC) became widely popular among image segmentation methods which dates back to the beginning of computer vision. Basically, a set of active moving curves is placed over the image, also known as active contours. AC are then used to compare the image content in order to reach a balance of regularity of the inner and the outer partitions. This leads to a semi-automatic region split and object segmentation.

Precisely, a given energy functional related to the region balance must be solved in order to obtain the minimum variance quantity among all the resulting regions. This solution allows AC to be studied as an optimization problem. Original paper in [[2]](#2).

<p align="center">
  <img width="34.3%" src="https://github.com/erikycd/Active_contours/blob/main/airplane_animation.gif?raw=true">
  <img width="50%" src="https://github.com/erikycd/Active_contours/blob/main/zebra_animation.gif?raw=true">
</p>

## References

<a id="1">[1]</a> 
[Erik Carbajal-Degante, Jimena Olveres, Boris Escalante-Ramírez, 
"A multiphase active contour model based on the Hermite transform for texture segmentation," 
Proc. SPIE 10679, Optics, Photonics, and Digital Technologies for Imaging Applications V, 
106791H (24 May 2018)](https://doi.org/10.1117/12.2306541)

<a id="2">[2]</a> 
[Tony F. Chan, B.Yezrielev Sandberg, Luminita A. Vese,
Active Contours without Edges for Vector-Valued Images,
Journal of Visual Communication and Image Representation,
Volume 11, Issue 2,
2000,
Pages 130-141,
ISSN 1047-3203](https://www.sciencedirect.com/science/article/pii/S104732039990442X)
