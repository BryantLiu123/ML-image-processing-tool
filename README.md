# ML-image-processing-tool
A Python-based data preprocessing tool that converts atomic simulation output files (extended CFG format) from Silicon Carbide (SiC-SiC) molecular dynamics simulations into structured 2D grid representations suitable for machine learning training.

The tool reads fractional coordinate data from CFG files and projects the 3D atomic positions onto a configurable 2D grid (default 192×192), producing a two-channel NumPy array per timestep:

- Channel 0 – Grain Boundary Map: Quantifies microstructural grain boundary mixing within each spatial bin. Bins containing atoms from multiple grains receive higher scores, enabling the model to learn how grain boundary topology influences crack behavior.
- Channel 1 – Void/Crack Map: A binary mask identifying spatial regions with no atomic occupancy, directly capturing crack geometry and void distribution.

Post-processing steps include optional Gaussian smoothing of grain boundary edges and automated Otsu thresholding to suppress noise and sharpen boundary detection. Each timestep also produces a three-panel diagnostic PNG — individual channel maps plus a color-coded overlay — for rapid visual quality checks.

Designed as a preprocessing front-end for ML models targeting crack propagation prediction in ceramic composites, the pipeline supports sequential timestep processing to build labeled image datasets from large-scale atomistic simulations.
