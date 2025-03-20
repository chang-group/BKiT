<div align="center">
  <h1>Binding Kinetic Toolkit (BKiT)</h1>
</div>
<h2>Introduction</h2>
Binding Kinetic Toolkit (BKiT) is an open-source, comprehensive post-molecular dynamics (MD) analysis toolkit providing a wide array of analysis functions and visualizations for researchers with varied levels of expertise in molecular mechanics. BKiT robustly guides the construction of ligand binding/unbinding free energy (FE) landscape, also termed potential of mean force (PMF), which reveals both binding affinity and kinetics for drug design. With the FE landscape using milestoning theory implemented in BKiT, previous studies demonstrated that our designed compound achieved better drug binding residence time in experiments. Its analytical functions include, but not limited to, reducing high-dimensional ligand–protein motions to a 2D or 3D space using principal component analysis (PCA) for conformation analysis and better description of molecular motions using Cartesian or internal bond-angle-torsion (BAT) coordinates.

<h2>Functions</h2>
1. PostMD_Analysis.ipynb.<br />
&nbsp;&nbsp;&nbsp;&nbsp;You can visualize organic molecules and/or bio-macromolecules, project Cartesian coordinates into 2D PC space, select frames within a user-defined radius on PC space, analyze dihedral angle PCA, etc. <br />
<br />
2. PMF_Calculations_1D_RMSD.ipynb.<br />
&nbsp;&nbsp;&nbsp;&nbsp;You can construct free energy landscape using ligand RMSD from many short MDs with Milestoning theory. <br />
<br />
3. 2D_PC_Milestoning_Generate.ipynb.<br />
&nbsp;&nbsp;&nbsp;&nbsp;You can build reaction path on 2D PC space and finetune the path to visualize the representative conformations. <br />
<br />
4. PMF_Calculation_2D_PC.ipynb.<br />
&nbsp;&nbsp;&nbsp;&nbsp;With the path built in notebook3 and PC result from short MDs, you can compute PMF and drug residence time.<br />
<br />

<h2>Installation</h2>
git clone https://github.com/chang-group/BKiT<br />
Dependencies are listed at beginning of each notebook<br />
