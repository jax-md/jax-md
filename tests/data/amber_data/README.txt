To generate CHARMM files
start with RAMP1.pdb from
https://ambermd.org/tutorials/basic/tutorial7/index.php
Go to PDB Reader & Manipulator
Don't check "Check/Correct PDB Format"
Select only protein chain (default)
Next page also default
Plain CHARMM outputs

for solvated system
Go to "Solution Builder"
Upload RAMP1.pdb, don't select "Check/Correct PDB Format"
Only select protein (default)
PDB Manipulation - leave default options
Leave default options for solvents, ion, etc
Same for PME FFT info and other options
Leave default forcefield option of CHARMM36m (OPLSAA/M is also available aparently)
Select GROMACS, AMBER, OpenMM, CHARMM/OpenMM for input generation
Under the main job folder, step3_pbcsetup.crd, step3_pbcsetup.psf, and the toppar.str file are used for omm inputs