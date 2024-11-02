#!/usr/bin/bash 

. ~/Apps/amber/install/amber.sh 
const_opt_dir=$(pwd)
. /etc/profile && module load amber/23 quick/23.08 xtb/6.6.1 geometric/1.0.1
if [ -f "${const_opt_dir}/geom.pids" ]; then rm -f ${const_opt_dir}/geom.pids; fi 
cd ./confs_999-999/dh_6-7-9-11/
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_000.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_000.rst7"}' --engine ase  dh_6-7-9-11_000.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_001.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_001.rst7"}' --engine ase  dh_6-7-9-11_001.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_002.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_002.rst7"}' --engine ase  dh_6-7-9-11_002.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_003.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_003.rst7"}' --engine ase  dh_6-7-9-11_003.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_004.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_004.rst7"}' --engine ase  dh_6-7-9-11_004.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_005.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_005.rst7"}' --engine ase  dh_6-7-9-11_005.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_006.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_006.rst7"}' --engine ase  dh_6-7-9-11_006.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_007.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_007.rst7"}' --engine ase  dh_6-7-9-11_007.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_008.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_008.rst7"}' --engine ase  dh_6-7-9-11_008.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_009.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_009.rst7"}' --engine ase  dh_6-7-9-11_009.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_010.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_010.rst7"}' --engine ase  dh_6-7-9-11_010.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_011.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_011.rst7"}' --engine ase  dh_6-7-9-11_011.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_012.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_012.rst7"}' --engine ase  dh_6-7-9-11_012.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_013.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_013.rst7"}' --engine ase  dh_6-7-9-11_013.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_014.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_014.rst7"}' --engine ase  dh_6-7-9-11_014.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_015.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_015.rst7"}' --engine ase  dh_6-7-9-11_015.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_016.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_016.rst7"}' --engine ase  dh_6-7-9-11_016.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_017.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_017.rst7"}' --engine ase  dh_6-7-9-11_017.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_018.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_018.rst7"}' --engine ase  dh_6-7-9-11_018.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_019.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_019.rst7"}' --engine ase  dh_6-7-9-11_019.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_020.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_020.rst7"}' --engine ase  dh_6-7-9-11_020.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_021.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_021.rst7"}' --engine ase  dh_6-7-9-11_021.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_022.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_022.rst7"}' --engine ase  dh_6-7-9-11_022.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_023.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_023.rst7"}' --engine ase  dh_6-7-9-11_023.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_024.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_024.rst7"}' --engine ase  dh_6-7-9-11_024.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_025.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_025.rst7"}' --engine ase  dh_6-7-9-11_025.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_026.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_026.rst7"}' --engine ase  dh_6-7-9-11_026.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_027.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_027.rst7"}' --engine ase  dh_6-7-9-11_027.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_028.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_028.rst7"}' --engine ase  dh_6-7-9-11_028.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_029.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_029.rst7"}' --engine ase  dh_6-7-9-11_029.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_030.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_030.rst7"}' --engine ase  dh_6-7-9-11_030.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_031.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_031.rst7"}' --engine ase  dh_6-7-9-11_031.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_032.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_032.rst7"}' --engine ase  dh_6-7-9-11_032.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_033.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_033.rst7"}' --engine ase  dh_6-7-9-11_033.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_034.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_034.rst7"}' --engine ase  dh_6-7-9-11_034.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      
      geometric-optimize --ase-class=ase.calculators.amber.Amber --ase-kwargs='{"amber_exe":"sander -O ", "infile":"mdin", "outfile":"dh_6-7-9-11_035.out", "topologyfile":"prmtop", "incoordfile":"dh_6-7-9-11_035.rst7"}' --engine ase  dh_6-7-9-11_035.xyz constraints.txt &
      pid=$!
      echo $pid >> ${const_opt_dir}/geom.pids
      wait $pid
      cd ${const_opt_dir}
module unload amber/23 quick/23.08 xtb/6.6.1 geometric/1.0.1
