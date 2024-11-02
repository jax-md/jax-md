
#!/usr/bin/bash

completed_steps=`grep -o 'step_' energies.json | wc -l`
total_steps=`grep 'maxiter=' scipyopt.py | cut -d'=' -f2`

echo "$completed_steps/$total_steps completed!"
