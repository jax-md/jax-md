
# Autogenerated script for running parameter optimization using scipy. 

import os, sys, json, glob, re
from scipy.optimize import minimize
import numpy as np
from parmedmod import UpdateParmTopCLI
from ase.io import read, write

# Reads json data 
def ReadJsonData(json_path='params.json'):

    with open(json_path, 'r') as f:
        json_data = json.load(f)
        
    return json_data

# dumps json data into an existing file
def SaveJsonData(field_dict, json_path='params.json'):

    with open(json_path, 'r') as f:
        json_data = json.load(f)
        
    for key in field_dict:
        json_data[key]=field_dict[key]

    with open(json_path, 'w') as outfile:
        json.dump(json_data, outfile)

# updates amber prmtop file, runs constrained optimizations, computes difference between ref and computed energy profiles and RMSD.  
def ObjectiveFunction(scipy_params, *args):

    params_dict, optvars_dict, ref_ene, output_flist=args

    # set new parameters
    i=0
    for key, value in params_dict.items():
        if(optvars_dict['height']):
            value['height']=scipy_params[i]
            i+=1

        if(optvars_dict['phase']):
            value['phase']=scipy_params[i]
            i+=1

        if(optvars_dict['periodicity']):
            value['periodicity']=scipy_params[i]
            i+=1

        if(optvars_dict['scee']):
            value['scee']=scipy_params[i]
            i+=1

        if(optvars_dict['scnb']):
            value['scnb']=scipy_params[i]
            i+=1

    # update parmtop file with new parameters
    dh_dir='./confs_999-999/dh_6-7-9-11/'
    UpdateParmTopCLI(dh_dir+'prmtop', params_dict)

    # run torsional scan calculations using geometric and sander
    ierr = os.system('cd %s && rm -rf *.tmp *.log *.out *_optim*' % (dh_dir))
    run_task_command="""bash run_task_0.sh > run_task_0.log 2>&1 &
    pid=$!
    echo $pid > run_task.pid
    wait $pid
    """
    ierr = os.system(run_task_command)
    if(ierr != 0):
        print('Error: Please check the run.log file.')
        return
    
    # extract data
    ene_list=list()
    for fname in output_flist:
        with open(dh_dir+fname, 'r') as f:
            lines=f.readlines()
    
        ene=[float(re.findall(r'(\-*\d+\.\d+)',l)[0]) for l in lines if re.match(r'.*Energy.*',l)]
        if(len(ene) == 0):
            print('Error: Failed to extract energy from %s.' % (fname)) 
            return

        ene_list.append(ene[-1])

        write(dh_dir+fname.replace('_optim.xyz','.Opt.xyz'), read(dh_dir+fname, index=-1))
    
    # save parameters and energies
    step_number=len(ReadJsonData())-1
    
    SaveJsonData({'step_%d' % (step_number):params_dict}, 'params.json')
    SaveJsonData({'step_%d' % (step_number):ene_list}, 'energies.json')
        
    # compute rmsd
    min_ene = min(ene_list)
    relative_ene_list = [(x - min_ene) * 627.5 for x in ene_list]

    np_relative_ene_list=np.array(relative_ene_list)
    np_ref_ene=np.array(ref_ene)

    difference=np_ref_ene-np_relative_ene_list
    return np.sum(difference ** 2)


# runs parameter optimization using scipy
def RunScipyOptimization(*args):

    initial_guess, ref_ene, algorithm, maxiter, step_size=args
    params_dict=ReadJsonData()[initial_guess]
    optvars_dict=ReadJsonData()['optvars']
    bounds_dict=ReadJsonData()['bounds']
    
    guess=list()
    bounds=list()

    for key, value in params_dict.items():
        if(optvars_dict['height']):
            guess.append(value['height'])
            bounds.append(bounds_dict['height'])

        if(optvars_dict['phase']):
            guess.append(value['phase'])
            bounds.append(bounds_dict['phase'])

        if(optvars_dict['periodicity']):
            guess.append(value['periodicity'])
            bounds.append(bounds_dict['periodicity'])

        if(optvars_dict['scee']):
            guess.append(value['scee'])
            bounds.append(bounds_dict['scee'])

        if(optvars_dict['scnb']):
            guess.append(value['scnb'])
            bounds.append(bounds_dict['scnb'])
    
        output_flist=['dh_6-7-9-11_%03d' % (i) + '_optim.xyz' for i in range(36)]
    
    minimization_result=minimize(ObjectiveFunction, guess, args=(params_dict, optvars_dict, ref_ene, output_flist), \
            bounds=bounds, method=algorithm, options={'maxiter':maxiter, 'eps': step_size})

    parameters = minimization_result.x

    print('Minimization finished!')

# Main function. Users may modify the default values assigned to variables. 
def main(arg_values):

    initial_guess='initial_guess'
    ref_ene=[0.0,0.24041407495730027,0.7180356999677429,1.523482149935944,2.3976774999388795,3.2420603249764213,3.9534696249666013,4.270306924982776,4.275803824991726,4.150128124955188,3.7827582499446066,3.668954849980821,3.443494099972213,3.341964599971732,3.356309249936942,3.4091008249399124,3.5792474499419313,3.9089673249762313,4.470510800001648,5.026563649951754,5.4152183249399855,5.627595699985193,5.775955524968026,5.81154104994738,5.6101888499449615,5.361052524988281,4.971136574998241,4.523440424971454,3.9399093499815763,3.2612116249737255,2.5247650749540185,1.8916175749831154,1.1293870499835634,0.5700021749584039,0.16918654995379256,0.029498775001854938]
    algorithm='L-BFGS-B'
    maxiter=1000
    step_size=0.100000
    
    RunScipyOptimization(initial_guess, ref_ene, algorithm, maxiter, step_size)

    return
    
if __name__ == "__main__":
    main(sys.argv)
