#driver for amber dynamics
#add file input
#also make sure that this interface is still relatively convenient for python usage
#also make amber_fe.py file for free energy

# TODO should create simulation environment and support all forms of geometry optimization
# TODO note to consider about units and time step units in particular (is 0.001 actually 1fs)
# https://academiccharmm.org/documentation/basicusage

# TODO should i stick to from style imports? 
# standard lib imports os, sys, etc
import os
import re
import sys
from typing import Optional, List, Dict # TODO unsure if this is safe with jax
import time
# jax and third party imports
import jax
import jax.numpy as jnp # TODO would it just be better to exclude this and force conversion where necessary
import numpy as np
# local application imports e.g. from amber_helper ...
import openmm as omm
from openmm import app
from jax_md import dataclasses, space, minimize, simulate, quantity
from jax_md.amber.amber_helper import load_amber_ff, move_dataclass, load_ffq_ff
from jax_md.amber.amber_energy_v2 import amber_energy
import parmed as pmd

# TODO do these need to be frozen=True or are they implicitly static?
@dataclasses.dataclass
class SimulationConfig(object):
    name: str
    iterations: float

@dataclasses.dataclass
class MdinConfig:
    """
    Amber25 Manual, doesn't include everything

    22.6.10. The “middle” scheme
    ischeme
    ithermostat
    therm_par

    22.6.11. Water cap
    ivcap
    fcap
    cutcap
    xcap
    ycap
    zcap

    22.6.12. NMR refinement options
    iscale
    noeskp
    ipnlty
    mxsub
    scalm
    pencut
    tausw

    22.7.2. Particle Mesh Ewald
    nfft1, nfft2, nfft3
    order
    verbose
    ew_type
    dsum_tol
    rsum_tol
    mlimit(1,2,3)
    ew_coeff
    nbflag
    skinnb
    skin_permit
    nbtell
    netfrc
    vdwmeth
    eedmeth
    eedtbdns
    column_fft

    26.1.4. Basic inputs for thermodynamic integration
    icfe
    clambda
    klambda
    tishake
    idecomp
    timask1
    timask2

    26.1.6. Softcore Potentials in Thermodynamic Integration
    ifsc
    scalpha
    logdvdl
    dvdl_norest
    dynlmb
    crgmask
    scmask
    scmask1
    scmask2

    26.1.7. pmemd.cuda-specific functionalities
    gti_lam_sch
    gti_ele_sc
    gti_vdw_sc
    gti_scale_beta
    gti_vdw_exp
    gti_ele_exp
    scbeta
    gti_cut_sc
    gti_add_sc
    gti_bat_sc
    sc_bond_mask1
    sc_bond_mask2
    sc_angle_mask1
    sc_angle_mask2
    sc_torsion_mask1
    sc_torsion_mask2
    tishake
    gti_syn_mass
    ti_vdw_mask
    gti_output
    gti_cpu_output
    gti_cut
    gti_chg_keep

    PySander API can also create simulation objects but it isn't clear
    if they contain all of the relevant flags

    """

    # 22.6.1. General flags describing the calculation
    imin: int = 0
    # nmropt

    # 22.6.2. Nature and format of the input
    ntx: int = 1
    irest: int = 0

    # 22.6.3. Nature and format of the output
    # ntxo
    ntpr: int = 50
    ntave: int = 0
    ntwr: int = 0 # TODO should technically default to nstlim
    # iwrap
    ntwx: int = 0
    ntwv: int = 0
    # ionstepvelocities
    ntwf: int = 0
    ntwe: int = 0
    # ioutfm
    # ntwprt
    idecomp: int = 0 # TODO this is going to require a lot of thought

    # 22.6.4. Frozen or restrained atoms
    # ibelly
    # ntr
    # restraint_wt
    # restraintmask
    # bellymask

    # 22.6.5. Energy minimization
    maxcyc: int = 1
    ncyc: int = 10
    ntmin: int = 1
    dx0: float = 0.01
    drms: float = 1e-4 # TODO caution, this is kcal * mol^-1 * A^-1

    # 22.6.6. Molecular dynamics
    nstlim: int = 1
    nscm: int = 1000
    t: float = 0.0
    dt: float = 0.001
    #nrespa

    # 22.6.7. Temperature regulation
    ntt: int = 0 # TODO it isn't clear if this is actualy the default
    temp0: float = 300.0
    # temp0les
    tempi: float = 0.0
    ig: int = -1
    # tautp # TODO unclear if this is needed for plain langevin dynamics
    gamma_ln: float = 0.0
    # vrand
    # vlimit
    # nkija
    # idistr
    # sinrtau

    # 22.6.8. Pressure regulation
    ntp: int = 0
    barostat: int = 1
    # mcbarint
    pres0: float = 1.0
    # comp
    taup: float = 1.0
    # baroscalingdir
    # csurften
    # gamma_ten
    # ninterface

    # 22.6.9. SHAKE bond length constraints
    ntc: int = 1
    # tol
    # jfastw
    # noshakemask

    # 22.7.1. Generic parameters
    ntf: int = 1
    ntb: int = 0 # TODO default behavior here is more complicated
    # dielc
    cut: float = 8.0 # Angstrom
    # fswitch
    # nsnb
    # ipol
    # ipgn
    # ifqnt
    # igb
    # ipb
    # irism
    # ievb
    # iamoeba
    # lj1264
    # plj1264
    # efx
    # more after this ...

    # misc parameters
    skinnb: float = 2 # Angstrom TODO should go to PME section
    sc_scaling: int = 0 # 0 no scaling, 1 chg scaling, 2 vdw scaling, 3 both scaling

    # Simulation steps and timing
    # nstlim: int = 0         # number of MD steps
    # dt: float = 0.002       # timestep in ps
    # ntpr: int = 500         # print interval
    # ntwx: int = 500         # trajectory output interval
    # ntwr: int = 5000        # restart file write interval

    # Thermostat and barostat
    # ntt: int = 3            # thermostat type
    # temp0: float = 300.0    # target temperature
    # gamma_ln: Optional[float] = None  # Langevin collision frequency
    # ig: int = -1            # random seed

    # ntb: int = 1            # constant volume (1) or pressure (2)
    # ntp: int = 0            # pressure coupling (1 or 2)
    # pres0: float = 1.0      # target pressure
    # taup: float = 1.0       # pressure relaxation time

    # Restraints
    # ntr: int = 0            # positional restraints flag
    # restraint_wt: Optional[float] = None
    # restraintmask: Optional[str] = None

    # Cutoffs and PME
    # cut: float = 8.0        # nonbonded cutoff
    # igb: int = 0            # implicit solvent model (0 = off)
    # saltcon: Optional[float] = None

    # Free energy
    # ifsc: int = 0           # softcore potential flag
    # clambda: float = 0.0    # lambda value
    # scalpha: Optional[float] = None
    # scbeta: Optional[float] = None
    # icfe: int = 0           # free energy calculation flag

    # PBC and PME options
    # iwrap: int = 1
    # ntc: int = 2
    # ntf: int = 2
    # nsnb: Optional[int] = None

    # Miscellaneous
    # ntx: int = 1
    # ioutfm: int = 1
    # ntxo: int = 2

    unknown_fields: Dict[str, str] = dataclasses.field(default_factory=dict)

def parse_mdin_file(filename: str) -> MdinConfig:
    # TODO these mdin files are essentially fortran namelist files
    # there may be a better way of parsing them
    with open(filename, 'r') as f:
        lines = f.readlines()

    param_map = {}
    in_cntrl_block = False
    for line in lines:
        line = line.strip()
        if line.lower().startswith("&cntrl"):
            in_cntrl_block = True
            line = line[6:].strip()  # remove '&cntrl'
        if in_cntrl_block:
            if '/' in line:
                line = line.replace('/', '')
                in_cntrl_block = False
            # Remove comments
            line = re.sub(r'!.*', '', line)
            items = line.split(',')
            for item in items:
                if '=' in item:
                    k, v = item.strip().split('=')
                    param_map[k.strip().lower()] = v.strip()

    # Initialize config and populate known fields
    config = MdinConfig()
    updates = {}
    for field_ in config.__dataclass_fields__:
        if field_ in param_map:
            val = param_map[field_]
            typ = type(getattr(config, field_))
            if typ == bool:
                #setattr(config, field_, val.lower() in ['1', 'true', 'yes'])
                #config = replace(config, **{field_: val.lower() in ['1', 'true', 'yes']})
                updates[field_] = val.lower() in ['1', 'true', 'yes']
            elif typ == Optional[float] or typ == float:
                #setattr(config, field_, float(val))
                #config = dataclasses.replace(config, **{field_: float(val)})
                updates[field_] = float(val)
            elif typ == Optional[int] or typ == int:
                #setattr(config, field_, int(val))
                #config = dataclasses.replace(config, **{field_: int(val)})
                updates[field_] = float(val)
            elif typ == str or typ == Optional[str]:
                #setattr(config, field_, val)
                #config = dataclasses.replace(config, **{field_: val})
                updates[field_] = val
    
    config = dataclasses.replace(config, **updates)

    # Store any unknown fields
    known = set(config.__dataclass_fields__)
    #config.unknown_fields = {k: v for k, v in param_map.items() if k not in known}

    updates = {k: v for k, v in param_map.items() if k not in known}
    config = dataclasses.replace(config, unknown_fields=updates)

    return config

# some general ideas for an openmm topology converter
# @dataclasses.dataclass(frozen=True)
# class SystemTopology:
#     masses: jax.Array
#     charges: jax.Array
#     positions: jax.Array
#     box_vectors: Optional[jax.Array]
#     bonds: jax.Array  # shape (N_bonds, 2)
#     bond_params: jax.Array  # shape (N_bonds, 2)
#     angles: jax.Array
#     dihedrals: jax.Array
#     lj_types: jax.Array  # or sigma/epsilon arrays
#     exclusions: jax.Array

# def openmm_to_jax(topology: app.Topology, system: openmm.System, positions: np.ndarray) -> SystemTopology:
#     masses = np.array([atom.mass.value_in_unit(unit.dalton) for atom in system.getMasses()])
#     charges = np.zeros(len(masses))
#     for force in system.getForces():
#         if isinstance(force, openmm.NonbondedForce):
#             for i in range(system.getNumParticles()):
#                 charge, sigma, epsilon = force.getParticleParameters(i)
#                 charges[i] = charge.value_in_unit(unit.elementary_charge)
#             break

#     bonds = []
#     for bond in topology.bonds():
#         i = bond[0].index
#         j = bond[1].index
#         bonds.append((i, j))
#     bonds = np.array(bonds)

#     pos = np.array(positions)  # shape (N, 3)

#     return SystemTopology(
#         masses=jnp.array(masses),
#         charges=jnp.array(charges),
#         positions=jnp.array(pos),
#         bonds=jnp.array(bonds),
#         ...
#     )

# TODO there is probably a cleaner way of doing this, also need format fn for softcore block

def _fmt(val: Optional[float], width=12, precision=4):
    if val is None:
        return " " * (width - 5) + "N/A"
    else:
        return f"{val:>{width}.{precision}f}"

def _fmt_sci(val: Optional[float], width=10):
    if val is None:
        return " " * (width - 5) + "N/A"
    else:
        return f"{val:>{width}.4E}"

def format_mdout_block(
    nstep: int,
    time_ps: float,
    temp: Optional[float] = None,
    press: Optional[float] = None,
    etot: Optional[float] = None,
    ektot: Optional[float] = None,
    eptot: Optional[float] = None,
    ebond: Optional[float] = None,
    eangle: Optional[float] = None,
    edihed: Optional[float] = None,
    e14nb: Optional[float] = None,
    e14eel: Optional[float] = None,
    evdw: Optional[float] = None,
    eel: Optional[float] = None,
    ehbond: Optional[float] = None,
    erest: Optional[float] = None,
    dvdln: Optional[float] = None,
    ekcmt: Optional[float] = None,
    virial: Optional[float] = None,
    volume: Optional[float] = None,
    density: Optional[float] = None,
    ewald_err: Optional[float] = None,
) -> str:
    # ret_str = f"""\
    #     NSTEP = {nstep:>8d}   TIME(PS) = {time_ps:>11.3f}  TEMP(K) = {_fmt(temp, width=8, precision=2)}  PRESS = {_fmt(press, width=8, precision=1)}
    #     Etot   = {_fmt(etot)}  EKtot   = {_fmt(ektot)}  EPtot      = {_fmt(eptot)}
    #     BOND   = {_fmt(ebond)}  ANGLE   = {_fmt(eangle)}  DIHED      = {_fmt(edihed)}
    #     1-4 NB = {_fmt(e14nb)}  1-4 EEL = {_fmt(e14eel)}  VDWAALS    = {_fmt(evdw)}
    #     EELEC  = {_fmt(eel)}  EHBOND  = {_fmt(ehbond)}  RESTRAINT  = {_fmt(erest)}
    #     DV/DL  = {_fmt(dvdln)}
    #     EKCMT  = {_fmt(ekcmt)}  VIRIAL  = {_fmt(virial)}  VOLUME     = {_fmt(volume)}
    #                                                         Density    = {_fmt(density)}
    #     Ewald error estimate:   {_fmt_sci(ewald_err)}"""

    ret_lines = [
        f"NSTEP = {nstep:>8d}   TIME(PS) = {time_ps:>11.3f}  TEMP(K) = {_fmt(temp, width=8, precision=2)}  PRESS = {_fmt(press, width=8, precision=1)}",
        f"Etot   = {_fmt(etot)}  EKtot   = {_fmt(ektot)}  EPtot      = {_fmt(eptot)}",
        f"BOND   = {_fmt(ebond)}  ANGLE   = {_fmt(eangle)}  DIHED      = {_fmt(edihed)}",
        f"1-4 NB = {_fmt(e14nb)}  1-4 EEL = {_fmt(e14eel)}  VDWAALS    = {_fmt(evdw)}",
        f"EELEC  = {_fmt(eel)}  EHBOND  = {_fmt(ehbond)}  RESTRAINT  = {_fmt(erest)}",
        f"DV/DL  = {_fmt(dvdln)}",
        f"EKCMT  = {_fmt(ekcmt)}  VIRIAL  = {_fmt(virial)}  VOLUME     = {_fmt(volume)}",
        f"                                                    Density    = {_fmt(density)}",
        f"Ewald error estimate:   {_fmt_sci(ewald_err)}\n"
    ]

    return "\n".join(ret_lines)

# TODO add kwargs?
def amber_engine(mdin=None,
                opt_dict:dict={},
                inpcrd=None,
                prmtop=None,
                out_dir=None,
                return_type=None,
                precision="double",
                fluctuating_charge=None, # can be either "FFQ" or "LRCH"
                ffq_file=None,
                **kwargs):
    """
    return_type: int in [-1,0,1,2,3,...]
    sets return values from this runner for when this is used programatically
    if -1/None, file output is enabled, else data is only returned from the function
    """
    # TODO this should be in a constants file
    # TODO is this correct for the system of units and being input to the integrator?
    kB = 0.00831446267

    if precision == "double" and not jax.config.read("jax_enable_x64"):
        raise RuntimeError("Double precision requested but jax_enable_x64 is not enabled.")
    
    if precision == "double":
        dtype = np.float64
    else:
        dtype = np.float32
    # need to document standard range of input choices
    # also should consider omm, parmed, ase, etc for parsing input files
    # TODO openmm custom force object support?
    config = MdinConfig()
    # TODO conceptual overview
    # front end parser that takes mdin file, coords, restarts, ff
    # intermediate data layer that is a universal abstraction
    # back end engine (mostly jax md) that is agnostic

    # md input file parser ###############################
    if mdin != None:
        print("[INFO] mdin file being loaded:", mdin)
        config = parse_mdin_file(mdin)

    # replace user specified parameters in the default config
    if opt_dict != None:
        print("[INFO] Options being overwritten from opt_dict:", opt_dict)
        config = dataclasses.replace(config, **opt_dict)

    # TODO look at mdout file for general formatting
    print("[INFO] Current config:", config)

    # setting internal flags based on config ##########################################################
    is_periodic = False
    # TODO might need flags for constant volume (1) and pressure (2)
    if config.ntb in [1,2]:
        is_periodic = True

    # which simulation environment to use (NVE, NVT, NPT)

    # topology/parameter parser ###############################
    # TODO will eventually be something like
    # inpcrd/prmtop/psf/... -> omm.app.Topology -> SystemTopology -> generic_mm_energy

    # example of this ###################################################################
    # # can use AMBER
    # prmtop = app.AmberPrmtopFile('sys.prmtop')
    # inpcrd = app.AmberInpcrdFile('sys.inpcrd')
    # system = prmtop.createSystem(nonbondedMethod=app.PME, constraints=app.HBonds)
    # topology = prmtop.topology
    # positions = inpcrd.positions.value_in_unit(nanometer)

    # # or also OpenFF
    # from openff.toolkit.topology import Molecule
    # from openff.toolkit.typing.engines.smirnoff import ForceField as OFFForceField
    # mol = Molecule.from_file('mol.sdf')
    # off_ff = OFFForceField('openff_unconstrained-2.0.0.offxml')
    # off_top = mol.to_topology()
    # omm_sys = off_ff.create_openmm_system(off_top.to_openmm())
    # positions = mol.conformers[0].value_in_unit(nanometer)
    # topology = off_top.to_openmm()

    # # convert to jax with common routine
    # jax_top = openmm_to_jax(topology, system, positions)

    # if inpcrd != None:
        
    # if prmtop != None:
    #     prmtop =
    if inpcrd == None or prmtop == None:
        raise Exception("inpcrd/prmtop not provided")

    # load softcore mask with parmed
    if "scmask" in opt_dict.values():
        structure_pmd = pmd.load_file(prmtop, inpcrd)
        softcore_atoms = pmd.amber.AmberMask(structure_pmd, opt_dict["scmask"]).Selection()
        softcore_atoms = jnp.array(softcore_atoms, dtype=bool)
        print("[INFO] Selected softcore mask is:", softcore_atoms)
    # unit parser and conversion (atomic units?) ###############################

    # interaction list/energy function builder ###############################
    # TODO should eventually fetch some or all of this from the omm topology object, or add it as part of the parser, this is just a temporary interface
    cutoff = config.cut/10.0
    dr_threshold = config.skinnb/10.0

    print(f"[WARNING] Internal nonbonded cutoff: {cutoff} NM - Neighbor rebuild threshold (skinnb): {dr_threshold} NM")

    if is_periodic:
        nonbonded_method = "PME"
    else:
        nonbonded_method = "NoCutoff"

    if fluctuating_charge == "FFQ":
        charge_method = "FFQ"
    elif fluctuating_charge == "LRCH":
        charge_method = "LRCH"
    else:
        charge_method = "GAFF"
    
    print(f"[INFO] Chosen charge method is {charge_method}")

    coul_scaling = None
    vdw_scaling = None
    if config.sc_scaling != 0:
        print(f"[INFO] alchemical scaling is active and set to: {config.sc_scaling}")
        print("For example, 0 no scaling, 1 linear chg scaling, 2 softcore vdw scaling")
        if config.sc_scaling == 1:
            coul_scaling="linear"
        elif config.sc_scaling == 2:
            vdw_scaling="softcore"

    print(f"[INFO] Electrostatic scaling = \"{coul_scaling}\" and vdW scaling = \"{vdw_scaling}\"")

    ff = load_amber_ff(inpcrd_file=None, prmtop_file=prmtop, 
                        ffq_file=ffq_file, nonbonded_method=nonbonded_method,
                        charge_method=charge_method, dr_threshold=dr_threshold, dtype=dtype, cutoff=cutoff)

    # moves dataclass objects from onp instantiation to jnp instance
    # TODO be careful about things captured by closure here, because they can't be changed, e.g. ff.grid_points
    ff = move_dataclass(ff, jnp)

    # TODO this only prints if ntb > 0
    print(f"[INFO] Unused box vectors from prmtop are: {ff.box_vectors}")

    # TODO at some point move dataclass to jax

    inpcrd_omm = app.AmberInpcrdFile(inpcrd)
    positions = inpcrd_omm.getPositions(asNumpy=True).value_in_unit(omm.unit.nanometer)
    
    if config.ntb > 0:
        box_vectors = ff.box_vectors
    # read velocities
    if config.ntx == 5:
        velocities = inpcrd_omm.getVelocities(asNumpy=True).value_in_unit(omm.unit.nanometer / omm.unit.picosecond)
        # if ntb > 0 also read in box vectors, necessary for npt restart i believe
        if config.ntb > 0:
            box_vectors = inpcrd_omm.getBoxVectors()
            # print(box_vectors)
            box_vectors = jnp.array([b.value_in_unit(omm.unit.nanometer) for b in box_vectors]) # TODO this may not be necessary at all

            if box_vectors.ndim == 2:
                print("[WARNING] Box vectors are assumed to be (3,) for an orthogonal box in current implementation")
                box_vectors = jnp.diag(box_vectors)

            print(f"[INFO] Box Vectors are loaded from inpcrd file as: {box_vectors}")
            ff = dataclasses.replace(ff, box_vectors=box_vectors)
            # print(box_vectors)

    # TODO i don't think this object updates if the parameters are changed, not an issue here, just an observation
    # should remove the other code path
    ffq_ff = None
    if ffq_file != None:
        ffq_ff = load_ffq_ff(ffq_file, dtype)
        ffq_ff = move_dataclass(ffq_ff, jnp)

    sim_tuple = amber_energy(ff=ff, nonbonded_method=nonbonded_method,
                                                charge_method=charge_method, ensemble=None,
                                                timestep=config.dt, init_temp=None, ffq_ff=ffq_ff, coul_scaling=coul_scaling, vdw_scaling=vdw_scaling, return_mode="simple")

    nrg_fn, ff, _body_fn, _state, nbr_fn = sim_tuple

    # TODO should ev
    # if config["ntb"] != :
    #     neighbor_fn = partition.neighbor_list(
    #             space.canonicalize_displacement_or_metric(disp_fn),
    #             box=box_vectors, # TODO avoid things that might become traced, such as from ff object
    #             r_cutoff=cutoff,
    #             dr_threshold=fr_threshold,
    #             disable_cell_list=False,
    #             custom_mask_function=mask_fn,
    #             fractional_coordinates=False,
    #             format=nbr_fmt, # TODO was ordered sparse, but need to figure out converting this from sparse to dense
    #         )
    #     nbr_list = neighbor_fn.allocate(positions)
    # else:
    #     neighbor_fn = {allocate:lambda x:None, update:lambda x:None} # or return something like []
    #     nbr_list = neighbor_fn.allocate(positions) # or just no op and make this []
    #     # or configure to actually generate neighbor list, but with infinite cutoff

    if is_periodic:
        print("[INFO] System is periodic, generating neighbor list")
        nbr_list = nbr_fn.allocate(positions)
        disp_fn, shift_fn = space.periodic(box_vectors) # TODO may need to come from prmtop
    else:
        nbr_list = None
        # class LambdaDict(dict):
        #     def __getattr__(self, key):
        #         return self[key]

        # nbr_list = LambdaDict({
        #     "update": lambda position: nbr_list,
        #     "allocate": lambda position: nbr_list,
        #     "did_buffer_overflow": False
        # })
        disp_fn, shift_fn = space.free()

    if config.imin == 1:
        if config.ntmin == 1:
            raise NotImplementedError("Hybrid gradient descent/cg method not implemented yet")
            # jax.scipy.cg()?
        elif config.ntmin == 2:
            print("[INFO] Running Gradient Descent Minimization")
            init_fn, apply_fn = minimize.gradient_descent(nrg_fn, shift_fn, step_size=config.dt)
        elif config.ntmin == 5:
            raise NotImplementedError("DLFind throuugh this interface is not yet supported")
            # dlfind_setup()
        elif config.ntmin == 6: # TODO not an official option
            print("[INFO] Running FIRE Minimization")
            print("[WARNING] ntmin == 6 is not an official option, subject to change in the future")
            init_fn, apply_fn = minimize.fire_descent(nrg_fn, shift_fn) # TODO how to set dt start and max?
    else:
        if config.ntt == 3:
            # TODO how to handle tempi and temp0? should center_velocity be false with separate cmm remover?
            print("[INFO] Running NVT Langevin Simulation")
            init_fn, apply_fn = simulate.nvt_langevin(nrg_fn, shift_fn, dt=config.dt, kT=config.temp0, gamma=config.gamma_ln, center_velocity=True)
        elif config.ntt == 9:
            raise NotImplementedError("Nose-Hoover thermostat not implemented, may not match AMBER implementation")
            #init_fn, apply_fn = simulate.nvt_nose_hoover()
        elif config.ntp == 1:
            # TODO need to look into isotropic vs anisotropic position scaling
            print("[INFO] Running NPT Nose-Hoover Simulation")
            print("[WARNING] Position scaling behavior may not match AMBER")
            # TODO what to use as pressure? what units? how to set taup/what units? should temp be temp0?
            init_fn, apply_fn = simulate.npt_nose_hoover(nrg_fn, shift_fn, dt=config.dt, pressure=config.pres0, kT=config.temp0, tau=config.taup)
        else:
            print("[INFO] Running NVE Simulation")
            init_fn, apply_fn = simulate.nve(nrg_fn, shift_fn, dt=config.dt)
        
    """
    nve
    nvt nose hoover
    npt nose hoover

    nvt langevin
    brownian?
    hybrid swap mc?

    simulate.nvt_nose_hoover_invariant/npt invariant for testing
    """
    # TODO add better handling for npt, simulate.npt_box(state)
    # should give the box information which needs to be used for the function

    # TODO interesting to note from amber
    #For example, when imin = 5 and maxcyc = 1000, sander will minimize each structure in the trajectory for 1000 steps
    # in the case of torsion scanning, this is basically just 36 frames from the same trajectory
    
    # init_fn, apply_fn = simulate.nve()

    # init_fn, apply_fn = simulate.nvt_langevin(solvated_nrg_fn, shift_fn, 1e-3, kT=1e-3*kB) # should probably be * kB
    # state = init_fn(jax.random.PRNGKey(0), positions, mass=ff.masses, kT=init_temp*kB, ff=ff, nbr_list=nbr_list)

    # TODO if tempi = 0.0 and ntx = 1, velocities are assigned from a Maxwellian distribution at tempi K
    #kwargs = {mass:ff.masses, kT:init_temp*kB}
    #print(type(ff.grid_points))
    # TODO should this be tempi or temp0?
    init_temp = config.tempi * kB
    print(f"[INFO] tempi: {config.tempi}, tempi * kB: {init_temp}")
    state = init_fn(jax.random.PRNGKey(0), positions, mass=ff.masses, kT=init_temp, ff=ff, nbr_list=nbr_list)

    # if restart, overwrite velocities
    if config.irest == 1:
        # p = mv
        print("[INFO] irest == 1, velocities are replaced with values from restart file")
        #print(velocities.shape)
        #print(ff.masses.shape)
        initial_momenta = velocities * ff.masses[:, None]
        #print("DEBUG NOT CHANGING MOMENTA")
        state = dataclasses.replace(state, momentum=initial_momenta)

    # TODO there should be a better way of doing this, making a dummy nblist seemed to have issues
    if is_periodic:
        def body_fn(i, state):
            state, ff, nbr_list = state
            nbr_list = nbr_list.update(state.position)
            state = apply_fn(state, ff=ff, nbr_list=nbr_list)

            return state, ff, nbr_list
    else:
        def body_fn(i, state):
            state, ff, nbr_list = state
            nbr_list = nbr_list
            state = apply_fn(state, ff=ff, nbr_list=nbr_list)

            return state, ff, nbr_list

    # num_steps is either ncyc or the step variable i think
    # output_freq = the smallest common factor of nt--
    # ntpr energy to mden
    ####
    # TODO create input flags for specifying file names and other sander options
    # ntave running energy averages -> mdout
    # ntwr crd/vel -> restrt
    # ntwx coordinates -> mdcrd
    # ntwv vel -> mdvel
    # ntwf frc -> mdfrc
    # ntwe 

    num_steps = jnp.max(jnp.array([config.maxcyc, config.nstlim], dtype=jnp.int32))

    # TODO this probably isn't very robust, omm's method of adding reporters seems better
    # this also strictly assumes that the most frequent output is a factor of the other outputs
    print("[WARNING] All output related intervals must be some multiple of the most frequent output")
    output_freqs = jnp.array([config.ntpr, config.ntave, config.ntwr, config.ntwx, config.ntwv, config.ntwf, config.ntwe], dtype=jnp.int32)
    output_freq = jnp.min(output_freqs[output_freqs>0])

    print(f"[INFO] Derived output frequency is {output_freq} while overall number of steps is {num_steps}")

    # collector for averages
    num_outputs = np.int32(num_steps/output_freq) + 1
    #print("num outputs",num_outputs)
    temp_vals = np.zeros(num_outputs)
    etot_vals = np.zeros(num_outputs)
    ektot_vals = np.zeros(num_outputs)
    eptot_vals = np.zeros(num_outputs)
    ebond_vals = np.zeros(num_outputs)
    eangle_vals = np.zeros(num_outputs)
    edihed_vals = np.zeros(num_outputs)
    e14nb_vals = np.zeros(num_outputs)
    e14eel_vals = np.zeros(num_outputs)
    evdw_vals = np.zeros(num_outputs)
    eel_vals = np.zeros(num_outputs)

    # initial output
    
    nrg_comps = nrg_fn(state.position, ff, nbr_list, debug=True)
    eptot = nrg_fn(state.position, ff, nbr_list)
    ektot = quantity.kinetic_energy(momentum=state.momentum, mass=state.mass)
    temp = quantity.temperature(momentum=state.momentum, mass=state.mass)/kB
    nstep = 0
    time_ps = config.t # TODO this should probably be overwritten with the rst file time
    print(format_mdout_block(
            nstep=nstep,
            time_ps=time_ps,
            temp=temp,
            #press=None,
            etot=eptot+ektot,
            ektot=ektot,
            eptot=eptot,
            ebond=nrg_comps["bond_pot"],
            eangle=nrg_comps["angle_pot"],
            edihed=nrg_comps["torsion_pot"],
            e14nb=nrg_comps["lj_14_pot"],
            e14eel=nrg_comps["coul_14_pot"],
            evdw=nrg_comps["lj_pot"],
            eel=nrg_comps["coul_pot"]
            #ebond=0.05,
            #evdw=800.123,
            #eel=-7200.55,
            #volume=30000.0,
            #density=0.9,
            #ewald_err=None
        ))

    temp_vals[0] = temp
    etot_vals[0] = eptot+ektot
    ektot_vals[0] = ektot
    eptot_vals[0] = eptot
    ebond_vals[0] = nrg_comps["bond_pot"]
    eangle_vals[0] = nrg_comps["angle_pot"]
    edihed_vals[0] = nrg_comps["torsion_pot"]
    e14nb_vals[0] = nrg_comps["lj_14_pot"]
    e14eel_vals[0] = nrg_comps["coul_14_pot"]
    evdw_vals[0] = nrg_comps["lj_pot"]
    eel_vals[0] = nrg_comps["coul_pot"]

    #print("state obj", state)
    #print("frcs", jax.grad(nrg_fn)(state.position, ff, nbr_list))

    ###################################################################################
    # # vdw test
    # if config.ntb == 0:
    #     print(ff.pairs.shape)
    #     pair0 = ff.pairs[:, 0]
    #     pair1 = ff.pairs[:, 1]
    # else:
    #     pair0 = nbr_list.idx[0, :]
    #     pair1 = nbr_list.idx[1, :]

    # sigma=0.5*(ff.sigma[pair0] + ff.sigma[pair1])
    # epsilon=jnp.sqrt(ff.epsilon[pair0] * ff.epsilon[pair1])

    # metric_fn = space.metric(disp_fn)
    # dist_fn = jax.vmap(metric_fn)
    # dr = dist_fn(state.position[pair0], state.position[pair1])

    # dr = jnp.where(jnp.isclose(dr, 0.), 1, dr)
    # idr = (sigma/dr)
    # idr2 = idr*idr
    # idr6 = idr2*idr2*idr2
    # idr12 = idr6*idr6
    # vdw_val = 4.0*epsilon*(idr12-idr6)

    # print(state.position)

    # print("14 pairs", ff.pairs_14)

    # if config.ntb == 0:
    #     print("num pairs vs n^2", len(ff.pairs))

    # print("Test vdw val", jnp.sum(vdw_val))

    # # it looks like in this case, <= or < doesn't matter
    # nb_mask = dr < 1.0
    # print("Test vdw val mask", jnp.sum(nb_mask * vdw_val))
    ####################################################################################

    if fluctuating_charge == "FFQ":
        _nrg, ffq_chgs = nrg_fn(state.position, ff, nbr_list, return_charges=True)
        print("FFQ modified charges", ffq_chgs[:ff.solute_cut])
        print("FF object charges", ff.charges[:ff.solute_cut])

    #sys.exit("[DEBUG] FIRST STEP VALIDATION")

    def wrap_crds(positions, residuce_indices, s_fn):
        # pos is N,3 - residue idx is N
        # the idea here for an alternate approach is to convert
        # the absolute wrapping from jax md to residue COM based wrapping
        # i think this will fix the visualization issues in something like vmd
        pos_wrapped = positions
        for res in np.unique(res_idx):
            idx = np.where(res_idx == res)[0]
            com = pos_wrapped[idx].mean(axis=0)
            com_wrapped = s_fn(com)
            delta = com_wrapped - com
            pos_wrapped = pos_wrapped.at[idx].add(delta)

    # TODO does this always need to be created?
    parm = pmd.load_file(prmtop, inpcrd)
    cell_angles = np.array([90.0,90.0,90.0])

    # TODO setup trajectory file/restart file and dump initial frame
    if config.ntwx != 0: # coordinate trajectory
        # TODO does this populate the trajectory with the initial frame?
        #parm = pmd.load_file(prmtop, inpcrd)  # reuse prmtop for topology/box
        #mdcrd = pmd.amber.netcdffiles.NetCDFTrajectory.open_new(os.path.join(out_dir, 'mdcrd'), parm, crds=True, vels=False)
        box_flag = config.ntb == 1
        # TODO should this deviate from amber behavior and includes forces/vels for convenience?
        mdcrd = pmd.amber.netcdffiles.NetCDFTraj.open_new(os.path.join(out_dir, 'mdcrd'), ff.atom_count, crds=True, box=box_flag, vels=False, frcs=False)
        # mdcrd.write(parm)
        #TODO add time?
        mdcrd.add_time(time_ps)
        mdcrd.add_coordinates(state.position * 10.0)
        #mdcrd.add_cell_lengths_angles(lengths=box_vectors, angles=cell_angles)
        if config.ntp == 1:
            print("current sim box for npt", state.box)
            mdcrd.add_cell_lengths_angles(lengths=state.box, angles=cell_angles)
        elif config.ntb == 1:
            mdcrd.add_cell_lengths_angles(lengths=box_vectors, angles=cell_angles)
    
    # if config.ntwr != 0: # restart
    #     # TODO does this populate the trajectory with the initial frame?
    #     #parm = pmd.load_file(prmtop, inpcrd)  # reuse prmtop for topology/box
    #     ncrst = pmd.amber.netcdffiles.NetCDFTrajectory.open_new(os.path.join(out_dir, 'ncrst'), parm, crds=True, vels=True)
    #     ncrst.write(parm)
    
    md_start_time = time.time()

    for i in range(int(num_steps/output_freq)):
        new_state, ff, nbr_list = jax.lax.fori_loop(0, output_freq, body_fn, (state, ff, nbr_list))
        # TODO this may be bad logic, i'm not sure if the neighbor list in amber_ff is even used?
        if is_periodic and nbr_list.did_buffer_overflow:
            print('Neighbor list overflowed, reallocating.')
            nbr_list = nbr_list.allocate(state.position)
        else:
            state = new_state
            #step += 1
        
        # pE = nrg_fn(state.position, ff, nbr_list)
        # kE = quantity.kinetic_energy(momentum=state.momentum, mass=state.mass)
        # temp = quantity.temperature(momentum=state.momentum, mass=state.mass)/kB
        # current_step = (i + 1) * output_freq
        # TODO debug print may have performance implications here
        #jax.debug.print("{step}, {pE}, {kE}, {pEkE}, {temp}", step=current_step, pE=pE, kE=kE, pEkE=(pE+kE), temp=temp)
        #print(pE, kE, pE+kE, temp, end='')
        # TODO probably move this to a separate function to handle corner cases and make formatting easier
        # nstep = 0
        # time_ps = 0.0
        # temp = 0.0
        # press = 0.0
        # etot = 0.0
        # ektot = 0.0
        # eptot = 0.0
        # ebond = 0.0
        # eangle = 0.0
        # edihed = 0.0
        # e14nb = 0.0
        # e14eel = 0.0
        # evdw = 0.0
        # eel = 0.0
        # ehbond = 0.0
        # erest = 0.0
        # dvdln = 0.0
        # ekcmt = 0.0
        # virial = 0.0
        # volume = 0.0
        # density = 0.0
        # ewald_err = 0.0

        nrg_comps = nrg_fn(state.position, ff, nbr_list, debug=True)
        eptot = nrg_fn(state.position, ff, nbr_list)
        ektot = quantity.kinetic_energy(momentum=state.momentum, mass=state.mass)
        temp = quantity.temperature(momentum=state.momentum, mass=state.mass)/kB
        nstep = (i + 1) * output_freq
        time_ps = config.t + (nstep * config.dt) # TODO probably need to also add value for restarts

        # out_str = f"""\
        #  NSTEP = {nstep:>8d}   TIME(PS) = {time_ps:>11.3f}  TEMP(K) = {temp:>8.2f}  PRESS = {press:>8.1f}
        #  Etot   = {etot:>13.4f}  EKtot   = {ektot:>12.4f}  EPtot      = {eptot:>13.4f}
        #  BOND   = {ebond:>12.4f}  ANGLE   = {eangle:>12.4f}  DIHED      = {edihed:>12.4f}
        #  1-4 NB = {e14nb:>12.4f}  1-4 EEL = {e14eel:>12.4f}  VDWAALS    = {evdw:>13.4f}
        #  EELEC  = {eel:>13.4f}  EHBOND  = {ehbond:>12.4f}  RESTRAINT  = {erest:>12.4f}
        #  DV/DL  = {dvdln:>12.4f}
        #  EKCMT  = {ekcmt:>12.4f}  VIRIAL  = {virial:>12.4f}  VOLUME     = {volume:>13.4f}
        #                                                      Density    = {density:>13.4f}
        #  Ewald error estimate:   {ewald_err:>10.4E}"""

        # TODO add conditional to change between amber and omm style reporting with units
        if nstep % config.ntpr == 0:
            print(format_mdout_block(
                nstep=nstep,
                time_ps=time_ps,
                temp=temp,
                #press=None,
                etot=eptot+ektot,
                ektot=ektot,
                eptot=eptot,
                ebond=nrg_comps["bond_pot"],
                eangle=nrg_comps["angle_pot"],
                edihed=nrg_comps["torsion_pot"],
                e14nb=nrg_comps["lj_14_pot"],
                e14eel=nrg_comps["coul_14_pot"],
                evdw=nrg_comps["lj_pot"],
                eel=nrg_comps["coul_pot"]
                #ebond=0.05,
                #evdw=800.123,
                #eel=-7200.55,
                #volume=30000.0,
                #density=0.9,
                #ewald_err=None
            ))

            temp_vals[i+1] = temp
            etot_vals[i+1] = eptot+ektot
            ektot_vals[i+1] = ektot
            eptot_vals[i+1] = eptot
            ebond_vals[i+1] = nrg_comps["bond_pot"]
            eangle_vals[i+1] = nrg_comps["angle_pot"]
            edihed_vals[i+1] = nrg_comps["torsion_pot"]
            e14nb_vals[i+1] = nrg_comps["lj_14_pot"]
            e14eel_vals[i+1] = nrg_comps["coul_14_pot"]
            evdw_vals[i+1] = nrg_comps["lj_pot"]
            eel_vals[i+1] = nrg_comps["coul_pot"]

            if fluctuating_charge == "FFQ":
                _nrg, ffq_chgs = nrg_fn(state.position, ff, nbr_list, return_charges=True)
                print(f"FFQ modified charges {ffq_chgs[:ff.solute_cut]}\n")

        # TODO add running averages with ntave

        # TODO add crd with ntwx
        if nstep % config.ntwx == 0:
            mdcrd.add_time(time_ps)

            mdcrd.add_coordinates(state.position * 10.0)
            #mdcrd.add_cell_lengths_angles(lengths=box_vectors, angles=cell_angles)

            # TODO this adds empty box frames to the end, i think something
            # else needs to be done to synchronize the crd/box frames
            if config.ntp == 1:
                print("current sim box for npt", state.box)
                mdcrd.add_cell_lengths_angles(lengths=state.box, angles=cell_angles)
            elif config.ntb == 1:
                mdcrd.add_cell_lengths_angles(lengths=box_vectors, angles=cell_angles)

        # TODO add vels with ntwv

        # TODO add frcs with ntwf

        # TODO add mden/temp with ntwe - review note about being synchronized with coordinates in amber manual

        # write restart
        if nstep % config.ntwr == 0:
            box_flag = config.ntb == 1
            ncrst = pmd.amber.netcdffiles.NetCDFRestart.open_new(os.path.join(out_dir, 'ncrst'), ff.atom_count, box=box_flag, vels=True)
            ncrst.coordinates  = state.position * 10.0
            ncrst.velocities = state.velocity * 10.0
            # a,b,c are the vectors and a,b,g the box angles
            if config.ntp == 1:
                print("current sim box for npt", state.box)
                ncrst.cell_lengths = state.box
                #a,b,c = state.box
                #parm.box = [a, b, c, 90.0, 90.0, 90.0]
                #parm.box = [a, b, c, alpha, beta, gamma]
            elif config.ntb == 1:
                ncrst.cell_lengths = box_vectors

            #ncrst.write(parm)
            # TODO might need to set time as well
            ncrst.close()

        # testing CM motion removal
        # total_mass = jnp.sum(state.mass)
        # jax.debug.print("center of mass {}", jnp.sum(state.position * state.mass, axis=0) / total_mass)

    if config.ntwx != 0: # coordinate trajectory
        #print("cell len angles", mdcrd.cell_lengths_angles, mdcrd.cell_lengths_angles.shape)
        mdcrd.close()

    # TODO write final restart regardless of ntwr
    box_flag = config.ntb == 1
    ncrst = pmd.amber.netcdffiles.NetCDFRestart.open_new(os.path.join(out_dir, 'ncrst'), ff.atom_count, box=box_flag, vels=True)
    ncrst.coordinates  = state.position * 10.0
    ncrst.velocities = state.velocity * 10.0
    if config.ntp == 1:
        print("current sim box for npt", state.box)
        ncrst.cell_lengths = state.box
    elif config.ntb == 1:
        ncrst.cell_lengths = box_vectors

    md_end_time = time.time()

    print(f"Overall MD run of {nstep} steps took {md_end_time-md_start_time} seconds")

    # print averages and rms for all timesteps
    # TODO in amber, is this just the averages of the steps where you do output
    # or is it for every step? might need to change it
    print(f"A V E R A G E S   O V E R  {nstep} S T E P S\n")
    print(format_mdout_block(
        nstep=nstep,
        time_ps=time_ps,
        temp=jnp.mean(temp_vals),
        etot=jnp.mean(etot_vals),
        ektot=jnp.mean(ektot_vals),
        eptot=jnp.mean(eptot_vals),
        ebond=jnp.mean(ebond_vals),
        eangle=jnp.mean(eangle_vals),
        edihed=jnp.mean(edihed_vals),
        e14nb=jnp.mean(e14nb_vals),
        e14eel=jnp.mean(e14eel_vals),
        evdw=jnp.mean(evdw_vals),
        eel=jnp.mean(eel_vals)
    ))

    def rms(vals):
        return jnp.sqrt(jnp.mean(vals**2))

    print("R M S  F L U C T U A T I O N S\n")
    print(format_mdout_block(
        nstep=nstep,
        time_ps=time_ps,
        temp=rms(temp_vals - jnp.mean(temp_vals)),
        etot=rms(etot_vals - jnp.mean(etot_vals)),
        ektot=rms(ektot_vals - jnp.mean(ektot_vals)),
        eptot=rms(eptot_vals - jnp.mean(eptot_vals)),
        ebond=rms(ebond_vals - jnp.mean(ebond_vals)),
        eangle=rms(eangle_vals - jnp.mean(eangle_vals)),
        edihed=rms(edihed_vals - jnp.mean(edihed_vals)),
        e14nb=rms(e14nb_vals - jnp.mean(e14nb_vals)),
        e14eel=rms(e14eel_vals - jnp.mean(e14eel_vals)),
        evdw=rms(evdw_vals - jnp.mean(evdw_vals)),
        eel=rms(eel_vals - jnp.mean(eel_vals))
    ))


    # custom force parser e.g. ###############################
    # def string_to_function(expression):
    #     def function(x):
    #         return eval(expression)
    #     return function

    # my_function = string_to_function("x**2 + 3 * x - 1")
    # result = my_function(2)
    # print(result)  # Output: 9

    # from https://stackoverflow.com/questions/74140788/how-to-wrap-a-numpy-function-to-make-it-work-with-jax-numpy
    # not the best way to do things but a simple example assuming there's no automated way
    # also a good reference for self documenting functions https://github.com/jax-ml/jax/discussions/13404
    # def function_np(x):
    # return np.maximum(0, x)

    # function_np_str = inspect.getsource(function_np) # getting the code as a string
    # function_jnp_str = re.sub(r"np", "jnp", function_code) #replacing all the 'np' with 'jnp'
    # # The line below creates a function defined in the 'jnp_function_str' string - which uses jnp instead of numpy
    # exec(jnp_activation_str) 


    # post analysis if necessary ###############################

    # output, logging, trajectories ###############################

    return

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                    prog='JAX-AMBER Interface',
                    description='Limited parser for AMBER MD input files')

    parser.add_argument('--mdin')
    parser.add_argument('--inpcrd')
    parser.add_argument('--prmtop')
    parser.add_argument('--ffq_file')
    parser.add_argument('--out_dir')
    parser.add_argument('--fluctuating_charge')

    # TODO this is mostly a placeholder until something is set up to convert
    # all the fields of the mdin class to valid arguments
    parser.add_argument('--nstlim')
    parser.add_argument('--ntpr')
    parser.add_argument('--ntwr')
    parser.add_argument('--ntwx')
    parser.add_argument('--ntb')

    args = parser.parse_args()
    print("Parsed args:", args)

    opt_dict = {}
    if args.nstlim != None:
        opt_dict['nstlim'] = int(args.nstlim)
    if args.ntpr != None:
        opt_dict['ntpr'] = int(args.ntpr)
    if args.ntwr != None:
        opt_dict['ntwr'] = int(args.ntwr)
    if args.ntwx != None:
        opt_dict['ntwx'] = int(args.ntwx)
    if args.ntb != None:
        opt_dict['ntb'] = int(args.ntb)

    print("opt_dict", opt_dict)

    test_data = amber_engine(mdin=args.mdin, inpcrd=args.inpcrd, prmtop=args.prmtop, ffq_file=args.ffq_file,
                            opt_dict=opt_dict, out_dir=args.out_dir, fluctuating_charge=args.fluctuating_charge)

    sys.exit()
    # in this case
    # export JAX_ENABLE_X64=True
    # inpcrd = "benzene.inpcrd"
    # prmtop = "benzene.prmtop"

    # pysander ketoprofen test ################################################################################
    # import sander
    # import parmed

    # # Load system
    # structure = parmed.load_file("/mnt/home/betanc18/JAX-ParamOpt/Datasets/ffq_nrg_drift_test/prmtop", "/mnt/home/betanc18/JAX-ParamOpt/Datasets/ffq_nrg_drift_test/7md.ncrst")

    # # Prepare sander input
    # sander.setup(
    #     prmtop=structure,
    #     coordinates=structure.positions,
    #     box=structure.box,
    #     cut=9.0,
    #     ntb=1,        # 1 = constant volume
    #     igb=0,        # 0 = explicit solvent
    #     ntpr=1        # print energies
    # )

    # # Get decomposed energy
    # energy = sander.energy_decomposition()
    # sander.cleanup()

    # # Print components
    # for key, value in energy.items():
    #     print(f"{key:10} = {value:15.6f}")

    # sys.exit()

    # quick npt test with benzene and a decharging transformation ###############################################
    # inpcrd = "benzene_solvated.inpcrd"
    # prmtop = "benzene_solvated.prmtop"
    # mdin = "/mnt/home/betanc18/JAX-ParamOpt/Datasets/test_fe/solvated/in_files/equil_0_decharged.in"
    # test_data = amber_engine(mdin=mdin, inpcrd=inpcrd, prmtop=prmtop)

    # TODO switch between kj/kcal programatically by setting an energy conversion variable
    # also add this for velocities and other outputs so this can be set by user
    # conversion_factor = 4.184

    # NVE test with benzene_solv same as job_78.out
    # inpcrd = "/mnt/home/betanc18/jax-md/jax_md/amber/data/benzene_solv.inpcrd"
    # prmtop = "/mnt/home/betanc18/jax-md/jax_md/amber/data/benzene_solv.prmtop"
    # #opt_dict = {"ntpr":100000, "ntwr":100000, "ntwx":100000, "nstlim":1000000, "ntb":1, "cut":10.0}
    # opt_dict = {"ntpr":10000, "ntwr":10000, "ntwx":10000, "nstlim":1000000, "ntb":1, "cut":10.0}
    # #opt_dict = {"ntpr":100000, "ntwr":100000, "ntwx":100000, "nstlim":1000000, "ntb":0, "cut":10.0}
    # test_data = amber_engine(inpcrd=inpcrd, prmtop=prmtop, opt_dict=opt_dict)
    # sys.exit()

    # NVE test with ramp in solvent for PME #######################################################################
    # inpcrd = "/mnt/home/betanc18/jax-md/jax_md/amber/data/ramp1/RAMP1_solv.inpcrd"
    # prmtop = "/mnt/home/betanc18/jax-md/jax_md/amber/data/ramp1/RAMP1_solv.prmtop"
    # #opt_dict = {"ntpr":10000, "ntwr":10000, "ntwx":10000, "nstlim":3000000, "ntb":1}
    # opt_dict = {"ntpr":1000, "ntwr":1000, "ntwx":1000, "nstlim":100000, "ntb":1}
    # test_data = amber_engine(inpcrd=inpcrd, prmtop=prmtop, opt_dict=opt_dict, out_dir="/mnt/home/betanc18/JAX-ParamOpt/Datasets/ffq_nrg_drift_test/md_files")
    # sys.exit()

    # gaff test with ketoprofen from freesolv #######################################################################
    # inpcrdFile = "/mnt/home/betanc18/JAX-ParamOpt/Datasets/ffq_nrg_drift_test/7md.ncrst"
    # prmtopFile = "/mnt/home/betanc18/JAX-ParamOpt/Datasets/ffq_nrg_drift_test/prmtop"
    # inpcrd = app.AmberInpcrdFile(inpcrdFile)
    # prmtop = app.AmberPrmtopFile(prmtopFile)
    # system = prmtop.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1.0*omm.unit.nanometer, removeCMMotion=False, rigidWater=False)

    # for i, f in enumerate(system.getForces()):
    #     f.setForceGroup(i)

    # platform = omm.Platform.getPlatformByName('CUDA')
    # properties = {'Precision': 'double'}
    # integrator = omm.VerletIntegrator(1e-3*omm.unit.picoseconds)
    # simulation = omm.app.Simulation(prmtop.topology, system, integrator, platform=platform, platformProperties=properties)
    # simulation.context.setPositions(inpcrd.positions)
    # simulation.context.setVelocities(inpcrd.velocities)
    # if inpcrd.boxVectors is not None:
    #     simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

    # frcsum = 0
    # for i, f in enumerate(system.getForces()):
    #     state = simulation.context.getState(getEnergy=True, getForces=True, groups={i})
    #     print(f.getName(), (state.getPotentialEnergy()._value))
    #     #print(f.getName(), "Uses PBCs", f.usesPeriodicBoundaryConditions())
    #     frcsum = frcsum + state.getPotentialEnergy()._value
    # print("OpenMM Potential Energy:", frcsum, sep="")
    # print("OpenMM Kinetic Energy:", state.getKineticEnergy()._value)
    # #print("OpenMM Box Vectors:", system.getDefaultPeriodicBoxVectors())
    # #print("OpenMM Forces:", state.getForces()._value[:10])

    # simulation.reporters.append(omm.app.StateDataReporter(sys.stdout, 100000, step=True,
    #     potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True))

    # start_time = time.time()
    # simulation.step(1000000)
    # end_time = time.time()
    # print("OMM elapsed time:", end_time-start_time)
    # sys.exit()

    # initial ffq test
    # mdin = "/mnt/home/betanc18/JAX-ParamOpt/Datasets/ffq_nrg_drift_test/7NVE.in" # just for example
    # inpcrd = "/mnt/home/betanc18/jax-md/jax_md/amber/data/planar_benzene/st.ncrst"
    # prmtop = "/mnt/home/betanc18/jax-md/jax_md/amber/data/planar_benzene/prmtop"
    # ffq_file = "/mnt/home/betanc18/jax-md/jax_md/amber/data/planar_benzene/ffqparam_opt.dat"
    # out_dir = "/mnt/home/betanc18/JAX-ParamOpt/Datasets/ffq_nrg_drift_test/md_files"
    # opt_dict = {"ntpr":10000, "ntwr":10000, "ntwx":10000, "nstlim":100000, "ntb":0}
    # test_data = amber_engine(mdin=mdin, inpcrd=inpcrd, prmtop=prmtop, ffq_file=ffq_file, opt_dict=opt_dict, out_dir=out_dir, fluctuating_charge="FFQ")
    # sys.exit()

    # gaff vs ffq vs lrch energy drift test with ketoprofen from freesolv #######################################################
    # timing here is 780 vs 232 seconds for 1ns of simulation, seems unusual
    inpcrd = "/mnt/home/betanc18/JAX-ParamOpt/Datasets/ffq_nrg_drift_test/7md.ncrst"
    prmtop = "/mnt/home/betanc18/JAX-ParamOpt/Datasets/ffq_nrg_drift_test/prmtop"
    mdin = "/mnt/home/betanc18/JAX-ParamOpt/Datasets/ffq_nrg_drift_test/7NVE.in"
    ffq_file = "/mnt/home/betanc18/JAX-ParamOpt/Datasets/ffq_nrg_drift_test/ffqparam_opt.dat"
    out_dir = "/mnt/home/betanc18/JAX-ParamOpt/Datasets/ffq_nrg_drift_test/md_files"
    opt_dict = {"ntpr":100000, "ntwr":100000, "ntwx":100000, "nstlim":1000000, "ntb":1}
    # opt_dict = {"ntpr":10000, "ntwr":10000, "ntwx":10000, "nstlim":1000000, "skinnb":0.0}
    # opt_dict = {"ntpr":10000, "ntwr":10000, "ntwx":10000, "nstlim":1000000, "ntb":0}
    #test_data = amber_engine(mdin=mdin, inpcrd=inpcrd, prmtop=prmtop, ffq_file=ffq_file, opt_dict=opt_dict, out_dir=out_dir, fluctuating_charge="FFQ")
    test_data = amber_engine(mdin=mdin, inpcrd=inpcrd, prmtop=prmtop, ffq_file=ffq_file, opt_dict=opt_dict, out_dir=out_dir, fluctuating_charge="LRCH")
    #test_data = amber_engine(mdin=mdin, inpcrd=inpcrd, prmtop=prmtop, ffq_file=None, opt_dict=opt_dict, out_dir=out_dir)
    sys.exit()