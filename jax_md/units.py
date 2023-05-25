import jax.numpy as jnp
from typing import Dict
from jax_md import util

"""Defines a units system and returns a dictionary of conversion factors.
   Units are defined similar to https://docs.lammps.org/units.html
"""

# Types
f64 = util.f64

# CODATA Recommended Values of the Fundamental Physical Constants: 2014
# http://arxiv.org/pdf/1507.07956.pdf
# https://wiki.fysik.dtu.dk/ase/_modules/ase/units.html#create_units

constants_CONDATA_2014 = {'_c': 299792458.,  # speed of light, m/s
                          '_mu0': 4.0e-7 * jnp.pi,  # permeability of vacuum
                          '_Grav': 6.67408e-11,  # gravitational constant
                          '_hplanck': 6.626070040e-34,  # Planck constant, J s
                          '_e': 1.6021766208e-19,  # elementary charge
                          '_me': 9.10938356e-31,  # electron mass
                          '_mp': 1.672621898e-27,  # proton mass
                          '_Nav': 6.022140857e23,  # Avogadro number
                          '_k': 1.38064852e-23,  # Boltzmann constant, J/K
                          '_amu': 1.660539040e-27}  # atomic mass unit, kg


def metal_unit_system(constants: Dict = constants_CONDATA_2014):
    """Metal unit system

  Args:
    constants: Dictionary of fundamental constants
  
  Returns:
    Dictionary of conversion factors
  """

    Angstrom = 1  # Default length scale
    eV = 1  # Default Energy scale
    amu = 1  # Default Mass scale
    charge = 1  # Default charge

    Ang_conv_factor = 1e-10  # Meter
    second = jnp.sqrt(eV * constants['_e'] / (constants['_amu'] * Ang_conv_factor * Ang_conv_factor))  # Kg, m, J
    picosecond = 1e-12 * second  # picosecond

    kB = constants['_k'] / constants['_e']  # Boltzmann constant in eV/K
    # Kelvin = 1 / kB

    pascal = (eV * Ang_conv_factor * Ang_conv_factor * Ang_conv_factor / constants[
        '_e'])  # 1 / pressure_conv_factor # J/m^3 i.e pascal
    # eV/A^{3} -> bar
    bar = 1e5 * pascal

    metal_units = {'mass': f64(amu),
                   'distance': f64(Angstrom),
                   'time': f64(picosecond),
                   'energy': f64(eV),
                   'velocity': f64(Angstrom / picosecond),
                   'force': f64(eV / Angstrom),
                   'torque ': f64(eV),
                   'temperature': f64(kB),  # JAX MD uses kT
                   'pressure': f64(bar),
                   'charge ': f64(charge),
                   'electric field': f64(charge * Angstrom)}

    return metal_units


def real_unit_system(constants: Dict = constants_CONDATA_2014):
    """Real unit system

  Args:
    constants: Dictionary of fundamental constants
  
  Returns:
    Dictionary of conversion factors
  """

    Angstrom = 1  # Default length scale
    Kcal_per_mol = 1  # Default Energy scale
    grams_per_mol = 1  # Default Mass scale
    charge = 1  # Default charge

    Ang_conv_factor = 1e-10  # Meter
    second = jnp.sqrt(Kcal_per_mol / (
                constants['_amu'] * Ang_conv_factor * Ang_conv_factor * constants['_Nav'] * 1e-3 / 4.184))  # Kg, m, J
    femtosecond = 1e-15 * second  # femtosecond

    kB = constants['_k'] * constants['_Nav'] / 4.184 / 1e3  # Boltzmann constant in kcal/mol K
    # Kelvin = 1 / kB

    pascal = (Ang_conv_factor * Ang_conv_factor * Ang_conv_factor) * (
                constants['_Nav'] * 1e-3 / 4.184)  # 1 / pressure_conv_factor # J/m^3 i.e pascal
    # kcal/mol A^{3} -> atm
    atm = 101325 * pascal

    real_units = {'mass': f64(grams_per_mol),
                  'distance': f64(Angstrom),
                  'time': f64(femtosecond),
                  'energy': f64(Kcal_per_mol),
                  'velocity': f64(Angstrom / femtosecond),
                  'force': f64(Kcal_per_mol / Angstrom),
                  'torque ': f64(Kcal_per_mol),
                  'temperature': f64(kB),  # JAX MD uses kT
                  'pressure': f64(atm),
                  'charge ': f64(charge),
                  'electric field': f64(charge * Angstrom)}

    return real_units
