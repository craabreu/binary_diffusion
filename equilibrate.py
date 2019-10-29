import argparse
import atomsmm
import numpy as np
import pandas as pd

from sys import stdout
from simtk import openmm
from simtk import unit
from simtk.openmm import app

parser = argparse.ArgumentParser()
parser.add_argument('-file', dest='file', required = True, help='Base name for PDB and XML files')
parser.add_argument('-steps', dest='nsteps', required = True, help='Number of MD steps')
parser.add_argument('-platform', dest='platform', required = True, choices=['CPU', 'CUDA', 'OpenCL'], help='Simulation platform')
parser.add_argument('-temp', dest='temp', type=float, help='System temperature in K', default=298.15)
parser.add_argument('-press', dest='press', type=float, help='System pressure in bar', default=1.01325)
args = parser.parse_args()
basename = args.file
nsteps = int(args.nsteps)

dt = 2*unit.femtoseconds
temp = args.temp*unit.kelvin
press = args.press*unit.bar
rcut = 12*unit.angstroms
rswitch = 11*unit.angstroms
seed = 98745
platform_name = args.platform
reportInterval = 100

platform = openmm.Platform.getPlatformByName(platform_name)
properties = dict(Precision='mixed') if platform_name == 'CUDA' else dict()

pdb = app.PDBFile('{}_raw.pdb'.format(basename))

forcefield = app.ForceField('{}.xml'.format(basename))

openmm_system = forcefield.createSystem(pdb.topology,
                                        nonbondedMethod=openmm.app.PME,
                                        nonbondedCutoff=rcut,
                                        rigidWater=False,
                                        constraints=None,
                                        removeCMMotion=False)

nbforce = openmm_system.getForce(atomsmm.findNonbondedForce(openmm_system))
nbforce.setUseSwitchingFunction(True)
nbforce.setSwitchingDistance(rswitch)
nbforce.setUseDispersionCorrection(True)
openmm_system.addForce(openmm.MonteCarloBarostat(press, temp, 25))

integrator = openmm.LangevinIntegrator(temp, 0.1/unit.femtoseconds, dt)
simulation = openmm.app.Simulation(pdb.topology, openmm_system, integrator, platform, properties)
simulation.context.setPositions(pdb.positions)
simulation.context.setVelocitiesToTemperature(temp, seed)

reporter = atomsmm.ExtendedStateDataReporter(stdout, reportInterval, separator=',', step=True,
    potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True,
    volume=True, density=True, speed=True, extraFile='properties.csv')
simulation.reporters.append(reporter)
simulation.step(nsteps)

df = pd.read_csv('properties.csv')
ndata = nsteps//(2*reportInterval)
Lbox = [df['Box Volume (nm^3)'].tail(ndata).mean()**(1/3)*10]*3

state = simulation.context.getState(getPositions=True)
coords = state.getPositions(asNumpy=True).value_in_unit(unit.angstroms)

out = open('box.temp', 'w')
print('box lengths {} {} {}'.format(*Lbox), file=out)
print('reset xyz', file=out)
print('build', file=out)
print(pdb.topology.getNumAtoms(), file=out)
for atom, coord in zip(pdb.topology.atoms(), coords):
    print(atom.name, *coord, file=out)
out.close()
