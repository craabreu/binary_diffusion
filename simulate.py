import argparse
import atomsmm

from sys import stdout
from simtk import openmm
from simtk import unit
from simtk.openmm import app

import my_integrators

parser = argparse.ArgumentParser()
parser.add_argument('-file', dest='file', required = True, help='Base name for PDB and XML files')
parser.add_argument('-steps', dest='nsteps', required = True, type=int, help='Number of MD steps')
parser.add_argument('-platform', dest='platform', required = True, choices=['CPU', 'CUDA', 'OpenCL'], help='Simulation platform')
parser.add_argument('-temp', dest='temp', type=float, help='System temperature in K', default=298.15)
parser.add_argument('-press', dest='press', type=float, help='System pressure in bar', default=1.01325)
args = parser.parse_args()

dt = 2*unit.femtoseconds
temp = args.temp*unit.kelvin
press = args.press*unit.bar
rcut = 13*unit.angstroms
tau = 100*dt
gamma = 0.1/unit.femtoseconds
seed = 98745
reportInterval = 100
rswitch = rcut - 1*unit.angstrom

platform = openmm.Platform.getPlatformByName(args.platform)
properties = dict(Precision='mixed') if args.platform == 'CUDA' else dict()

pdb = app.PDBFile(f'{args.file}.pdb')
forcefield = app.ForceField(f'{args.file}.xml')

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

Nd = 3*(openmm_system.getNumParticles() - 1)
# integrator = my_integrators.MiddleNoseHooverChainIntegrator(dt, temp, tau, Nd)
integrator = my_integrators.MiddleNoseHooverLangevinIntegrator(dt, temp, gamma, tau, Nd)
simulation = openmm.app.Simulation(pdb.topology, openmm_system, integrator, platform, properties)
simulation.context.setPositions(pdb.positions)
simulation.context.setVelocitiesToTemperature(temp, seed)

dataReporter = atomsmm.ExtendedStateDataReporter(stdout, reportInterval, separator=',',
    step=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
    temperature=True, speed=True, extraFile=f'{args.file}.csv')
comReporter = atomsmm.CenterOfMassReporter(f'{args.file}.xyz', reportInterval)
velReporter = atomsmm.CenterOfMassReporter(f'{args.file}.vel', reportInterval, velocities=True)

simulation.reporters += [dataReporter, comReporter, velReporter]
simulation.step(args.nsteps)

