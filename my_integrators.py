from simtk import openmm
from simtk import unit


class MiddleNoseHooverChainIntegrator(openmm.CustomIntegrator):
    def __init__(self, dt, temp, tau, Nd):
        super().__init__(dt)
        kT = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA*temp
        self.addGlobalVariable('kT', kT)
        self.addGlobalVariable('NkT', Nd*kT)
        self.addGlobalVariable('v1', 0.0)
        self.addGlobalVariable('v2', 0.0)
        self.addGlobalVariable('Q1', Nd*kT*tau**2)
        self.addGlobalVariable('Q2', kT*tau**2)
        self.addGlobalVariable('twoK', 0.0)

        self.addComputePerDof('v', 'v+0.5*dt*f/m')
        self.addComputePerDof('x', 'x+0.5*dt*v')
        self.addComputeSum('twoK', 'm*v*v')
        self.addComputeGlobal('v1', 'v1 + 0.5*dt*(twoK - NkT)/Q1')
        self.addComputePerDof('v', 'v*exp(-0.5*dt*v1)')
        self.addComputeGlobal('v2', 'v2 + 0.5*dt*(Q1*v1*v1 - kT)/Q2')
        self.addComputeGlobal('v1', 'v1*exp(-dt*v2)')
        self.addComputeGlobal('v2', 'v2 + 0.5*dt*(Q1*v1*v1 - kT)/Q2')
        self.addComputePerDof('v', 'v*exp(-0.5*dt*v1)')
        self.addComputeSum('twoK', 'm*v*v')
        self.addComputeGlobal('v1', 'v1 + 0.5*dt*(twoK - NkT)/Q1')
        self.addComputePerDof('x', 'x+0.5*dt*v')
        self.addComputePerDof('v', 'v+0.5*dt*f/m')


class MiddleNoseHooverLangevinIntegrator(openmm.CustomIntegrator):
    def __init__(self, dt, temp, gamma, tau, Nd):
        super().__init__(dt)
        kT = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA*temp
        self.addGlobalVariable('friction', gamma)
        self.addGlobalVariable('kT', kT)
        self.addGlobalVariable('NkT', Nd*kT)
        self.addGlobalVariable('xi', 0.0)
        self.addGlobalVariable('Q', Nd*kT*tau**2)
        self.addGlobalVariable('twoK', 0.0)

        self.addComputePerDof('v', 'v+0.5*dt*f/m')
        self.addComputePerDof('x', 'x+0.5*dt*v')
        self.addComputeSum('twoK', 'm*v*v')
        self.addComputeGlobal('xi', 'xi + 0.5*dt*(twoK - NkT)/Q')
        self.addComputePerDof('v', 'v*exp(-0.5*dt*xi)')
        self.addComputeGlobal('xi', 'z*xi + sqrt(kT*(1 - z*z)/Q)*gaussian; z = exp(-dt*friction)')
        self.addComputePerDof('v', 'v*exp(-0.5*dt*xi)')
        self.addComputeSum('twoK', 'm*v*v')
        self.addComputeGlobal('xi', 'xi + 0.5*dt*(twoK - NkT)/Q')
        self.addComputePerDof('x', 'x+0.5*dt*v')
        self.addComputePerDof('v', 'v+0.5*dt*f/m')

