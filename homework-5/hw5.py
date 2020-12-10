# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:58:05 2020

@author: gusfl
"""

import numpy as np

dY = 100e-6 # m
eps_g = 0.57 # porosity
n_Brugg = -0.5

d_part = 0.5e-6
r_p = 2e-6

T = 333.15 # K
P_1 = 101325 # Pa
P_2 = 100000 # Pa

R = 8.3145 # J/mol-K
F = 96485  # C/mol equiv

# Species order: O2, N2, H2O_v
X_k_1 = np.array([0.21, 0.79, 0.0])
X_k_2 = np.array([0.16, 0.80, 0.04])

mu = 2.08e-5 #kg/m-s

D_k = np.array([2.438e-5, 2.798e-5, 1.9e-5]) #m2/s


##############


# State variables for node 1:
state1 = {'X_k':X_k_1, 'P':P_1, 'T':T}
# State variables for node 2:
state2 = {'X_k':X_k_2, 'P':P_2, 'T':T}

# Geometric and microstructure parameters:
geom = {'eps_g':eps_g, 'n_Brugg':n_Brugg, 'd_part':d_part, 'dY':dY}
# Gas properties
gas_props = {'D_k':D_k, 'mu':mu}



################

# To access a dictionary value:
print(gas_props['mu'])
print(state1['X_k'])
X_sum = sum(state1['X_k'])
print('The mole fractions sum to',X_sum)


#################


# The Catalyst layer is the control volume in the following formulation.

def pemfc_gas_transport(state1, state2, geom, gas_props):
    
    N_k = np.zeros_like(state1['X_k'])
    
    #Concentrations
    C_k_1 = state1['X_k']*state1['P']/R/state1['T']
    C_k_2 = state2['X_k']*state2['P']/R/state2['T']
    
    X_k_int = (state1['X_k'] + state2['X_k'])/2
    C_int = sum((C_k_1+C_k_2)/2)
    
    #tortuosity factor from Bruggman 
    tau_fac = geom['eps_g']**n_Brugg 

    #effective eiffusion coeff
    D_k_eff =  geom['eps_g']**1.5 *gas_props['D_k']
   
    #Permiability fro Kozney-Carman
    K_m = (geom['eps_g']**3*geom['d_part']**2)/(72*tau_fac**2*(1-geom['eps_g'])**2) 
  
    # #Convection velocity
    V_conv_k = -K_m * (state2['P'] - state1['P'])/geom['dY']/gas_props['mu'] 

    # #Diffusion Velocity
    V_difn_k = -D_k_eff * (state2['X_k'] - state1['X_k'])/geom['dY']/X_k_int

    #total flux
    N_k = C_int*X_k_int*(V_conv_k + V_difn_k)
    
    
    return N_k


#####################

N_k_calc = pemfc_gas_transport(state1, state2, geom, gas_props)

from matplotlib import pyplot as plt
width = 0.35
N_k_check = np.array([0.19913, -0.007275, -0.11794]) #mol/m2/s

fig, ax = plt.subplots()

labels = ['O2', 'N2', 'H2O']
x = np.arange(len(labels))

ax.bar(x+width/2,N_k_check,width)
ax.bar(x-width/2,N_k_calc,width)
ax.legend(['DeCaluwe\'s Answer','My Answer'],frameon=False)

ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.set_ylabel('Diffusion coefficient (m$^2$/s)',fontsize=14)
ax.set_xlabel('Species',fontsize=14)

plt.show()



#########Part 2##############

C_elyte = 1100 #mol/m3

# Species order: Li+, PF6-, solvent
X_k_1 = np.array([0.03, 0.03, 0.94])
X_k_2 = np.array([0.06, 0.06, 0.88])

z_k = np.array([1., -1., 0.])

T = 318.  #K

dY = 30e-6

D_k = np.array([1.52e-10, 0.25e-10, 1e-12])

phi_1 = 0.9
phi_2 = 0.5

d_part = 5e-6

eps_elyte = 0.23
n_brugg = -0.5


################

    
s1 = {'X_k':X_k_1,'phi':phi_1}
s2 = {'X_k':X_k_2,'phi':phi_2}

geom = {'dY':dY,'d_p':d_part,'eps_elyte':eps_elyte}

elyte_pars = {'D_k':D_k,'z_k':z_k,'C':C_elyte}



##################


def electrolyte_transport(state1, state2, geom, elyte_pars):
    N_k = np.zeros_like(state1['X_k'])
     
    #Concentrations
    C_k_1 = elyte_pars['C']*state1['X_k']
    C_k_2 = elyte_pars['C']*state2['X_k']
    
    #X_k_int = (state1['X_k'] + state2['X_k'])/2
    C_k_int = (C_k_1+C_k_2)/2
    
    D_k_eff = geom['eps_elyte']**1.5 * elyte_pars['D_k']
    
    D_k_mig = -D_k_eff*C_k_int*elyte_pars['z_k']*F/R/T
    
    N_k = -D_k_eff*(C_k_2-C_k_1)/geom['dY'] + D_k_mig*(state2['phi']-state1['phi'])/geom['dY']
     
    return N_k

###############


dPhi = np.linspace(0,1.1,25)
currents = np.zeros_like(dPhi)
N_k = np.zeros((len(dPhi), len(z_k)))


for j, phi in enumerate(dPhi):
    s2['phi'] = phi
    N_k[j,:] = electrolyte_transport(s1,s2, geom, elyte_pars)
    currents[j] = np.dot(z_k,N_k[j,:])*F
    
    
##################

current_check = np.array([100.,  95.,  90.,  85.,  80.,  75.,  69.,
                          64.,  59.,  54.,  49.,  43.,  38.,  33.,
                          28.,  23.,  17.,  12.,   7.,   2.,  -3.,
                          -9.,    -14., -19., -24.])

plt.plot(dPhi, currents, 'k')
plt.plot(dPhi, current_check, 'ro', markerfacecolor=None)
plt.plot(dPhi, np.zeros_like(dPhi),'--',color='0.5')
plt.xlabel('Electric potential difference (V)',fontsize=14)
plt.ylabel('Current density (A/m$^2$)',fontsize=14)

plt.show()

zero=np.interp(0, np.flip(currents), np.flip(dPhi))
print('Zero current at dPhi = ',zero)


plt.plot(dPhi, N_k[:,0],linewidth=2.5)
plt.plot(dPhi, N_k[:,1],linewidth=2.5)

plt.xlabel('Electric potential difference (V)',fontsize=14)
plt.ylabel('Molar flux (mol/m$^2$-s)',fontsize=14)
plt.legend(['Li$^+$','PF$_6^-$'],frameon=False,fontsize=14)
plt.plot([zero,zero],[N_k[-1,0],N_k[0,0]],'--',color='0.5')
plt.plot([0,1],[0,0],'--',color='0.5')

plt.show()


###################

eps_array = np.linspace(0.05,0.95,25)

currents = np.zeros_like(eps_array)
for j, eps in enumerate(eps_array):
    geom['eps_elyte'] = eps
    N_k = electrolyte_transport(s1, s2, geom, elyte_pars)
    currents[j] = np.dot(elyte_pars['z_k'],N_k)*F
    
plt.plot(eps_array, currents,'b',linewidth=2.5)
plt.xlabel('Electrolyte Volume Fraction', fontsize=14)
plt.ylabel('Current Density (A/m$^2$)',fontsize=14)


#################
C_site = 22325 #site concentrations [mol/m^3]

#Temperature 
T = 873  #[K]

#Charge coefficients
z_k = np.array([2., 0., 1.])

#diffusion coefficients [V^++, O^x, OH^+] [m^2/s]
D_k = np.array([1.28E-12, 0., 7.46E-11]) 

#mole fractions
X_k_1 = np.array([0.336,0.627,0.037])
X_k_2 = np.array([0.018,0.938,0.044])

#potentals
phi_1 = 1.1 #[V]
phi_2 = 0.0 #[V]

#membrane copnductivity
sigma_el = 0.001 #{A/V-m]

#node step
dY = 20E-6 #[m]

s1 = {'X_k':X_k_1, 'phi':phi_1, 'T':T}
s2 = {'X_k':X_k_2, 'phi':phi_2, 'T':T}

geom = { 'dY':dY}

ceramic_pars = {'z_k':z_k, 'D_k':D_k, 'C':C_site,'sigma_el':sigma_el }




##########################
def protonic_transport(state1, state2, geom, ceramic_pars):
    N_k = np.zeros_like(state1['X_k'])
    
   #Concentrations
    C_k_1 = ceramic_pars['C']*state1['X_k']
    C_k_2 = ceramic_pars['C']*state2['X_k']
    
    #X_k_int = (state1['X_k'] + state2['X_k'])/2
    C_k_int = (C_k_1+C_k_2)/2
    
    D_k_eff = ceramic_pars['D_k']
    
    D_k_mig = -D_k_eff*C_k_int*ceramic_pars['z_k']*F/R/T
    
    N_k = -D_k_eff*(C_k_2-C_k_1)/geom['dY'] + D_k_mig*(state2['phi']-state1['phi'])/geom['dY']
    
    i_ionic = np.dot(ceramic_pars['z_k'], N_k)*F
    
    i_electric = ceramic_pars['sigma_el']*(state2['phi']-state1['phi'])/geom['dY']
    

    current =  -i_electric + i_ionic
    
    return N_k, current



###########################
dPhi = np.linspace(0.0, 1.0, 100)
eta_Far = np.zeros_like(dPhi)
i_tot = np.zeros_like(dPhi)

for j, deltaPhi in enumerate(dPhi):
    s2['phi'] = s1['phi']-deltaPhi
    N_k, i_tot[j] = protonic_transport(s1, s2, geom, ceramic_pars)
    i_ion = np.dot(ceramic_pars['z_k'],N_k)*F
    
    eta_Far[j] = 100*i_ion/i_tot[j]
    
    
# Plot the results:
fig, ax = plt.subplots()
plt.plot(dPhi, i_tot,color='b')

# Create a 2nd y axis:
ax2 = ax.twinx()
ax2.plot(dPhi, eta_Far,'r')

# Formatting:
ax.set_xlabel('Electric Potential Difference ($\phi_1 - \phi_2$, V)', fontsize=14)

ax.tick_params(axis='y',color='r',labelcolor='b',labelsize=12)
ax.set_ylabel('Current density (A/m$^2$)',color='b',fontsize=14)
ax.tick_params(axis='y',color='r',labelcolor='b',labelsize=12)

ax2.set_ylabel('Faradaic Efficiency (%)',color='r',fontsize=14)
ax2.tick_params(axis='y',color='r',labelcolor='r',labelsize=12)
ax2.tick_params(axis='y',color='r',labelcolor='r',labelsize=12)
ax2.set_ylim((99,100))

plt.show()



##############################

dPhi = np.linspace(-0.01, 0.01, 20)
N_k = np.zeros((len(dPhi),len(state1['X_k'])))
i_tot = np.zeros_like(dPhi)

for j, deltaPhi in enumerate(dPhi):
    s2['phi'] = s1['phi']-deltaPhi
    N_k[j,:], i_tot[j] = protonic_transport(s1, s2, geom, ceramic_pars)

fig, ax = plt.subplots()
plt.plot(1000*dPhi, i_tot,color='b')

zero = 1000*np.interp(0,i_tot,dPhi)
plt.plot([zero,zero],[i_tot[0],i_tot[-1]],'--',color='0.5')

ax.set_ylabel('Current density (A/m$^2$)',color='b',fontsize=14)
ax.tick_params(axis='y',color='k',labelcolor='b',labelsize=12)
ax.tick_params(axis='y',color='k',labelcolor='b',labelsize=12)

ax.tick_params(axis='x',color='k',labelcolor='k',labelsize=12)
ax.set_xlabel('Electric Potential Difference ($\phi_1 - \phi_2$, mV)', fontsize=14)
# ax.set_xlim((-0.01,0.01))

ax2 = ax.twinx()
ax2.plot(1000*dPhi, N_k[:,0],'r.--')
ax2.plot(1000*dPhi, N_k[:,2],color='r')

ax2.set_ylabel('Species Flux (mol/m$^2$-s)',color='r',fontsize=14)
ax2.tick_params(axis='y',color='r',labelcolor='r',labelsize=12)

ax2.legend(['Vacancy','Proton'],fontsize=14,frameon=False)

plt.show()