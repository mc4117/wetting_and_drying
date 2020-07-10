#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 23:47:46 2019

@author: mc4117
"""

import numpy as np
import pylab as plt
import time
import datetime
import os

import thetis_adjoint as th_adj

mesh2d = th_adj.Mesh('mesh.msh')


tape = th_adj.get_working_tape()
print(len(tape.get_blocks()))

from matplotlib import rc

fac = (4*10**10)*(1.5*13800)

def control_plot(times, controls, filename, ylabel = r"$\eta_D$ [m]", subtract_optimal_controls = False):
    ''' Saves a plot of the controls '''
    endtime = times[-1]/60/60
    times = [time/60/60 for time in times[:-int(2*60*60/600)]]
    if subtract_optimal_controls:
        Y = [control-opt_control for control, opt_control in zip(controls, eta_bc_proj)]
    else:
        Y = [control for control in controls]
 
    scaling = 0.7
    rc('text', usetex = True)
    plt.figure(1, figsize = (scaling*7., scaling*4.))
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.3)
    plt.plot(times, Y, color = 'black')
    if not subtract_optimal_controls:
        plt.xlim([0, endtime])#, -0.1, elevation*1.2])
    else:
        plt.xlim([0, endtime])
    #plt.xticks(numpy.arange(0, times[-1]+1, 5))
    #plt.yticks(numpy.arange(14, basin_x_total/1000, 2))
    plt.ylabel(ylabel)
    plt.xlabel(r"Time [h]")
    plt.show()

# And create a callback that saves the control values at every optimisation iteration
eval_counter = [0]
max_error_list = []
functional_list = []

def eval_callback(functional_value, value):
    print(scale)
    eval_counter[0] += 1
    functional_list.append(functional_value)
    times = np.linspace(0, (24*3600)+1200, (24*6)+3)
    control_plot(times, value, "controls/controls_"+str(eval_counter[0])+".pdf")
    control_plot(times, value, "controls/controls_final.pdf")
    control_plot(times, value, "controls/controls_errors_"+str(eval_counter[0])+".pdf", subtract_optimal_controls=True, ylabel=r"$\eta_D - \eta_D^{\textrm{exact}}$ [m]")
    control_plot(times, value, "controls/controls_errors_final.pdf", subtract_optimal_controls=True, ylabel = r"$\eta_D - \eta_D^{\textrm{exact}}$ [m]")
    max_err = max([abs(control-opt_control) for control, opt_control in zip(value, eta_bc_proj)])
    print("Maximal control value error: ", max_err, "m")
    max_error_list.append(max_err)
    if max_err < 0.001:
        print("********************************** Control is 0.001m close to the solution found after %i iterations ************************" % eval_counter[0])



def forward(eta_bc_const, mesh2d):
    
    def heavyside_approx(H, alpha):
        return 0.5*(H/(th_adj.sqrt(H**2+alpha**2)))+0.5


    def update_forcings_hydrodynamics(t_new):
        #print(t_new)
        #print(((dirk22_factor)*(t_new-options.timestep)) +((1-dirk22_factor)*(t_new)))
        in_fn = ((1-dirk22_factor)*eta_bc_const[int(t_new/options.timestep)-1]) + (dirk22_factor*eta_bc_const[int(t_new/options.timestep)]) 
        eta_list.append(in_fn)       
            
        #print(int(t_new/options.timestep))
        

        in_c.assign(in_fn)

        if np.round(t_new%options.timestep, 2) == 0.00:
        
            elev1 = (solver_obj.fields.solution_2d[2])
            
            in_fn = eta_bc_const[int(t_new/options.timestep)]
            eta_list.append(in_fn)       
            

            in_c.assign(in_fn)
            
            wd_obs_arr.append(th_adj.project(heavyside_approx(- elev1 - h_static, wetting_alpha), P1_2d, annotate = False))

            wd_obs.assign(wd_obs_arr[int(t_new/dt)], annotate = False)

            # Record the simulated wetting and drying front
            wd_tmp = th_adj.project(heavyside_approx(- elev1 - h_static, wetting_alpha), P1_2d, annotate = False)
            wd.assign(wd_tmp, annotate = False)
        
            form = 0.5*th_adj.inner(wd - wd_obs_arr[int(t_new/dt)], wd - wd_obs_arr[int(t_new/dt)])*th_adj.dx
            

            #form = 0.5*th_adj.inner(wd, wd)*th_adj.dx
            #print(J_mc)
            #J_list.append(th_adj.assemble(dt*scale*form))

            J_list.append(th_adj.assemble(scale*options.timestep*form, annotate = False))
            #J_list.append(th_adj.assemble(options.timestep*form))
            

            

    wetting_and_drying = True
    ks = 0.025
    average_size = 200 * (10**(-6))
    t_end=24*3600.
    friction = 'manning'
    friction_coef = 0.025
    diffusivity = 0.15
    viscosity = 10**(-6)
    dt = 600
            
    # choose directory to output results
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    outputdir = 'outputs'+ st
 
    # export interval in seconds
    t_export = np.round(t_end/40,0)

    th_adj.print_output('Exporting to '+outputdir)

    #th_adj.RectangleMesh(12, 6, 13800, 7200)
    
    # define function spaces
    V = th_adj.FunctionSpace(mesh2d, 'CG', 2)
    P1_2d = th_adj.FunctionSpace(mesh2d, 'DG', 6)
    R = th_adj.FunctionSpace(mesh2d, 'DG', 0)
    
    x, y = th_adj.SpatialCoordinate(mesh2d)

    max_depth = 5
    basin_x = 13800

    h_static = (1.-x/basin_x)*max_depth

    bathymetry_2d = th_adj.Function(V)
    bathymetry_2d.project((1.-x/basin_x)*max_depth, annotate = False)
    
    #elev1 = th_adj.Function(P1_2d).interpolate(th_adj.Constant(0.0))
    # define parameters
    #    ksp = th_adj.Constant(3*average_size)
    
    # initial condition: assign non-zero velocity
    elev_init = th_adj.Constant(0.01)
    uv_init = th_adj.Constant((10**(-4), 0.))    
    
    
    wetting_alpha = 0.43
    
    wd_obs_arr =[(th_adj.project(heavyside_approx(- elev_init - h_static, wetting_alpha), P1_2d, annotate = False))]
    wd_obs = th_adj.project(wd_obs_arr[0], P1_2d, annotate = False)
    # Record the simulated wetting and drying front
    wd = th_adj.project(heavyside_approx(- elev_init - h_static, wetting_alpha), P1_2d, annotate = False)   
    eta_list = []
    J_list = []
    scale = th_adj.AdjFloat(1)


    th_adj.parameters['form_compiler']['quadrature_degree'] = 20
    th_adj.parameters['form_compiler']['cpp_optimize'] = True
    th_adj.parameters['form_compiler']['cpp_optimize_flags'] = '-O3 -ffast-math -march=native'
    th_adj.parameters['form_compiler']['optimize'] = True
        
    # set up solver 
    solver_obj = th_adj.solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    options = solver_obj.options 
    options.simulation_export_time = t_export
    options.simulation_end_time = t_end
    options.output_directory = outputdir
    options.norm_smoother = th_adj.Constant(0.43)
    options.check_volume_conservation_2d = True
    options.fields_to_export = ['uv_2d', 'elev_2d']
    options.solve_tracer = False
    options.use_lax_friedrichs_tracer = False
    if friction == 'nikuradse':
        options.quadratic_drag_coefficient = cfactor
    elif friction == 'manning':
        if friction_coef == 0:
                friction_coef = 0.02
        options.manning_drag_coefficient = th_adj.Constant(friction_coef)
    else:
        print('Undefined friction')

    # set horizontal diffusivity parameter
    options.horizontal_diffusivity = th_adj.Constant(diffusivity)
    if viscosity == None:
        print('no viscosity')
    else:
        options.horizontal_viscosity = th_adj.Constant(viscosity)

    # crank-nicholson used to integrate in time system of ODEs resulting from application of galerkin FEM
    options.timestepper_type = 'DIRK22'#CrankNicolson'
    dirk22_factor = (2 - th_adj.sqrt(2))/2
    #options.timestepper_options.implicitness_theta = 0.5
    options.use_wetting_and_drying = wetting_and_drying
    options.wetting_and_drying_alpha = th_adj.Constant(wetting_alpha)
  
    

    if not hasattr(options.timestepper_options, 'use_automatic_timestep'):
        options.timestep = dt

    
    # set boundary conditions

    in_constant = eta_bc_const[0]

    in_c = th_adj.Constant(0.0)
    in_c.assign(in_constant)

    #in_c.assign(in_constant)
    swe_bnd = {}
    swe_bnd[1] = {'elev': in_c}
   
    solver_obj.bnd_functions['shallow_water'] = swe_bnd
    


    solver_obj.assign_initial_conditions(uv = uv_init, elev= elev_init)
    #solver_obj.assign_initial_conditions(uv = uv_init)


    # User-defined output: moving bathymetry and eta_tilde
    wd_bathfile = th_adj.File(os.path.join(outputdir, 'moving_bath.pvd'))
    moving_bath = th_adj.Function(P1_2d, name="moving_bath")
    eta_tildefile = th_adj.File(os.path.join(outputdir, 'eta_tilde.pvd'))
    eta_tilde = th_adj.Function(P1_2d, name="eta_tilde")



    # user-specified export function
    def export_func():
        wd_bath_displacement = solver_obj.eq_sw.bathymetry_displacement_mass_term.wd_bathymetry_displacement
        eta = solver_obj.fields.elev_2d
        moving_bath.project(bathymetry_2d + wd_bath_displacement(eta))
        wd_bathfile.write(moving_bath)
        eta_tilde.project(eta+wd_bath_displacement(eta))
        eta_tildefile.write(eta_tilde)

    solver_obj.iterate(update_forcings=update_forcings_hydrodynamics, export_func=export_func)

    regularisation_sum_init = sum([1/(options.timestep)*((eta_bc_const[i])-eta_bc_const[i-1])**2 for i in range(1, len(eta_bc_const)-int(2*60*60/600))])
    

    regularisation_init = th_adj.AdjFloat(fac) * th_adj.AdjFloat(scale) * regularisation_sum_init

    #regularisation_init = fac * scale * sum([1/(options.timestep*len(th_adj.Function(R).dat.data[:]))*(th_adj.Function(R).assign(eta_bc_const[i])-th_adj.Function(R).assign(eta_bc_const[i-1]))**2*th_adj.dx for i in range(3, 10)])
    #regularisation_out = fac * scale * sum([1/(options.timestep*len(th_adj.Function(R).dat.data[:]))*(th_adj.Function(R).assign(eta_bc_const[i])-th_adj.Function(R).assign(eta_bc_const[i-1]))**2*th_adj.dx for i in range(len(eta_bc_const)-10, len(eta_bc_const)-3)])
    #import ipdb; ipdb.set_trace()
    J = sum(J_list) + regularisation_init# + th_adj.assemble(regularisation_out)
    return J, scale, wd_obs_arr


h_amp = 0.5     # ocean boundary forcing amplitude
h_T = 12*3600.    # ocean boundary forcing period
ocean_elev_func = lambda t: h_amp * (-th_adj.cos(2 * th_adj.pi * (t-(6*3600)) / h_T) + 1) 

times = np.linspace(0, (24*3600)+1200, (24*6)+3)



eta_bc_proj = []

for i in times:
    if i <= 18*3600:
        if i >= 6*3600:
            eta_bc_proj.append(th_adj.AdjFloat(0.01+ocean_elev_func(i)))
        else:
            eta_bc_proj.append(th_adj.AdjFloat(0.01))
    else:
        eta_bc_proj.append(th_adj.AdjFloat(0.01))
       

J, scale, wd_obs_array = forward(eta_bc_proj, mesh2d)
print(J)

for i in range(len(wd_obs_array)):
    wd_obs_array[i].dat.data[:] = wd_obs_array[i].dat.data[:] + np.random.normal(0, 0.1, len(wd_obs_array[i].dat.data[:]))


def forward_next(eta_bc_const, mesh2d, wd_obs_arr, scale, scale_return = False):
    
    def heavyside_approx(H, alpha):
        return 0.5*(H/(th_adj.sqrt(H**2+alpha**2)))+0.5

    


    def update_forcings_hydrodynamics(t_new):
        in_fn = ((1-dirk22_factor)*eta_bc_const[int(t_new/options.timestep)-1]) + (dirk22_factor*eta_bc_const[int(t_new/options.timestep)]) 
        eta_list.append(in_fn)       
            
        #print(int(t_new/options.timestep))

        in_c.assign(in_fn)
            

        if np.round(t_new%options.timestep, 2) == 0.00:
        
            elev1 = (solver_obj.fields.solution_2d[2])
            
            in_fn = eta_bc_const[int(t_new/options.timestep)]
            eta_list.append(in_fn)      
            
            in_c.assign(in_fn)
            

            wd_obs = th_adj.project(wd_obs_arr[int(t_new/dt)], P1_2d)

            # Record the simulated wetting and drying front
            wd_tmp = th_adj.project(heavyside_approx(- elev1 - h_static, wetting_alpha), P1_2d)
            wd.assign(wd_tmp)
        
            form = 0.5*th_adj.inner(wd - wd_obs, wd - wd_obs)*th_adj.dx
            

            #form = 0.5*th_adj.inner(wd, wd)*th_adj.dx
            #print(J_mc)
            #J_list.append(th_adj.assemble(dt*scale*form))

            J_list.append(th_adj.assemble(scale*options.timestep*form))
            #J_list.append(th_adj.assemble(options.timestep*form))
            

            

    wetting_and_drying = True
    ks = 0.025
    average_size = 200 * (10**(-6))
    t_end=24*3600.
    friction = 'manning'
    friction_coef = 0.025
    diffusivity = 0.15
    viscosity = 10**(-6)
    dt = 600
            
    # choose directory to output results
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    outputdir = 'outputs'+ st
 
    # export interval in seconds
    t_export = np.round(t_end/40,0)

    th_adj.print_output('Exporting to '+outputdir)

    #mesh2d = th_adj.Mesh('mesh.msh')#th_adj.RectangleMesh(12, 6, 13800, 7200)
    
    # define function spaces
    V = th_adj.FunctionSpace(mesh2d, 'CG', 2)
    P1_2d = th_adj.FunctionSpace(mesh2d, 'DG', 6)
    R = th_adj.FunctionSpace(mesh2d, 'DG', 0)
    
    x, y = th_adj.SpatialCoordinate(mesh2d)

    max_depth = 5
    basin_x = 13800

    h_static = (1.-x/basin_x)*max_depth

    bathymetry_2d = th_adj.Function(V)
    bathymetry_2d.project((1.-x/basin_x)*max_depth)
    
    #elev1 = th_adj.Function(P1_2d).interpolate(th_adj.Constant(0.0))
    # define parameters
    #    ksp = th_adj.Constant(3*average_size)
    
    # initial condition: assign non-zero velocity
    elev_init = th_adj.Constant(0.01)
    uv_init = th_adj.Constant((10**(-4), 0.))    
    
    depth = th_adj.Function(V).interpolate(elev_init + bathymetry_2d)
    
    wetting_alpha = 0.43
    
    #wd_obs_arr =[(th_adj.project(heavyside_approx(- elev_init - h_static, wetting_alpha), P1_2d, annotate = False))]
    wd_obs = th_adj.project(wd_obs_arr[0], P1_2d)#, annotate = False)
    # Record the simulated wetting and drying front
    wd = th_adj.project(heavyside_approx(- elev_init - h_static, wetting_alpha), P1_2d)   
    eta_list = []
    J_list = []

    th_adj.parameters['form_compiler']['quadrature_degree'] = 20
    th_adj.parameters['form_compiler']['cpp_optimize'] = True
    th_adj.parameters['form_compiler']['cpp_optimize_flags'] = '-O3 -ffast-math -march=native'
    th_adj.parameters['form_compiler']['optimize'] = True
        
    # set up solver 
    solver_obj = th_adj.solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    options = solver_obj.options 
    options.simulation_export_time = t_export
    options.simulation_end_time = t_end
    options.output_directory = outputdir
    options.norm_smoother = th_adj.Constant(0.43)
    options.check_volume_conservation_2d = True
    options.fields_to_export = ['uv_2d', 'elev_2d']
    options.solve_tracer = False
    options.use_lax_friedrichs_tracer = False
    if friction == 'nikuradse':
        options.quadratic_drag_coefficient = cfactor
    elif friction == 'manning':
        if friction_coef == 0:
                friction_coef = 0.02
        options.manning_drag_coefficient = th_adj.Constant(friction_coef)
    else:
        print('Undefined friction')

    # set horizontal diffusivity parameter
    options.horizontal_diffusivity = th_adj.Constant(diffusivity)
    if viscosity == None:
        print('no viscosity')
    else:
        options.horizontal_viscosity = th_adj.Constant(viscosity)

    # crank-nicholson used to integrate in time system of ODEs resulting from application of galerkin FEM
    options.timestepper_type = 'DIRK22'
    dirk22_factor = (2 - th_adj.sqrt(2))/2
    #options.timestepper_options.implicitness_theta = 0.5
    options.use_wetting_and_drying = wetting_and_drying
    options.wetting_and_drying_alpha = th_adj.Constant(wetting_alpha)
  
    

    if not hasattr(options.timestepper_options, 'use_automatic_timestep'):
        options.timestep = dt

    
    # set boundary conditions

    in_constant = eta_bc_const[0]

    in_c = th_adj.Constant(0.0)
    in_c.assign(in_constant)

    #in_c.assign(in_constant)
    swe_bnd = {}
    swe_bnd[1] = {'elev': in_c}
   
    solver_obj.bnd_functions['shallow_water'] = swe_bnd
    


    solver_obj.assign_initial_conditions(uv = uv_init, elev= elev_init)
    #solver_obj.assign_initial_conditions(uv = uv_init)
    


    solver_obj.iterate(update_forcings=update_forcings_hydrodynamics)

    regularisation_sum_init = sum([1/(options.timestep)*((eta_bc_const[i])-eta_bc_const[i-1])**2 for i in range(1, len(eta_bc_const)-int(2*60*60/600))])

    regularisation_init = th_adj.AdjFloat(fac) * th_adj.AdjFloat(scale) * regularisation_sum_init

    
    #regularisation_init = fac * scale * sum([1/(options.timestep*len(th_adj.Function(R).dat.data[:]))*(th_adj.Function(R).assign(eta_bc_const[i])-th_adj.Function(R).assign(eta_bc_const[i-1]))**2*th_adj.dx for i in range(3, 10)])
    #regularisation_out = fac * scale * sum([1/(options.timestep*len(th_adj.Function(R).dat.data[:]))*(th_adj.Function(R).assign(eta_bc_const[i])-th_adj.Function(R).assign(eta_bc_const[i-1]))**2*th_adj.dx for i in range(len(eta_bc_const)-10, len(eta_bc_const)-3)])
    #import ipdb; ipdb.set_trace()
    J = sum(J_list) + regularisation_init# + th_adj.assemble(regularisation_out)
    
    if scale_return:
        return J, scale
    else:
        return J


eta_bc_new_proj = []

for i in times:
    eta_bc_new_proj.append(th_adj.AdjFloat(0.01))

scale = th_adj.AdjFloat(1.0)

J_tmp, scale_new = forward_next(eta_bc_new_proj, mesh2d, wd_obs_array, scale, scale_return = True)
print(J_tmp)


rf_tmp = th_adj.ReducedFunctional(J_tmp, [th_adj.Control(p) for p in eta_bc_new_proj])#, eval_cb_pre=eval_callback)

scalar = rf_tmp(eta_bc_new_proj)

scale_new = th_adj.AdjFloat(1/scalar)

tape = th_adj.get_working_tape()

tape.clear_tape()

tape = th_adj.get_working_tape()
print(len(tape.get_blocks()))


testing_derivative = False

if testing_derivative == True:
    
    J_new = forward_next(eta_bc_new_proj, mesh2d, wd_obs_array, scale_new)
    print(J_new)
    
    
    rf = th_adj.ReducedFunctional(J_new, [th_adj.Control(p) for p in eta_bc_new_proj[:-int(2*60*60/600)]])#, eval_cb_pre=eval_callback)
    
    
    eta_bc_test_proj = []

    for i in times:
        eta_bc_test_proj.append(th_adj.AdjFloat(0.01-10**(-6)))  


    J_h = rf(eta_bc_new_proj[:-int(2*60*60/600)])
    print(J_h)
    
    der = rf.derivative()
    
    print(der)
    
    plt.plot(der)
    
    
    J_0 = rf(eta_bc_test_proj[:-int(2*60*60/600)])
    print(J_0)


    sum_list = []

    for i in range(len(der)):
        sum_list.append(der[i]*10**(-6))
    
    print(sum(sum_list))
    print(J_h-J_0)

    stop
    
taylor_test = False

if taylor_test == True:

    J_new = forward_next(eta_bc_new_proj, mesh2d, wd_obs_array, scale_new)

    rf = th_adj.ReducedFunctional(J_new, [th_adj.Control(p) for p in eta_bc_new_proj[:-int(2*60*60/600)]], eval_cb_post=eval_callback)

    h = []
    for i in range(len(eta_bc_new_proj[:-int(2*60*60/600)])):
        h.append(th_adj.AdjFloat(0.001*(np.random.random()-0.5)))

    conv_rate = th_adj.taylor_test(rf, eta_bc_new_proj[:-int(2*60*60/600)], h)

    
    if conv_rate > 1.9:
        print('*** test passed ***')
    else:
        print('*** ERROR: test failed ***')
        
    stop
        
J_new = forward_next(eta_bc_new_proj, mesh2d, wd_obs_array, scale_new)

print(J_new)

rf = th_adj.ReducedFunctional(J_new, [th_adj.Control(p) for p in eta_bc_new_proj[:-int(2*60*60/600)]], eval_cb_post=eval_callback)    


#test = rf(eta_bc_new_proj[1:-int(2*60*60/600)])


min_values = th_adj.minimize(rf, options={'disp':True, 'gtol':  1e-200, 'ftol': 1e-9, 'maxfun': 1000})