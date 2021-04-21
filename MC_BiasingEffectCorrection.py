## Last edited 20210421 Haipei
## Nashlab


if 1: ## IMPORT AND PRE-FORMAT
    from matplotlib import pyplot as plt
    from scipy.optimize import curve_fit
    from scipy import stats
    import cython
    import scipy
    import numpy as np
    import argparse
    import math
    import seaborn as sns
    import matplotlib as mpl
    import time as timer
    import logging as log
    import os, os.path
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    import scipy
    from scipy.sparse import spdiags
    from scipy.stats import ttest_ind
    from scipy import stats
    from scipy import linalg
    from scipy.optimize import curve_fit
    from scipy.optimize import leastsq
    from numpy import trapz
    import scipy.optimize as optimize
    from datetime import datetime
    import pandas as pd

    now = datetime.now()
    timeindex = now.strftime("%Y%m%d_%H%M_")
    np.random.seed(None)
    # formatting for graphics
    fontsize = 24
    sns.set(style="ticks",font="Arial")
    sns.set_style("ticks", {"xtick.direction":u'in', "ytick.direction":u'in'})
    colors = sns.color_palette()

if 1: ## HERE ARE ALL THE INPUT PARAMETERS
    use_afm_tip = 1 # set to 1 if an AFM tip is used
    kT = 4.14 # in pN*nm
    k = 91. # in pN/nm
    linker = 76
    # if dispersity is True, a random number drawn from a gaussian dist. around zero with this sigma will be added to spacer
    dispers = 0
    persist = 0.365 #in nm

    #domain params syntax params = [contL, dx, k0]
    Sdrg = ("Sdrg",[204, 0.063, 1.8e-11])
    B1 = ("B1",[36, 0.083, 5.36e-15])
    B2 = ("B2",[36, 0.0768, 4.292e-14])
    NoFP = ("NoFP",[36, 0.001, 4.25e-14])


    type3 = ("type3",[204, 0.13, 7.3e-7])
    domain2 = ("domain2",[241, 0.15, 2.6e-6])

    xMod = ("xMod",[36, 0.0578, 3.0e-4])
    Rc =("Rc",[204, 0.0919, 4.15e-7])
    Scbd =("Scbd",[204, 0.57, 0.09])
    FIVAR =("FIVAR",[31, 0.592, 6.2e-2])
    ddfln4 =("ddfln4",[31, 0.48, 0.136])
    CD =("CD",[31, 0.72, 0.011])    # From C.perfringens Coh Doc, accroding to  JPC Lukas 2017 paper.

    cbm = ("cbm",[54, 0.4, 0.005])
    testfp = ("testfp",[54, 0.4, 0.01])

    load = 184000 #in pN/s
    speed = 800 #in nm/s
    nu =2./3.
    k0_scale = 1.0
    x0_scale = 1.0

def format_figure(axis):
    #axis = plt.gca()
    axis.tick_params(axis = "both", direction = "in", width = 1.2)
    for item in [axis.title, axis.xaxis.label, axis.yaxis.label]:
        item.set_fontsize(fontsize)
    for item in (axis.get_xticklabels() + axis.get_yticklabels()):
        item.set_fontsize(fontsize-2)

def generate_axis_cs(time,step,speed,noiseval,lc,time_from): # axis under constant speed mode

    timeaxis = np.linspace(time_from,time,np.int(1+(time-time_from)/step))
    ext = timeaxis*speed

    #setup a baseline with some gaussian noise
    fnoise = np.zeros(len(timeaxis))
    if noiseval == 0:
        fnoise_base = np.zeros(len(timeaxis))
    else:
        fnoise_base = np.random.normal(0,noiseval,len(timeaxis))

    #in constant speed, force is calculated using th WLC model
    fnoise = fnoise_base + f_WLC(ext,lc,persist)
    if use_afm_tip == 1:    # only do bending correction with AFM tip
        dist = ext+fnoise/k
    else:
        dist = ext
    timeaxis = dist/speed

    i = 0
    resolution_F = 1.0
    while (ext[i] < lc*0.95):
    #Produce a finer axis when F resolution is worse then a value(0.1pN)
        if (fnoise[i+1]-fnoise[i]) > resolution_F :
        # print ("fix to a smaller ext interval")
            n_step= np.int(2* (lc*0.97-ext[i])/(ext[i+1]-ext[i]))
            ext= np.concatenate((ext[:i],np.linspace(ext[i],lc*0.97,n_step)),axis= None)
        # Add noise
            fnoise_c = np.zeros(len(ext))
            if noiseval == 0:
                fnoise_base = np.zeros(len(ext))
            else:
                fnoise_base = np.random.normal(0,noiseval,len(ext))
        # Build updated F, time and dist upon bending correction
            fnoise_c = fnoise_base + f_WLC(ext,lc,persist)
            fnoise= np.concatenate((fnoise[:i],fnoise_c[i:]),axis= None)
            if use_afm_tip == 1:    # only do bending correction with AFM tip
                dist = ext+fnoise/k
            else:
                dist = ext
            timeaxis = dist/speed
        i += 1
    return timeaxis,fnoise,ext,dist

def generate_axis_fr(time,step,lr,noiseval,lc,time_from,ext_from):    # axis under force ramp mode, not AFM force ramp
    # when extra lc added, time_from, ext_from is the current status to continue.

    #timeaxis = np.linspace(time_from,time,np.int(1+(time-time_from)/step))
    ext = np.linspace(ext_from,0.99*lc,np.int(1+(time-time_from)/step))

    #setup a baseline with some gaussian noise
    fnoise = np.zeros(len(ext))
    if noiseval == 0:
        fnoise_base = np.zeros(len(ext))
    else:
        fnoise_base = np.random.normal(0,noiseval,len(ext))

    #force is calculated using th WLC model
    f_clean = f_WLC(ext,lc,persist)
    fnoise = fnoise_base + f_clean
    timeaxis = (f_clean-f_clean[0])/lr+time_from

    if use_afm_tip == 1:    # only do bending correction with AFM tip
        dist = ext+fnoise/k
    else:
        dist = ext

    i = 0
    resolution_F = 1.0
    while (ext[i] < lc*0.95):
    #Produce a finer axis when F resolution is worse then a value(0.1pN)
        if (fnoise[i+1]-fnoise[i]) > resolution_F :
            #print ("fix to a smaller ext interval")
            n_step= np.int(2* (lc*0.97-ext[i])/(ext[i+1]-ext[i]))
            ext= np.concatenate((ext[:i],np.linspace(ext[i],lc*0.97,n_step)),axis= None)
        # Add noise
            fnoise_c = np.zeros(len(ext))
            if noiseval == 0:
                fnoise_base = np.zeros(len(ext))
            else:
                fnoise_base = np.random.normal(0,noiseval,len(ext))
        # Build updated F, time and dist upon bending correction
            f_clean_c = f_WLC(ext,lc,persist)
            fnoise_c = fnoise_base + f_clean_c
            fnoise = np.concatenate((fnoise[:i],fnoise_c[i:]),axis= None)
            timeaxis_c = (f_clean_c-f_clean_c[0])/lr+time_from
            timeaxis = np.concatenate((timeaxis[:i],timeaxis_c[i:]),axis= None)
            if use_afm_tip == 1:    # only do bending correction with AFM tip
                dist = ext+fnoise/k
            else:
                dist = ext
        i += 1
    return timeaxis,fnoise,ext,dist

def pdfBell(force,r,dx,k0):
    """probability density function in the BE model"""
    pdf = k0/r * np.exp(dx*force/kT - k0*kT*(np.exp(dx*force/kT)-1)/(dx*r))
    return pdf

def pdfDudko(force,r,G,dx,k0):
    """pdf in the DHS model"""
    pdf = dudkoRate(force,G,dx,k0)/r * np.exp(kT*k0/(dx*r)) * np.exp(-(dudkoRate(force,G,dx,k0)*kT/(dx*r)*(1.-(nu*force*dx/G))**(1.-1./nu)))
    return pdf

def intbell(force,r,dx,k0):
    """Integrated Bell pdf, cumulative distribtion function"""
    ipdf = 1.0-np.exp(-(k0*kT/(r*dx)*(np.exp((dx*force)/kT)-1.0)))
    return ipdf

def fpbell(force,r,dx,k0,dxc,k0c):
    """unnormalized biased fingerprint distibution"""
    pdf = pdfBell(force,r,dx,k0)*(1.0-intbell(force,r,dxc,k0c))
    return pdf

def cplbiased(force,r,dxc,k0c,dxfp,k0fp):
    """unnormalized biased complex distribution"""
    pdf = pdfBell(force,r,dxc,k0c)*intbell(force,r,dxfp,k0fp)
    return pdf

def rate(force,dx,k0):  ##off rate in the BE model:
    """off rate in the BE model"""
    rate = k0 * np.exp(dx*force/kT)
    return rate

def dudkoRate(force,G,dx,k0):
    rate = k0*(1. - (nu*force*dx/G))**(1./nu -1.) * np.exp((G/kT) *(1.-(1.-(nu*force*dx/G))**(1./nu)))
    return rate

def intrate(force,step,ldr,dx,k0):
    """integrated off rate, to get rupture probability"""
    intrate = kT*k0/(dx*ldr)*(np.exp(dx/kT*(force+step*ldr))-np.exp(dx*force/kT))
    return intrate

def f_WLC(x,contL,lp):
    """force extension behavior in the WLC model"""
    if len(x)>1:
        x = np.concatenate((x[x<contL],np.zeros(len(x[x>=contL]))),axis= None)
    wlc = kT/lp * (1/(4.0*(1.0-x/contL)**2.0) + x/contL - 1.0/4.0)
    return wlc

def normalize(ldr,cpl,fingerprint):
    """normalization function for the biased fingerprint distribution"""
    #    integrand = pdfBell(x,ldr,cpl[1:])*intbell(x,ldr,fingerprint[1:])
    norm, err = scipy.integrate.quad(lambda x: fpbell(x,ldr,fingerprint[1],fingerprint[2],cpl[1],cpl[2]), 0, 3000)
    return norm

def eta_ratio(ldr, FP_dx, FP_k, cpl):
    """normalization function for the biased fingerprint distribution"""
    norm, err = scipy.integrate.quad(lambda x: fpbell(x,ldr,FP_dx,FP_k,cpl[1],cpl[2]), 0, 3000)
    return norm

def fitting_eta(ldr, FP_dx, FP_k, cpl):
    """for a list in put, use this for lsqfitting"""
    FP_k_ture = FP_k / k0_scale
    FP_dx_ture = FP_dx / x0_scale
    answerlist = np.asarray([ eta_ratio(loadr,FP_dx_ture, FP_k_ture, cpl) for loadr in ldr])
    chisq = 0
    for ik in range(len(ldr)):
        chisq = chisq + np.square(answerlist[ik] - ydata[ik])
    print ('{:14.10f} {:14.10e} {} chisq: {:8.4f}'.format(FP_dx_ture,FP_k_ture, answerlist,chisq))
    return answerlist

def fitting_eta_MCS_cs(speed_list,fp_dx,fp_k,fp_lc,cpl):
    """for a list in put, use this for lsqfitting"""
    fp_k_ture = fp_k / k0_scale
    fp_dx_ture = fp_dx / x0_scale
    fingerprint_calc = [fp_lc,fp_dx_ture,fp_k_ture]
    answerlist = np.asarray([eta_constspeed(cpl[1],fingerprint_calc,speed) for speed in speed_list])
    chisq=0
    for ik in range(len(speed_list)):
        chisq = chisq + np.square(answerlist[ik] - ydata[ik])
    print ('{:14.10f} {:14.10e} {} chisq: {:8.4f}'.format(fp_dx_ture,fp_k_ture, answerlist,chisq))
    return answerlist

def eta_constspeed(cpl,fingerprint,speed, num=2000, n_step=500, k=91, spacer=linker, dispers=0,d4mode=0,noiseval=0,savecurves=False):
    """This is just simple version to get eta, it only focus on if FP unfolds or not"""
    time = (spacer+30)/speed
    step = time / n_step
    curve_with_fp = 0
    timeaxis, fnoise, ext, dist = generate_axis_cs(time,step,speed,noiseval,spacer, time_from=0)
    p_rup = np.zeros(len(timeaxis))
    p_unf = np.zeros(len(timeaxis))
    p_d4unf = np.zeros(len(timeaxis))

    # _d4u :status after ddfln4 unfolding.
    p_rup_d4u = np.zeros(len(timeaxis))
    p_unf_d4u = np.zeros(len(timeaxis))
    timeaxis_d4u, fnoise_d4u, ext_d4u, dist_d4u = generate_axis_cs(time,step,speed,noiseval,spacer+32, time_from=0)

    for i in range(len(timeaxis)-1):
        step_bc = (timeaxis[i+1]-timeaxis[i])
        p_unf[i] = rate(fnoise[i],*fingerprint[1:])*step_bc
        p_rup[i] = rate(fnoise[i],*cpl[1:])*step_bc
        p_d4unf[i] = rate(fnoise[i],*ddfln4[1][1:])*step_bc

    for i in range(len(timeaxis_d4u)-1):
        step_bc_d4u = (timeaxis_d4u[i+1]-timeaxis_d4u[i])
        p_unf_d4u[i] = rate(fnoise_d4u[i],*fingerprint[1:])*step_bc_d4u
        p_rup_d4u[i] = rate(fnoise_d4u[i],*cpl[1:])*step_bc_d4u

    for i_sim in range (num):
        if fingerprint != None:
            # the probability = koff * StepTime
            lr_rup = 0
            lr_uf = 0
            i = 100

            if d4mode != 1:
    # original one fingerprint case without ddfln4:
                while i < len(timeaxis):
                    pick_rup = np.random.random_sample()
                    pick_unf = np.random.random_sample()
        # pick_unf < p_unf[i] --> fingerprint unfolds
        # pick_rup < p_rup[i] --> complex ruptures
                    if pick_unf > p_unf[i] and pick_rup > p_rup[i]: #both survive, nothing happens
                        i+=1
                    elif pick_unf < p_unf[i] and pick_rup > p_rup[i]: #fingerprint unfolds
                        curve_with_fp += 1
                        break
                    else: #complex ruptures with fingerprint intact, experiment ends
                        break
    # below shows the ddfln4 solution:
            else:
                i_unf=0
                i_rup=0
                i_d4=0
                while i < len(timeaxis):
                    pick_rup = np.random.random_sample()
                    pick_unf = np.random.random_sample()
                    pick_d4unf = np.random.random_sample()
                    if i_unf==0 and pick_unf < p_unf[i] and pick_rup > p_rup[i]:        #fingerprint unfolds
                        curve_with_fp += 1
                        i_unf=i
                    elif i_d4 ==0 and pick_d4unf < p_d4unf[i]:    #d4 starts to unfold, simulate 2nd peak of d4, update force-extension.
                        i_d4 = i
                        while timeaxis_d4u[i] > timeaxis[i_d4]:
                            i=i-1
                        while i < len(timeaxis_d4u):
                            pick_rup = np.random.random_sample()
                            pick_unf = np.random.random_sample()
                            if i_unf==0 and pick_unf < p_unf_d4u[i] and pick_rup > p_rup_d4u[i]:        #fp unfolds\
                                curve_with_fp += 1
                                i_unf=i
                                break
                            elif pick_rup < p_rup_d4u[i]:
                                i_rup=i
                                break
                            i+=1
                    elif pick_rup < p_rup[i]:            #complex ruptures, experiment ends.
                        break
                    if i_unf!=0 or i_rup!=0:
                        break
                    i += 1
    eta_sim = curve_with_fp / num
    return eta_sim

def correction_MCS_cs(cpl,fingerprint):
    xdata = [400., 800., 1600., 3200.]

    global x0_scale, k0_scale
    while fingerprint[1][1]*x0_scale < 1.0:
        x0_scale = x0_scale*1e1

    while fingerprint[1][2]*k0_scale < 1.0:
        k0_scale = k0_scale*1e1 # a scalling factor is applied: fitting input smaller then 1e-10 has numerical issues in trf.

    initial_FP = (fingerprint[0],[fingerprint[1][0],fingerprint[1][1]*x0_scale,fingerprint[1][2]*k0_scale])
    cor_FP = (fingerprint[0],[fingerprint[1][0],fingerprint[1][1],fingerprint[1][2]])

    fit_precision_x = 1.0e-3
    fit_precision_y = 1.0e-9

    print ("                            -------------- eta --------------")
    print ("    pulling speed[nm/s]:    {:8},{:8},{:8},{:8}".format(xdata[0],xdata[1],xdata[2],xdata[3]))
    print ("       dx       ko                    ")

    FP_lc = initial_FP[1][0]
    eta_fits, pcov = curve_fit(lambda speeds, FP_dx, FP_k: fitting_eta_MCS_cs(speeds,FP_dx,FP_k,FP_lc,cpl) ,
                xdata,ydata,[initial_FP[1][1],initial_FP[1][2]], ftol=fit_precision_y, xtol=fit_precision_x, method='trf',
                diff_step=[0.05,0.2], bounds=([0., 0.], [50., 10000.]))

    print (eta_fits)
    print (pcov)
    print ("original:", initial_FP)
    cor_FP[1][1] = eta_fits[0]/x0_scale
    cor_FP[1][2]  = eta_fits[1]/k0_scale
    print ("Now try corrected:", cor_FP)

    fingerprint = cor_FP
    return cor_FP

def normalizecpl(ldr,cpl,fingerprint):
    """normalization function for the biased complex distribution"""
    norm, err = scipy.integrate.quad(lambda x: cplbiased(x,ldr,cpl[1],cpl[2],fingerprint[1],fingerprint[2]), 0, 3000)
    return norm

def mean_cpl(cpl,fingerprint,ldr):
    mean_un, err = scipy.integrate.quad(lambda x: x*cplbiased(x,ldr,cpl[1][1],cpl[1][2],fingerprint[1][1],fingerprint[1][2]),0,3000)
    mean = mean_un/normalizecpl(ldr,cpl[1],fingerprint[1])
    return mean

def mean_fp(cpl,fingerprint,ldr):
    mean_un, err = scipy.integrate.quad(lambda x: x*fpbell(x,ldr,fingerprint[1][1],fingerprint[1][2],cpl[1][1],cpl[1][2]),0,3000)
    mean = mean_un/normalize(ldr,cpl[1],fingerprint[1])
    return mean

def mean_bell(ldr,cpl):
    mean, err = scipy.integrate.quad(lambda x: x*pdfBell(x,ldr,cpl[1][1],cpl[1][2]),0,3000)
    return mean

def simulate_constspeed(time,step,noiseval,spacer,cpl,fingerprint,speed,dispers=0):
    """function to simulate constant speed experiments"""
 # find E for cpl only
    lc_t = spacer
    timeaxis, fnoise, ext, dist = generate_axis_cs(time,step,speed,noiseval,spacer, time_from=0)
    p_rup_fpunf = 1-np.exp(-rate(fnoise,*cpl[1:])*step)
    f_rupture, lr_rup, i2 = compare_nofp(timeaxis,fnoise,p_rup_fpunf,i_nofp=10)
    soloe = trapz(fnoise[1:i2+1], ext[1:i2+1],dx=1.0)
 # Initialize
    #polydispersity if needed
    if dispers != 0:
        lc_t += np.random.normal(0,dispers)

    #setup a time and extension axis
    timeaxis, fnoise, ext, dist = generate_axis_cs(time,step,speed,noiseval,lc_t, time_from=0)
    i_rup = 0
 # Monte Carlo test: roll a dice
    if fingerprint != None:
 #   M1. Define indexes
        p_rup = np.zeros(len(timeaxis))
        p_unf = [np.zeros(len(timeaxis))]
        for i in range((len(fingerprint)-1)):
            p_unf.append(np.zeros(len(timeaxis)))
        f_unfold = np.zeros(len(fingerprint))
        i_unf = np.zeros(len(fingerprint))
        fpleft = len(fingerprint)
        lr_rup = 0.
        f_rupture = 0.
        lr_unf = 0.
        i = 10
        while fnoise[i] < 20. : # Skip several points that will be rupture.
            i+=1
 #   M2. Start time axis
        while i < len(timeaxis):
            step_bc = (timeaxis[i+1]-timeaxis[i])
            p_rup[i] = 1-np.exp(-rate(fnoise[i],*cpl[1:])*step_bc)
            pick_rup = np.random.random_sample()
 #       M2.a1. complex survive
            if pick_rup > p_rup[i]:
 #       M2.a2. check all fingerprint
                for i_fp in range(len(fingerprint)):
                    p_unf[i_fp][i] = 1-np.exp(-rate(fnoise[i],*fingerprint[i_fp][1][1:])*step_bc)
                    pick_unf = np.random.random_sample()
                    if pick_unf < p_unf[i_fp][i] and i_unf[i_fp] == 0:
                        i_unf[i_fp] = i
                        f_unfold[i_fp] = fnoise[i]
                        lr_unf = (fnoise[i] - fnoise[i-1]) / step_bc
                        lc_t = lc_t + fingerprint[i_fp][1][0]
                    # calc dissipate energy, no need so far.
                    #    f_nofp = f_WLC(ext,lc_t,persist)
                    #    disse = trapz(fnoise[1:i+1]-f_nofp[1:i+1],ext[1:i+1],dx=1.0)
                    # update the axis for the Lc increment
                        timeaxis_c, fnoise_c, ext_c, dist_c = generate_axis_cs(time,step,speed,noiseval,lc_t,timeaxis[i])
                        timeaxis= np.concatenate((timeaxis[:(i-1)],timeaxis_c),axis= None)
                        fnoise= np.concatenate((fnoise[:(i-1)],fnoise_c),axis= None)
                        ext= np.concatenate((ext[:(i-1)],ext_c),axis= None)
                        dist= np.concatenate((dist[:(i-1)],dist_c),axis= None)
                        for i_updt in range(len(fingerprint)):
                            p_unf[i_updt] = np.concatenate((p_unf[i_updt][:(i-1)],dist_c),axis= None)
                        p_rup= np.concatenate((p_rup[:(i-1)],dist_c),axis= None)
                        time_step_list = np.zeros(len(timeaxis))
                        for ii in range(len(timeaxis)-2):
                            time_step_list[ii+1] = (timeaxis[ii+2]-timeaxis[ii])/2
                        p_rup_fpunf = 1-np.exp(-rate(fnoise,*cpl[1:])*(time_step_list))
                        fpleft = fpleft - 1
 #       M2.a3. if all fingerprint unfolds, go to rupture
                if fpleft == 0:
                    f_rupture, lr_rup, i = compare_nofp(timeaxis,fnoise,p_rup_fpunf,i_nofp=i)
                    fnoise[i+1:] = np.random.normal(0,noiseval,len(fnoise[i+1:]))
                    disp_rup = f_rupture/k
                    if use_afm_tip == 1:
                        ext[i+1:] = ext[i+1:]+disp_rup
                    break
                i += 1
 #       M2.b. complex ruptures experiment ends
            else:
                f_rupture = fnoise[i]
                fnoise[i+1:] = np.random.normal(0,noiseval,len(fnoise[i+1:]))
                disp_rup = f_rupture/k
                if use_afm_tip == 1:
                    ext[i+1:] = ext[i+1:]+disp_rup
                disse=0.
                break
    else:
        i = i2
        fpleft=0
        f_unfold=0
    disse=0.
    totale = trapz(fnoise[1:i+1], ext[1:i+1],dx=1.0)

    return fpleft, disse, totale, soloe, lr_rup, f_rupture, f_unfold, ext, fnoise

def compare_nofp(timeaxis,fnoise,p_rup,i_nofp=0):
    f_rupture_nofp = 0.
    rup_lr_nofp = 0.
    while i_nofp < len(timeaxis):
        pick = np.random.random_sample()
        if pick < p_rup[i_nofp]:
            f_rupture_nofp = fnoise[i_nofp]
            rup_lr_nofp = (fnoise[i_nofp] - fnoise[i_nofp-1]) / (timeaxis[i_nofp] - timeaxis[i_nofp-1])
            break
        else:
            i_nofp+=1
    return f_rupture_nofp, rup_lr_nofp, i_nofp

def correction(cpl,fingerprint):
    xdata = [1500. , 2618. , 7431. , 18067.] # Loading rates in pN/s
    print ()
    print ("                            ---------- eta ------------")
    print ("    pulling speed[nm/s]:    lr1,    lr2,    lr3,    lr4")
    print ("       dx       ko                    ")

    global k0_scale, x0_scale
    while fingerprint[1][1]*x0_scale < 1.0:
        x0_scale = x0_scale*1e1
    while fingerprint[1][2]*k0_scale < 1.0:
        k0_scale = k0_scale*1e1 # a scalling factor is applied: fitting input smaller then 1e-10 has numerical issues in trf.


    initial_FP = (fingerprint[0],[fingerprint[1][0],fingerprint[1][1]*x0_scale,fingerprint[1][2]*k0_scale])
    cor_FP = (fingerprint[0],[fingerprint[1][0],fingerprint[1][1],fingerprint[1][2]])

    fit_precision_x = 1.5e-3 # [!] should be smaller than both expected dx and ko
    fit_precision_y = 1.5e-3

    print ('initial dx ko input is {:12.10f} {:12.8e}'.format(initial_FP[1][1],initial_FP[1][2]))
    eta_fits, pcov = curve_fit(lambda ldr, FP_dx, FP_k: fitting_eta(ldr,FP_dx,FP_k,cpl[1]) ,
                xdata,ydata,[initial_FP[1][1],initial_FP[1][2]], ftol=fit_precision_y, xtol=fit_precision_x
                , method='trf',bounds=([1.e-3, 1.e-5], [20., 10000.]))
    print (eta_fits)
    print (pcov)

    print ("original:", initial_FP)
    cor_FP[1][1] = eta_fits[0]/x0_scale
    cor_FP[1][2]  = eta_fits[1]/k0_scale
    print ("Now try corrected:", cor_FP)

    fingerprint = cor_FP
    return cor_FP

def run_simulation(num,time,step,noiseval,spacer,cpl,ldr,fingerprint,forceramp=True,verbose=False,speed=speed,savecurves=False):
    rup_forces = np.zeros(num)
    fp_left = np.zeros(num)
    if fingerprint != None:
        unf_forces = [[0]*len(fingerprint)]*num
    else:
        unf_forces = np.zeros(num)
    disse = np.zeros(num)
    soloe = np.zeros(num)
    totale = np.zeros(num)
    totale_0FP_unf = np.zeros(num)
    totale_1FP = np.zeros(num)
    totale_0FP = np.zeros(num)
    mlr  = np.zeros(num)

    if forceramp == True:
        method = "ramp"+str(ldr)+"pNps"
    else:
        method = "speed_"+str(speed)+"nmps"

    #setup directories for saving files
    current_dir = os.getcwd()
    if fingerprint == None:
        pathname = timeindex + method+"_NoFP_n-"+str(num)+"_"+cpl[0]
    elif len(fingerprint) > 1:
        pathname = timeindex + method+"_multipleFP_n-"+str(num)+"_"+fingerprint[0][0]+"_"+cpl[0]
    else:
        pathname = timeindex + method+"_n-"+str(num)+"_"+fingerprint[0][0]+"_"+cpl[0]

    data_dir = os.path.join(current_dir, pathname)
    if not (os.path.exists(data_dir)):
        os.mkdir(pathname)
    os.chdir(data_dir)

    #log all parameters of the simulation
    print ("Saving simulation parameters")
    print ("Simulating {0} curves".format(num))
    log.basicConfig(filename="logfile.log",filemode='w',level=log.DEBUG)
    log.info("num: {0:0}".format(num))
    log.info("time: {0} sec".format(time))
    log.info("step: {0} sec".format(step))
    log.info("noise: {0} pN".format(noiseval))
    log.info("complex: {0}".format(cpl))
    log.info("fingerprint: {0}".format(fingerprint))
    log.info("forceramp: {0}".format(forceramp))
    if forceramp == True:
        log.info("ldr: {0} pN/s".format(ldr))
    else:
        log.info("speed: {0} nm/s".format(speed))
        log.info("spacer: {0} nm".format(spacer))

    #setup directory to save force-ext traces
    if savecurves == True:
        print ("Saving force-ext-traces")
        curve_dir = os.path.join(data_dir, "curves")
        if not (os.path.exists(curve_dir)):
            os.mkdir("curves")
        os.chdir(curve_dir)

    if forceramp == True:
        maxforce = ldr*time
        print ("Starting force ramp with {0} pN/s".format(ldr))
        print ("Computing curves up to {0} pN".format(maxforce))
    else:
        print ("Starting constant speed with {0} nm/s".format(speed))
        print ("Sampling {0:.1f} points/nm".format(1/(speed*step)))

    starttime2 = timer.perf_counter()
    #run the constant speed simulation num times
    for i in range(num):
        if forceramp == True:
            fp_left[i], disse[i], totale_1FP[i], soloe[i], mlr[i], rup_forces[i], unf_forces[i], extension, force = simulate_forceramp(time,step,noiseval,spacer,cpl[1],fingerprint,ldr)
        else:
            fp_left[i], disse[i], totale_1FP[i], soloe[i], mlr[i], rup_forces[i], unf_forces[i], extension, force = simulate_constspeed(time,step,noiseval,spacer,cpl[1],fingerprint,speed)
        if savecurves == True:
            np.savetxt("ext-force-{0}.txt".format(i),np.transpose([extension,force]))
            if use_afm_tip == 1:    # only do bending correction with AFM tip
                distance = extension+force/k
            else:
                distance = extension
            plot_edf_trace(extension,distance,force)
            plt.savefig("ext-force-trace-{0}.pdf".format(i))
    endtime2 = timer.perf_counter()
    print ("Computation took {0:.2f} sec".format(endtime2-starttime2))
    #sort out curves without fingerprint unfolding

    rup_forces_withFP = rup_forces[fp_left!=len(fingerprint)]
    rup_forces_withallFP = rup_forces[fp_left==0]
    #print ("total dissipating energy:{0}, ratio:{1}".format(np.sum(disse),np.sum(disse)/np.sum(totale)))
    #log.info("total dissipating energy:{0}, ratio:{1}".format(np.sum(disse),np.sum(disse)/np.sum(totale)))
    print ("{0}/{1} curves all FP unfolding ({2:.1f}%)".format(len(rup_forces_withallFP),num,100*len(rup_forces_withallFP)/num))
    log.info("{0}/{1} curves all FP unfolding ({2:.1f}%)".format(len(rup_forces_withallFP),num,100*len(rup_forces_withallFP)/num))
    print ("{0}/{1} curves at least one FP unfolding ({2:.1f}%)".format(len(rup_forces_withFP),num,100*len(rup_forces_withFP)/num))
    log.info("{0}/{1} curves at least one FP unfolding ({2:.1f}%)".format(len(rup_forces_withFP),num,100*len(rup_forces_withFP)/num))
    os.chdir(data_dir)
    #Plot resulting distributions
    #maxis = 100*math.ceil(np.amax(rup_forces)/100)
    maxis = 50*round(np.amax(rup_forces)/50)+300
    forceax = np.linspace(0,150,300)


    fig, ax = plt.subplots()
    if fingerprint == None:
        sns.histplot(rup_forces,ax=ax,label= "sim ruptures")
    else:
        sns.histplot(rup_forces,ax=ax,label= "Complex Rupture",binwidth=5,kde=True,stat="density",color=colors[0])
        for fp_i in range(len(fingerprint)):
            layoutdata = np.array([val[fp_i] for val in unf_forces])
            cleanf = layoutdata[layoutdata>0]
            if len(cleanf)>1:
                sns.histplot(cleanf,label=fingerprint[fp_i][0],binwidth=5,kde=True,stat="density",color=colors[fp_i+1])
            #if len(rup_forces) != len(rup_forcesclean):
            #    sns.distplot(rup_forces,ax=ax,label="all ruptures")
        ax.set(xlabel="Force [pN]")
        #ax.tick_params(axis = "y", labelleft = "off")
        for item in fig.get_axes():
            format_figure(item)
        fig.tight_layout()
        ax.legend(loc ="best", prop={"size": fontsize-6})
        plt.savefig("force-hist.pdf")
        plt.close('all')

    os.chdir(data_dir)
    np.savetxt("energy.txt",np.transpose(totale),header="#totale")
    np.savetxt("unfoldingforces.txt",unf_forces)
    np.savetxt("complexruptures.txt",np.transpose(rup_forces),header="#rup_forces")

    for i in range(num):
        if forceramp == True:
            fp_left[i], disse[i], totale_0FP[i], soloe[i], mlr[i], rup_forces[i], unf_forces[i], extension, force = simulate_forceramp(time,step,noiseval,spacer,cpl[1],[NoFP],ldr)
        else:
            fp_left[i], disse[i], totale_0FP[i], soloe[i], mlr[i], rup_forces[i], unf_forces[i], extension, force = simulate_constspeed(time,step,noiseval,spacer,cpl[1],[NoFP],speed)

    for i in range(num):
        if forceramp == True:
            fp_left[i], disse[i], totale_0FP_unf[i], soloe[i], mlr[i], rup_forces[i], unf_forces[i], extension, force = simulate_forceramp(time,step,noiseval,spacer+fingerprint[0][1][0],cpl[1],[NoFP],ldr)
        else:
            fp_left[i], disse[i], totale_0FP_unf[i], soloe[i], mlr[i], rup_forces[i], unf_forces[i], extension, force = simulate_constspeed(time,step,noiseval,spacer+fingerprint[0][1][0],cpl[1],[NoFP],speed)


    fig, ax = plt.subplots(figsize=(6, 2))
    totale_iso = totale[unf_forces==0]
    totale_dis = totale[unf_forces!=0]

    ax = sns.histplot(totale_0FP,ax=ax,label= "Energy dissipation", binwidth=25,kde=False,stat="density",color=colors[7])
    ax = sns.histplot(totale_0FP_unf,ax=ax,label= "Energy dissipation", binwidth=25,kde=False,stat="density",color=colors[4])
    ax = sns.histplot(totale_1FP,ax=ax,label= "Energy dissipation", binwidth=25,kde=False,stat="density",color=colors[6])

    ave_e0 = np.sum(totale_0FP) / num
    ave_e0_unf = np.sum(totale_0FP_unf) / num
    ave_e1 = np.sum(totale_1FP) / num

    np.savetxt("ave_e.txt",[ave_e0,ave_e0_unf,ave_e1],header=" #No FP #No FP unf #1 FP ")
    ax.set(xlabel="Energy [pN·nm]", xlim=(0,3500))
    #ax.set_xscale("log")
    fig.tight_layout()
    #note that startintg from seaborn 0.11.0 dist is switched to displot or histplot, good feature is that binwidth is available.

    plt.savefig("energy-hist.pdf")
    plt.close()
    #plt.show()

    os.chdir(current_dir)

    print ("Average energy: ({0:.3f},   {1:.3f})".format(ave_e0,ave_e1))
    print ("Note that the linker length is withFP {0:.3f} nm,  no FP {1:.3f} nm".format(spacer, spacer+fingerprint[0][1][0]))
    return ave_e0,ave_e0_unf,ave_e1,totale_0FP,totale_0FP_unf,totale_1FP

def plot_fd_trace(x,f):
    fig, ax = plt.subplots()
    ax.plot(x,f,marker=".")
    ax.set(xlabel="Extension [nm]",ylabel="Force [pN]")
    for item in fig.get_axes():
        format_figure(item)
    fig.tight_layout()
    #plt.show()
    return 0

    #Special version with filled color
def plot_edf_trace(x,d,f):
    fig, ax = plt.subplots()
    ax.plot(x,f,label="Tip-sample sepration",marker=",",color='black')
    ax.plot(d,f,label="AFM head height",marker=",",color='r')
    #ax.set_xlim(0, 250)
    #ax.set_ylim(0, 2500)

    ax.set(xlabel="Distance [nm]",ylabel="Force [pN]")
    ax.legend(loc ="upper right", prop={"size": fontsize-6})
    for item in fig.get_axes():
        format_figure(item)
    fig.tight_layout()
    return 0

def plot_fed_trace(u,x,d,f):
    fig, ax = plt.subplots()
    ax.plot(x,f,label="Tip-sample sepration",marker=",",color='black')
    #    ax.plot(d,f,label="AFM head height",marker=",",color='black')
    ax.set_xlim(0, 250)
    ax.set_ylim(0, 2500)
    if u > 0.:
        fbase = f_WLC(x,186,persist)
        ax.fill_between(x,fbase,f,where=fbase<=f,facecolor='red', alpha=0.8)

    ax.set(xlabel="Distance [nm]",ylabel="Force [pN]")
    #    ax.legend(loc ="upper right", prop={"size": fontsize-6})
    for item in fig.get_axes():
        format_figure(item)
    fig.tight_layout()
    return 0
def plot_dists2(cpl,fp):
    forces = np.linspace(0,3000,30000)
    fp_dist = pdfBell(forces,load,*fp[1][1:])
    #dist2 = pdfDudko(forces,load,19*kT,*testcp[1][1:])
    cp_dist = pdfBell(forces,load,*cpl[1][1:])

    fig, ax = plt.subplots()
    ax.plot(forces,fp_dist,label="Fingerprint")
    ax.plot(forces,cp_dist,label="Complex")
    #ax.plot(forces,fp_new)
    #ax.plot(forces,cp_new)
    ax.fill_between(forces,fp_dist,alpha=0.5)
    ax.fill_between(forces,cp_dist,alpha=0.5,color=colors[1])

    ax.set(xlabel="Force [pN]",ylabel="Probabilty density [1/pN]")
    ax.legend(loc ="upper left", prop={"size": fontsize-6})
    #ax.fill_between(forces,dist,alpha=0.2)


    fig.tight_layout()
    plt.savefig("dist.pdf")
    #plt.show()
    return 0

def plot_p(step,cpl,ldr):
    forces = np.linspace(0,150,1./step)
    p_lin = rate(forces,*cpl[1:])*step
    p_int = intrate(forces,step,ldr,*cpl[1:])

    fig, ax = plt.subplots()
    ax.plot(forces,p_lin,label="lin")
    ax.plot(forces,p_int,label="int")
    ax.legend(loc ="upper left", prop={"size": fontsize-6})
    for item in fig.get_axes():
        format_figure(item)

    fig.tight_layout()
    #ax.set_yscale("log")
    plt.show()
    return 0

def plot_e(result_e):
    plt.close()

    g = sns.catplot(data=result_e, kind="bar", x="lr", y="energy", hue="n_FP", ci=95, palette=[colors[7],colors[4],colors[6]], alpha=.9, height=6,aspect=1.2)
    g.set_axis_labels("Loading rate", "Energy (nN·nm)")
    g.legend.set_title("")
    plt.savefig(timeindex + "energy_nFP.pdf")
    plt.show()

    return 1

"""
Reference for parameters
# How long time?        Answer: ~400/speed, i.e: 0.5s@800nm/s
# Proper step?          Answer: for 10 points/nm, ~time/4000, i.e: 0.0001 (high force gets optimized then)
"""

print ('start at', now)

ydata = [0.43,0.46,0.44,0.49] # input eta observation first
correctedpara = correction(cpl=Scbd,fingerprint=FIVAR)

print ('Simulation took:', datetime.now()-now)
