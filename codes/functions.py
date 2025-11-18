import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import optimize

def drift_diffusion(x,Dt,dt,inc):
    op = np.linspace(-1,1,201)
#     x = X200

    drift = np.zeros(len(x))
    diffusion = np.zeros(len(x))
#     Dt = 3
#     dt = 1/30

    for i in range(len(x)-Dt):
        drift[i] = (x[i+Dt] - x[i]) / (Dt*dt)
        diffusion[i] = np.square((x[i+1] - x[i])) / (dt)


#     inc = 0.02
    plot_op = np.arange(-1,1,inc)
    bin = np.min(plot_op)
    avgDrift = np.zeros(len(plot_op))
    avgDiffusion = np.zeros(len(plot_op))
    stderrDrift = np.zeros(len(plot_op))
    stderrDiffusion = np.zeros(len(plot_op))
    i = 0
    while bin < np.max(plot_op):
        ind = np.where((x > bin) & (x <= bin+inc))
        avgDrift[i] = np.nanmean(drift[ind])
        avgDiffusion[i] = np.nanmean(diffusion[ind])
        stderrDrift[i] = np.nanstd(drift[ind]) / np.sqrt(len(drift[ind]))
        stderrDiffusion[i] = np.nanstd(diffusion[ind]) / np.sqrt(len(diffusion[ind]))
        i+=1
        bin+= inc
    return [avgDrift, avgDiffusion, stderrDrift, stderrDiffusion, plot_op]


def func_drift(x, a, b, c, d):

    return a*x**3 + b*x**2 + c*x + d


def func_diff(x, a, b, c):

    return a*x**2 + b*x + c


def simulate_fitted_sde(drift, diffusion, op):
    
    global dri_a, dri_b, dri_c, dri_d, diff_a, diff_b, diff_c
    
    #fit functions
    ind = np.where(~np.isnan(drift))
    y_value = drift[ind]
    x_value = op[ind]
    popt, pcov = curve_fit(func_drift, x_value, y_value)
    dri_a, dri_b, dri_c, dri_d = popt


    ind = np.where(~np.isnan(diffusion))
    x_value = op[ind]
    y_value = diffusion[ind]
    popt, pcov = curve_fit(func_diff, x_value, y_value)
    diff_a, diff_b, diff_c = popt

    #simulate sde
    repete = 5
    x_sim = []
    for rep in range(repete):
        iterations = 21000
        x = np.zeros(iterations, dtype = np.float32)
        dt = 1/30
        x[0] = np.random.rand()*2 - 1
        for i in range(iterations-1):
            eta = np.random.rand()*2 - 1

    #         dri_a, b = [-3.27732754, -0.0291525]
            drift_sim = (dri_a*x[i]**3 + dri_b*x[i]**2 + dri_c*x[i] + dri_d)

    #         a, b, c = [ 0.32057659, -0.02837962,  0.33008611]
            diffusion_sim = np.sqrt(diff_a*x[i]**2 + diff_b*x[i] + diff_c)
        
            x[(i+1)] = x[i]  + drift_sim*dt + diffusion_sim*np.sqrt(dt)*eta


            x_sim.append(x[i])
    x_sim = np.array(x_sim)
    
    return x_sim


def simulate_ibm_fitted_sde(drift, diffusion, op, N, model):
    
    diffusion[diffusion > 5] = np.nan
    global p_best
    ind = np.where(~np.isnan(diffusion))
    if model == 'tern':
        p_global=[1,1,1,1]
    elif model == 'tern_p_neg':
        p_global = [1,1,1,1,1]
    elif model == 'penta':
        p_global = [1,1,1,1,1,1,1]
    elif model == 'pp':
        p_global=[1,1,1]
    z = np.array(op[ind])
#     ind = np.where(~np.isnan(drift))
    p_best,success=optimize.leastsq(err_global, p_global,args=(z,drift[ind],diffusion[ind], N, model),maxfev=40000)
    
    repete = 1
    x_sim = []
    for rep in range(repete):
        iterations = 50000
        x = np.zeros(iterations, dtype = np.float32)
        dt = 1/30
        if model == 'tern':
            r1f = p_best[0]
            r1b = p_best[1]
            r2 = p_best[2]
            r3 = p_best[3]
        elif model == 'tern_p_neg':
            r1f = p_best[0]
            r1b = p_best[1]
            r2 = p_best[2]
            r3 = p_best[3]
            r2_neg = p_best[4]
        elif model == 'penta':
            r1f = p_best[0]
            r1b = p_best[1]
            r2 = p_best[2]
            r3 = p_best[3]
            r4 = p_best[4]
            r5 = p_best[5]
            r2_neg = p_best[6]
        elif model == 'pp':
            r1f = p_best[0]
            r1b = p_best[1]
            r2 = p_best[2]
            
#         N = 50
        x[0] = np.random.rand()*2 - 1
        for i in range(iterations-1):
#             eta = np.random.rand()*2 - 1
            eta = np.random.normal()
            if model == 'tern':
                drift_sim = -r1f*(1+x[i])/2 + r1b*(1-x[i])/2 + r3*x[i]*(1-x[i]**2)/4
                diffusion_sim = np.sqrt((r1f + r1b + (1-x[i]**2)*r2 + (1-x[i]**2)*r3/2) / N)
                
            elif model == 'tern_p_neg':
                drift_sim = -r1f*(1+x[i])/2 + r1b*(1-x[i])/2 + r3*x[i]*(1-x[i]**2)/4 -r2_neg*x[i]
                diffusion_sim = np.sqrt((r1f + r1b + (1-x[i]**2)*r2 + (1-x[i]**2)*r3/2 + (1+x[i]**2)*r2_neg) / N)
                
            elif model == 'penta':
                drift_sim = -r1f*(1+x[i])/2 + r1b*(1-x[i])/2 + r3*x[i]*(1-x[i]**2)/4 + r4*x[i]*(1-x[i]**2)/4 - r5*((1-x[i]**2)**2)*x[i]/16 - r2_neg*x[i]
                diffusion_sim = np.sqrt((r1f + r1b + (1-x[i]**2)*r2 + (1-x[i]**2)*r3/2 + (1-x[i]**4)*r4/4 + r5*((1-x[i]**2)**2)/8 + (1+x[i]**2)*r2_neg) / N)
            elif model == 'pp':
                drift_sim = -r1f*(1+x[i])/2 + r1b*(1-x[i])/2
                diffusion_sim = np.sqrt((r1f + r1b + (1-x[i]**2)*r2) / N)
            
            x[(i+1)] = x[i] + drift_sim*dt + diffusion_sim*np.sqrt(dt)*eta

            x_sim.append(x[i])
    x_sim = np.array(x_sim)
    
    return x_sim

def KL(cirOP, Dt, N, inc, hist_inc, ibm, model):
    
    global drift, diffusion, op, values_data, values_sim, bins_data, bins_sim  
    global seDrift, seDiffusion
    
    drift, diffusion, seDrift, seDiffusion, op = drift_diffusion(cirOP[:,0],Dt,1/30,inc)
    
    if ibm == False:
        x_sim = simulate_fitted_sde(drift, diffusion, op)
    else:
        x_sim = simulate_ibm_fitted_sde(drift, diffusion, op, N, model)
    
    
    bin_range = np.arange(-1,1,hist_inc)
    
    counts, bins_data = np.histogram(cirOP,bins=bin_range)
    values_data = counts/np.sum(counts) + 1e-5

    counts, bins_sim = np.histogram(x_sim,bins=bin_range)
    values_sim = counts/np.sum(counts) + 1e-5
    
    a = values_data#np.asarray(a, dtype=np.float)
    b = values_sim#np.asarray(b, dtype=np.float)
    
    a_b = np.sum(np.where((a != 0), a * np.log(a / b), 0))
    b_a = np.sum(np.where((b != 0), b * np.log(b / a), 0))
    
    ind = np.where(~np.isnan(drift))
    r_sq = 1 - (np.nanmean(err_global(p_best,op[ind],drift[ind],diffusion[ind], N, model)**2)/np.nanvar(np.concatenate([drift[ind]/np.sqrt(np.nanmean((drift[ind])**2)),diffusion[ind]/np.sqrt(np.nanmean((diffusion[ind])**2))])))
    
    return r_sq, (a_b + b_a)/2

def func_drift_ibm_pp(z, r1f, r1b, r2):

    return -r1f*(1+z)/2 + r1b*(1-z)/2

def func_drift_ibm_tern(z, r1f, r1b, r2, r3):

    return -r1f*(1+z)/2 + r1b*(1-z)/2 + r3*z*(1-z**2)/4

def func_drift_ibm_tern_p_neg(z, r1f, r1b, r2, r3,r2_neg):

    return -r1f*(1+z)/2 + r1b*(1-z)/2 + r3*z*(1-z**2)/4 -r2_neg*z

def func_drift_ibm_penta(z, r1f, r1b, r2, r3, r4, r5, r2_neg):
    
    return -r1f*(1+z)/2 + r1b*(1-z)/2 + r3*z*(1-z**2)/4 + r4*z*(1-z**2)/4 - r5*((1-z**2)**2)*z/16 - r2_neg*z

# def func_drift50(z, r1, r4, r5):

#     return -r1*z -r4*z - r5*z*(3+z**2)/4

def func_diff_ibm_pp(z, r1f, r1b, r2, N):
    
    return (r1f + r1b + (1-z**2)*r2) / N

def func_diff_ibm_tern(z, r1f, r1b, r2, r3, N):
    
    return (r1f + r1b + (1-z**2)*r2 + (1-z**2)*r3/2) / N

def func_diff_ibm_tern_p_neg(z, r1f, r1b, r2, r3, r2_neg, N):
    
    return (r1f + r1b + (1-z**2)*r2 + (1-z**2)*r3/2+ (1+z**2)*r2_neg) / N

def func_diff_ibm_penta(z, r1f, r1b, r2, r3, r4, r5, r2_neg, N):
    return (r1f + r1b + (1-z**2)*r2 + (1-z**2)*r3/2+ (1-z**4)*r4/4 + r5*((1-z**2)**2)/8 + (1+z**2)*r2_neg) / N


def err_drift(z,popt_drift,y, model):
#     error = (func_drift_ibm(z, *popt_drift)-y)
#     return (func_drift_ibm(z, *popt_drift)-y)# / np.sqrt((np.mean(y**2)))
    if model == 'tern':
        return (func_drift_ibm_tern(z, *popt_drift)-y) / np.sqrt((np.nanmean(y**2)))
    elif model == 'tern_p_neg':
        return (func_drift_ibm_tern_p_neg(z, *popt_drift)-y) / np.sqrt((np.nanmean(y**2)))
    elif model == 'penta':
        return (func_drift_ibm_penta(z, *popt_drift)-y) / np.sqrt((np.nanmean(y**2)))
    elif model == 'pp':
        return (func_drift_ibm_pp(z, *popt_drift)-y) / np.sqrt((np.nanmean(y**2)))

# def err_drift50(z,popt_drift,y):
#     return (func_drift50(z, popt_drift[0], popt_drift[3],popt_drift[4])-y) / (np.mean(y**2)) 

def err_diff(z,popt_diff,y, N, model):
#     error = (func_diff_ibm(z, *popt_diff, N)-y)
#     return (func_diff_ibm(z, *popt_diff, N)-y)# / np.sqrt((np.mean(y**2)))
    if model == 'tern':
        return (func_diff_ibm_tern(z, *popt_diff, N)-y) / np.sqrt((np.nanmean(y**2)))
    elif model == 'tern_p_neg':
        return (func_diff_ibm_tern_p_neg(z, *popt_diff, N)-y) / np.sqrt((np.nanmean(y**2)))  
    elif model == 'penta':
        return (func_diff_ibm_penta(z, *popt_diff, N)-y) / np.sqrt((np.nanmean(y**2)))
    elif model == 'pp':
        return (func_diff_ibm_pp(z, *popt_diff, N)-y) / np.sqrt((np.nanmean(y**2)))


def err_global(p,z,avgDrift,avgDiffusion, N, model):
    
#     avgDrift = (avgDrift - np.nanmean(avgDrift)) / np.nanstd(avgDrift)
#     avgDiffusion = (avgDrift - np.nanmean(avgDiffusion)) / np.nanstd(avgDiffusion)
    
    err_dr = err_drift(z, p, avgDrift, model)
    err_di = err_diff(z, p, avgDiffusion, N, model)
    err1=np.concatenate((err_dr,err_di))
    if (np.max(p) > 100) or (np.min(p) <= 0):
        return err1*9999
    err0=err1
    return err0

def plot_output(N,ibm):
    if ibm == False:
        plt.figure(figsize=[21,7])
        plt.subplot(1,3,1)
        plt.plot(op, drift, 'o',np.arange(-1,1,0.02), func_drift(np.arange(-1,1,0.02), dri_a, dri_b, dri_c, dri_d),'r-')
        plt.hlines(0,-1,1)
        plt.subplot(1,3,2)
        plt.plot(op, diffusion, 'o',np.arange(-1,1,0.02), func_diff(np.arange(-1,1,0.02), diff_a, diff_b, diff_c),'r-')
        plt.subplot(1,3,3)
        plt.step(bins_data[:-1],values_data)
        plt.step(bins_sim[:-1],values_sim)
        plt.legend(['data','simulation'])
        
    else:
        
        plt.figure(figsize=[20,7])
        plt.subplot(1,3,1)
        plt.plot(op, drift, 'o' , np.arange(-1,1,0.02), func_drift_ibm(np.arange(-1,1,0.02), *p_best), 'r-')
        plt.hlines(0,-1,1)
#         plt.xlim([-0.8, 0.8])
#         plt.axvline()
        plt.legend(['Data','Model fitted'])
        plt.ylabel(r'$F(M)$', fontsize=16)
        plt.xlabel(r'$M$',fontsize=16)
#         plt.title(r'$N=50$',fontsize=16)
        plt.subplot(1,3,2)
        plt.plot(op, diffusion, 'o',np.arange(-1,1,0.02), func_diff_ibm(np.arange(-1,1,0.02), *p_best, N),'r-')
        plt.ylim([0,1])
        plt.subplot(1,3,3)
        plt.step(bins_data[:-1],values_data)
        plt.step(bins_sim[:-1],values_sim)
        plt.legend(['data','simulation'])
        
    return 0