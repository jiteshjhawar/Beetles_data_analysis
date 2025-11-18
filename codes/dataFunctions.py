import numpy as np

# def calc_store_omega_df(df):
#     df['omega'] = np.nan
#     i = 0
#     for f in np.arange(df.frame[0],df.frame[len(df.frame)-1]+1):
#         y_centroid = np.nanmean(df.y[df.frame==f])
#         x_centroid = np.nanmean(df.x[df.frame==f])

#         r_y_vec = df.y[df.frame==f] - y_centroid
#         r_x_vec = df.x[df.frame==f] - x_centroid
#         # normalization
#         r_y_norm_vec = (r_y_vec)/ (np.sqrt(r_y_vec**2 + r_x_vec**2) + 1e-15)
#         r_x_norm_vec = (r_x_vec)/ (np.sqrt(r_y_vec**2 + r_x_vec**2) + 1e-15)

#         v_y_vec = df.vy[df.frame==f]
#         v_x_vec = df.vx[df.frame==f]
#         # normalization
#         v_y_norm_vec = (v_y_vec)/ (np.sqrt(v_y_vec**2 + v_x_vec**2) + 1e-15)
#         v_x_norm_vec = (v_x_vec)/ (np.sqrt(v_y_vec**2 + v_x_vec**2) + 1e-15)

#         # omega_ind = [np.cross([r_x_vec[i],r_y_vec[i]],[v_x_vec[i],v_y_vec[i]]) for i in np.arange(i,i+len(r_y_vec))]
#         omega_ind = np.zeros(len(r_y_vec))
#         ## some complexity because next index r_y_vec for next frame does not starts from 0
#         j = 0
#         for i in np.arange(i,i+len(r_y_vec)):
#             omega_ind[j] = np.cross([r_y_norm_vec[i],r_x_norm_vec[i]],[v_y_norm_vec[i],v_x_norm_vec[i]])
#             j+=1
#         i+=1 #increasing index by 1 to start fom the index of element starting next frame

#         df.omega[df.frame==f] = omega_ind
#     return df

def calc_store_omega_df(df):
    df['omega'] = np.nan
    df['dist_center'] = np.nan
#     i = 0
    for f in np.arange(df.frame[0],df.frame[len(df.frame)-1]+1):
        y_centroid = np.nanmean(df.y[df.frame==f].values)
        x_centroid = np.nanmean(df.x[df.frame==f].values)

        r_y_vec = df.y[df.frame==f].values - y_centroid
        r_x_vec = df.x[df.frame==f].values - x_centroid
        # normalization
        r_y_norm_vec = (r_y_vec)/ (np.sqrt(r_y_vec**2 + r_x_vec**2) + 1e-15)
        r_x_norm_vec = (r_x_vec)/ (np.sqrt(r_y_vec**2 + r_x_vec**2) + 1e-15)

        v_y_vec = df.vy[df.frame==f].values
        v_x_vec = df.vx[df.frame==f].values
        # normalization
        v_y_norm_vec = (v_y_vec)/ (np.sqrt(v_y_vec**2 + v_x_vec**2) + 1e-15)
        v_x_norm_vec = (v_x_vec)/ (np.sqrt(v_y_vec**2 + v_x_vec**2) + 1e-15)

        # omega_ind = [np.cross([r_x_vec[i],r_y_vec[i]],[v_x_vec[i],v_y_vec[i]]) for i in np.arange(i,i+len(r_y_vec))]
        omega_ind = np.zeros(len(r_y_vec))
        dist_ind = np.zeros(len(r_y_vec))
        ## some complexity because next index r_y_vec for next frame does not starts from 0
        j = 0
        for i in np.arange(0,len(r_y_vec)):
            omega_ind[j] = np.cross([r_x_norm_vec[i],r_y_norm_vec[i]],[v_x_norm_vec[i],v_y_norm_vec[i]])
            dist_ind[j] = np.sqrt(r_y_vec[i]**2 + r_x_vec[i]**2)
            j+=1
#         i+=1 #increasing index by 1 to start fom the index of element starting next frame

        df.omega[df.frame==f] = omega_ind
        df.dist_center[df.frame==f] = dist_ind
    return df
    
def calc_pol_from_df(df):
    cirOP = []
    for i in df['frame'].unique():
        cirOP.append(np.nanmean(df[(df['frame']==i)&(df['omega']!=0)]['omega']))
    return np.array(cirOP)

    
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

def drift_diffusion_raw(x,Dt,dt,inc):
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
    # plot_op = np.arange(-1,1,inc)
    # bin = np.min(plot_op)
    # avgDrift = np.zeros(len(plot_op))
    # avgDiffusion = np.zeros(len(plot_op))
    
    # stderrDrift = np.zeros(len(plot_op))
    # stderrDiffusion = np.zeros(len(plot_op))
    # i = 0
    # while bin < np.max(plot_op):
    #     ind = np.where((x > bin) & (x <= bin+inc))
    #     avgDrift[i] = np.nanmean(drift[ind])
    #     avgDiffusion[i] = np.nanmean(diffusion[ind])
        
    #     stderrDrift[i] = np.nanstd(drift[ind]) / np.sqrt(len(drift[ind]))
    #     stderrDiffusion[i] = np.nanstd(diffusion[ind]) / np.sqrt(len(diffusion[ind]))
        
    #     i+=1
    #     bin+= inc
    return [drift, diffusion]


def cirOP_concentric_rings(df):
    cirOP_cent50 = []
    cirOP_cent100 = []
    cirOP_cent200 = []
    for i in df['frame'].unique():
        cirOP_cent50.append(np.nanmean(df[(df['frame']==i)&(df['omega']!=0)&(df['dist_center']<50)]['omega']))
        cirOP_cent100.append(np.nanmean(df[(df['frame']==i)&(df['omega']!=0)&(df['dist_center']>=50)&(df['dist_center']<100)]['omega']))
        cirOP_cent200.append(np.nanmean(df[(df['frame']==i)&(df['omega']!=0)&(df['dist_center']>=100)]['omega']))

    return cirOP_cent50, cirOP_cent100, cirOP_cent200


def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 
    
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag))