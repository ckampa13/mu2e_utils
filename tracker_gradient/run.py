"""
Author: Cole Kampa
Date: 09-14-2020
Description: Fits simple linear gradient model that satisfies Maxwell's equations
(i.e. div(B)=0, assuming Bphi=0) to DS Tracker region magnetic field. Current use case is for KinKal corrections due to BField inhomogeneities. KinKal uses a constant: double grad_ with units tesla/mm, so these units are used here as well.
"""
import os
import numpy as np
import pandas as pd
import lmfit as lm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# user configs file
from user_configs import *

# set plot configs
plt.rcParams['figure.figsize'] = [10, 8] # larger figures
plt.rcParams['axes.grid'] = True         # turn grid lines on
plt.rcParams['axes.axisbelow'] = True    # put grid below points
plt.rcParams['grid.linestyle'] = '--'    # dashed grid
plt.rcParams.update({'font.size': 12.0})   # increase plot font size

# check if proper directories exist
def check_dirs(outdir=outdir):
    print('Checking directories')
    to_check = ['plots/', 'pickles/']
    for tc in to_check:
        if not os.path.exists(outdir+tc):
            os.mkdir(outdir+tc)

# read raw map information
# units = [mm, mm, mm, tesla, tesla, tesla]
def read_Bmap_txt(filename=mapdir+mapfile):
    print('Reading raw data')
    df = pd.read_csv(filename, header=None, names=['X', 'Y', 'Z', 'Bx', 'By', 'Bz'],
                     delim_whitespace=True, skiprows=4)
    print(df.head())
    return df

# shift and calculate
def calculate_Bmap_extras(df):
    print('Calculating extras for dataframe')
    df.eval('X = X + 3896', inplace=True) # want x=y=0 along magnet axis
    df.eval('R = (X**2 + Y**2)**(1/2)', inplace=True) # radius
    df.eval('Phi = arctan2(Y,X)', inplace=True) # phi
    df.eval('Br = Bx*cos(Phi)+By*sin(Phi)', inplace=True) # calculate Br for fitting
    df.eval('Bphi = -Bx*sin(Phi)+By*cos(Phi)', inplace=True) # Bphi calculated for completion...not needed
    print(df.head())
    return df

# query proper region
def query_tracker(df):
    print('Query for tracker region')
    # region requested by D. Brown -- Tracker region
    rmax = 650 # mm
    zcent = 10175 # mm
    zpm = 1500 # mm
    # query
    df = df.query(f'R <= {rmax} & Z >= {zcent-zpm} & Z <= {zcent+zpm}')
    df.reset_index(drop=True, inplace=True)
    return df

# pickle/unpickle data
def write_pickle_df(df, filename=outdir+'pickles/'+mapfile_pkl):
    print('Saving pickle')
    df.to_pickle(filename)

def read_pickle_df(filename=outdir+'pickles/'+mapfile_pkl):
    print('Loading pickle')
    return pd.read_pickle(filename)

# model function
def maxwell_gradient(r, z, **params):
    Bz = params['dBzdz'] * z + params['B0']
    Br = - r / 2 * params['dBzdz']
    return np.concatenate([Bz, Br])

# fit data
def run_fit(df, model_func=maxwell_gradient):
    print('Running fit')
    model = lm.Model(model_func, independent_vars=['r','z'])
    params = lm.Parameters()
    params.add('dBzdz', value=0)
    params.add('B0', value=0)
    samples = np.concatenate([df.Bz.values, df.Br.values])
    result = model.fit(samples, r=df.R.values, z=df.Z.values, params=params,
                       method='least_squares', fit_kws={'loss': 'linear'})
    result_array = result.eval().reshape((2,-1))
    df.loc[:, 'Bz_fit'] = result_array[0]
    df.loc[:, 'Br_fit'] = result_array[1]
    df.eval('Bz_res = Bz - Bz_fit', inplace=True)
    df.eval('Br_res = Br - Br_fit', inplace=True)
    df.eval('Bz_res_rel = (Bz - Bz_fit)/Bz', inplace=True)
    df.eval('Br_res_rel = (Br - Br_fit)/Br', inplace=True)
    df.to_pickle(outdir+'pickles/df_results.pkl')
    print(result.fit_report())
    return result, df

def write_result(result, filename=outdir+'fit_result.txt'):
    with open(filename, 'w+') as f:
        f.write(result.fit_report())

# plotting
def make_plots(df, result):
    # wireframes
    df0 = df.query('Y==0')
    df0.sort_values(by=['X', 'Z'])
    xs = df0.X.unique()
    zs = df0.Z.unique()
    elevs = [27, 40]
    azims = [21, 73]
    for B, el, az in zip(['Bz', 'Br'], elevs, azims):
        X = df0['X'].values.reshape((len(xs), len(zs)))
        Z = df0['Z'].values.reshape((len(xs), len(zs)))
        B_fit = df0[f'{B}_fit'].values.reshape((len(xs), len(zs)))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df0.X, df0.Z, df0[B], c='black', s=1, label='Data (Mau13)')
        ax.plot_wireframe(X, Z, B_fit, color='green', label='Fit (Linear Gradient)')
        ax.view_init(elev=el, azim=az)
        ax.set_xlabel('X [mm]', labelpad=30)
        ax.set_ylabel('Z [mm]', labelpad=30)
        ax.set_zlabel(f'{B} [Tesla]', labelpad=30)
        ax.xaxis.set_rotate_label(False)
        ax.yaxis.set_rotate_label(False)
        ax.zaxis.set_rotate_label(False)
        plt.legend()
        fig.suptitle(f'{B} vs. X, Z for Y==0')
        fig.tight_layout(rect=(0,0.04,1,1))
        fig.savefig(outdir+f'plots/{B}_vs_X_Z_Y=0.pdf')
        fig.savefig(outdir+f'plots/{B}_vs_X_Z_Y=0.png')
    # residual histograms
    plt.rcParams['figure.figsize'] = [16, 8] # larger figures
    plt.rcParams.update({'font.size': 18.0})   # increase plot font size
    label_temp = r'$\mu = {0:.3E}$'+ '\n' + 'std' + r'$= {1:.3E}$' + '\n' +  'Integral: {2}\n' + 'Underflow: {3}\nOverflow: {4}'
    N_bins = 200
    lsize = 16
    for res in ['res', 'res_rel']:
        if res == 'res':
            xlabel_z = r'$\Delta B_z$'+' [Tesla]'
            xlabel_r = r'$\Delta B_r$'+' [Tesla]'
            title_ = ''
            fname_ = ''
            scale = 'linear'
            xmin_z = df[[f'Bz_{res}',f'Br_{res}']].min().min()
            xmax_z = df[[f'Bz_{res}',f'Br_{res}']].max().max()+1e-5
            xmin_r = xmin_z
            xmax_r = xmax_z
        else:
            xlabel_z = r'$\Delta B_z / B_z$'
            xlabel_r = r'$\Delta B_r / B_r$'
            title_ = 'Relative '
            fname_ = '_relative'
            scale = 'log'
            xmin_z = -1e-2
            xmax_z = 1e-2
            xmin_r = -100
            xmax_r = 100
        under_z = (df[f'Bz_{res}'] < xmin_z).sum()
        over_z = (df[f'Bz_{res}'] >= xmax_z).sum()
        under_r = (df[f'Br_{res}'] < xmin_r).sum()
        over_r = (df[f'Br_{res}'] >= xmax_r).sum()
        bins_z = np.linspace(xmin_z, xmax_z, N_bins+1)
        bins_r = np.linspace(xmin_r, xmax_r, N_bins+1)
        fig, axs = plt.subplots(1, 2)
        axs[0].hist(df[f'Bz_{res}'], bins=bins_z, label=label_temp.format(df[f'Bz_{res}'].mean(), df[f'Bz_{res}'].std(), len(df)-under_z-over_z, under_z, over_z))
        axs[0].set(xlabel=xlabel_z, ylabel="Count", yscale=scale)
        axs[0].legend(prop={'size': lsize})
        axs[1].hist(df[f'Br_{res}'], bins=bins_r, label=label_temp.format(df[f'Br_{res}'].mean(), df[f'Br_{res}'].std(), len(df)-under_r-over_r, under_r, over_r))
        axs[1].set(xlabel=xlabel_r, ylabel="Count", yscale=scale)
        axs[1].legend(prop={'size': lsize})
        title_main=f'Linear Gradient Tracker Region {title_}Residuals'
        fig.suptitle(title_main)
        fig.tight_layout(rect=[0,0,1,1])
        plot_file = outdir+f'plots/B{fname_}_residuals_hist'
        fig.savefig(plot_file+'.pdf')
        fig.savefig(plot_file+'.png')


if __name__ == '__main__':
    # check if proper directories exist
    check_dirs()
    # calculate data from raw file or pickle from previous calculation
    pickle_exists = os.path.exists(outdir+'pickles/'+mapfile_pkl)
    if pickle_exists and usepickle:
        df_tracker = read_pickle_df()
    else:
        df_raw = read_Bmap_txt()
        df_calc = calculate_Bmap_extras(df_raw)
        df_tracker = query_tracker(df_calc)
        write_pickle_df(df_tracker)
    # run linear gradient fit
    result, df_tracker = run_fit(df_tracker)
    write_result(result)
    # make some plots
    make_plots(df_tracker, result)
