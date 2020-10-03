import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
from PyPDF2 import PdfFileMerger

# user configs file
# from user_configs import *
# from user_configs import ntup_dir, outdir, usepickle
from user_configs import outdir

# set plot configs
plt.rcParams['figure.figsize'] = [10, 8] # larger figures
plt.rcParams['axes.grid'] = True         # turn grid lines on
plt.rcParams['axes.axisbelow'] = True    # put grid below points
plt.rcParams['grid.linestyle'] = '--'    # dashed grid
plt.rcParams.update({'font.size': 12.0})   # increase plot font size

# globals
NUM_CPU = multiprocessing.cpu_count()

# functions
# check if proper directories exist
def check_dirs(outdir=outdir):
    print('Checking plotting directories')
    pkl_files = os.listdir(outdir+'pickles/mu_bottle/')
    configs = list(set(f.split('.')[0] for f in pkl_files))
    to_check = ['plots/', 'plots/mu_bottle_analysis/', 'plots/mu_bottle_analysis/merged/']+[f'plots/mu_bottle_analysis/{c}/' for c in configs] #, 'plots/mu_bottle_analysis/mau12/',
                #'plots/mu_bottle_analysis/mau13/', 'plots/mu_bottle_analysis/mau14/',
                #'pickles/', 'pickles/mu_bottle/']
    for tc in to_check:
        if not os.path.exists(outdir+tc):
            os.mkdir(outdir+tc)

def get_pkl_list(outdir=outdir):
    return os.listdir(outdir+'pickles/mu_bottle/')

def load_pkl(filename):
    df = pd.read_pickle(filename)
    return df

# plotting functions
def make_scatter3d_plot(df, df0, title, fname):
    fig = go.Figure(data=[go.Scatter3d(
        x=df['x']*1e-3 + 3.896,
        y=df['y']*1e-3,
        z=df['z']*1e-3,
        mode='markers',
        marker=dict(
            size=2,
            # line=dict(width=0.01, color='grey'),# color='LightSlateGrey'),
            color=df0['p'],
            colorscale='Viridis',
            colorbar=dict(thickness=20, title='Generator<br>Momentum [MeV]',),
            opacity=1.0,
        )
    )])

    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=-0.2, z=0.325),
        eye=dict(x=-1.65, y=0.3, z=.95)
    )

    sf = 0.6

    fig.update_layout(
        margin=dict(l=5, r=0, b=0, t=50),
        title=title,
        font=dict(
            family='sans-serif',
            size= 14,#18,
        ),
        scene_camera=camera,
        scene=dict(
            xaxis=dict(
                title='X [m]',
                range=[-0.9, 0.9],
            ),
            yaxis=dict(
                title='Y [m]',
                range=[-0.9, 0.9],
            ),
            zaxis=dict(
                title='Z [m]',
                range=[4., 14.],
            ),
            # xaxis_title='X [m]',
            # yaxis_title='Y [m]',
            # zaxis_title='Z [m]',
            # aspectmode='data',
            aspectmode='manual',
            aspectratio=dict(x=sf, y=sf, z=sf*4.5)
        ),
    )

    # manually set axis ranges
    # fig.update_xaxes(range=[-0.9, 0.9])
    # fig.update_yaxes(range=[-0.9, 0.9])
    # fig.update_zaxes(range=[4, 14])

    # fig.show()

    fig.write_html(fname+'.html')

    fig.update_layout(
        autosize=False,
        width=1000,
        height=400,
    )

    fig.write_image(fname+'.pdf')
    fig.write_image(fname+'.png')


def make_momentum_hist(df, df0, xcol, xlabel, dx, title, filename):
    label_temp = '{0}\n' + r'$\mu = {1:.3E}$'+ '\n' + 'std' + r'$= {2:.3E}$' + '\n' +  'Integral: {3}\n' + 'Underflow: {4}\nOverflow: {5}'

    xmin = np.min(np.array([df[xcol].values, df0[xcol].values]))
    xmax = np.max(np.array([df[xcol].values, df0[xcol].values])) + 1e-7
    bins = np.arange(xmin, xmax+dx, dx)
    fig = plt.figure()
    plt.hist(df0[xcol], bins=bins, label=label_temp.format('t = 0 ns (Generator)', df0[xcol].mean(), df0[xcol].std(), len(df0), 0, 0))
    plt.hist(df[xcol], bins=bins, histtype='step', label=label_temp.format('t = 700 ns', df[xcol].mean(), df[xcol].std(), len(df), 0, 0))
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.title(title)
    plt.legend()
    fig.tight_layout()
    fig.savefig(filename+'.pdf')
    fig.savefig(filename+'.png')

def gen_plots(config):
    str0_ = config[:5].capitalize()
    # str0 = rf'$\text{{{str0_}}}$'
    if 'full' in config:
        str1_ = ' (Full DS)'
        # str1 = r'$\text{ (Full DS, Isometric) } \mu^-$'
        N_gen = 2.5e8 # bad... should get in data processing somehow
    elif '2a' in config:
        str1_ = ' (Region 2a)'
        # str1_ = ' (Targeted, Region 2a)'
        # str1 = r'$\text{ (Targeted, Region 2a)}$'
        N_gen = 5e7
    elif '2b' in config:
        str1_ = ' (Region 2b)'
        # str1_ = ' (Targeted, Region 2b)'
        # str1 = r'$\text{ (Targeted, Region 2b)}$'
        N_gen = 5e7
    elif '2c' in config:
        str1_ = ' (Region 2c)'
        # str1_ = ' (Targeted, Region 2c)'
        # str1 = r'$\text{ (Targeted, Region 2c)}$'
        N_gen = 5e7
    elif '3' in config:
        str1_ = ' (Region 3)'
        # str1_ = ' (Targeted, Region 3)'
        # str1 = r'$\text{ (Targeted, Region 3)}$'
        N_gen = 5e7
    df = pd.read_pickle(outdir+f'pickles/mu_bottle/{config}.trap.df.pkl')
    df0 = pd.read_pickle(outdir+f'pickles/mu_bottle/{config}.trap-gen.df.pkl')
    ### 3d scatter plot
    # make_scatter3d_plot(df, df0, str0 + str1 + rf"$\text{{ Trapped Particle Locations at }} t=700 \text{{ns}}<br> N_{{\text{{gen.}}}} = {N_gen:0.1E}, N_{{\text{{trapped}}}} = {len(df)}$", f'{outdir}plots/mu_bottle_analysis/{config}/{config}_scatter3d_locations_gen_mom')
    # make_scatter3d_plot(df, df0, rf"$\text{{{str0_}{str1_} Trapped Particle Locations at }} t=700 \text{{ns}}\linebreak N_{{\text{{gen.}}}} = {N_gen:0.1E}, N_{{\text{{trapped}}}} = {len(df)}$", f'{outdir}plots/mu_bottle_analysis/{config}/{config}_scatter3d_locations_gen_mom')
    make_scatter3d_plot(df, df0, rf"$\text{{{str0_}{str1_}: Particles at }} t=700 \text{{ ns, }} N_{{\text{{gen.}}}} = {N_gen:0.1E}, N_{{\text{{trapped}}}} = {len(df)}$", f'{outdir}plots/mu_bottle_analysis/{config}/{config}_scatter3d_locations_gen_mom')
    ### histogram of initial and final momentum
    make_momentum_hist(df, df0, "p", "Momentum [MeV]", 1., str0_+str1_+' Trapped Muon Momentum', f'{outdir}plots/mu_bottle_analysis/{config}/{config}_trapped_p_hist')
    make_momentum_hist(df, df0, "pz", "Momentum, "+ r"$p_z$"+" [MeV]", 1., str0_+str1_+r' Trapped Muon $p_z$', f'{outdir}plots/mu_bottle_analysis/{config}/{config}_trapped_pz_hist')
    ### cos(theta)
    df.eval('costheta = pz / p', inplace=True)
    df0.eval('costheta = pz / p', inplace=True)
    make_momentum_hist(df, df0, "costheta", r'$\cos{(\theta)}$'+'(momentum vector)', 0.01, str0_+str1_+' Trapped Muon '+r'$\cos{(\theta)}$', f'{outdir}plots/mu_bottle_analysis/{config}/{config}_trapped_costheta_hist')


def merge_pdfs(configs, Bmap, filename):
    pdfs = [outdir+f'plots/mu_bottle_analysis/{Bmap}-{config}/{Bmap}-{config}_{filename}.pdf' for config in configs]
    merger = PdfFileMerger()
    for pdf in pdfs:
        merger.append(pdf)
    merger.write(outdir+f'plots/mu_bottle_analysis/merged/{Bmap}_{filename}_merged.pdf')
    merger.close()


if __name__ == '__main__':
    check_dirs() # make sure output directories exist
    pkl_list = get_pkl_list() # get available pickle files
    configs = [f.split('.')[0] for f in pkl_list]
    configs_unique = sorted(list(set(configs)))
    df_types = [f.split('.')[1] for f in pkl_list]

    # generate plots for each config (in parallel)
    Parallel(n_jobs=NUM_CPU)(delayed(gen_plots)(config) for config in tqdm(configs_unique, file=sys.stdout, desc='config #'))

    # merge plots
    maps = ['mau12', 'mau13', 'mau14']
    r_configs = ['full', 'target-2a', 'target-2b', 'target-2c', 'target-3']
    plotfiles = ['scatter3d_locations_gen_mom', 'trapped_p_hist', 'trapped_pz_hist', 'trapped_costheta_hist']
    for f in plotfiles:
        for Bmap in maps:
            merge_pdfs(r_configs, Bmap, f)

