import os
import sys
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
# import uproot4 as uproot
import uproot
# user configs file
from user_configs import ntup_dir, outdir

# globals
cols_keep = ['run','subrun','evt','time','x','y','z','px','py','pz']
NUM_CPU = multiprocessing.cpu_count()

# functions
# check if proper directories exist
def check_dirs(outdir=outdir):
    print('Checking directories')
    to_check = ['pickles', 'pickles/mu_bottle/']
    for tc in to_check:
        if not os.path.exists(outdir+tc):
            os.mkdir(outdir+tc)

# find unique ntuple files and split by field map
def find_ntuple_files(ntup_dir=ntup_dir):#, config_str='full'):
    file_set = set() # contains unique entries
    config_set = set() # contains unique configurations
    ntuple_files = []
    # recursively search through base ntuple directory
    for root, dirs, files in os.walk(ntup_dir):
        # for each directory, loop through the files
        for name in files:
            # split filename into components
            try:
                data_tier, owner, description, configuration, sequencer, file_format = name.split('.')
            except:
                data_tier, owner, description, configuration, sequencer, file_format = 6*[None]
            # check configuration
            # if config_str in configuration:
                # check if 'nts' (ntuple) and ROOT file
            if (data_tier == 'nts') & (file_format == 'root'):
                # if (name[:3] == 'nts') & (name[-5:] == '.root'):
                # check if ntuple is unique from current list
                if not name in file_set:
                    # add path+filename to file list
                    ntuple_files.append(os.path.join(root,name))
                # add to file set
                file_set.add(name)
                config_set.add(configuration)
    config_list = list(config_set)
    files_dict = {config:[] for config in config_list}
    for f in ntuple_files:
        _, _, _, configuration, _, _ = f[f.rfind('/')+1:].split('.')
        files_dict[configuration].append(f)

    return files_dict

# inner loop
def process_file(filename, config):
    ntup_file = uproot.open(filename) # i/o with uproot
    # check if file has "generate" histograms (adds one level to file)
    if any(['generate' in j for j in [k.decode() for k in ntup_file.keys()]]):
    # if any(['generate' in j for j in ntup_file.keys()]):
        nt_key = 'readvd/nttvd'
    else:
        nt_key = 'nttvd'
    # uproot to pandas dataframe
    df = ntup_file[nt_key].pandas.df(cols_keep)
    df['config'] = config
    # query for trapped particles
    df_tr = df.query('time > 0.1').copy()
    # grab by trapped particle evt numbers for t=0 dataframe
    df_tr0 = df[(df['evt'].isin(df_tr['evt'].values)) & (df['time'] < 0.1)].copy()
    # calculate |p| (not in ntuple)
    df_tr.eval('p = (px**2 + py**2 + pz**2)**(1/2)', inplace=True)
    df_tr0.eval('p = (px**2 + py**2 + pz**2)**(1/2)', inplace=True)

    return df_tr, df_tr0

# create "traps" dataframes for t=0 and t=700 ns
def generate_traps_dataframes(config, files):
    # loop through each file and return dataframes with traps and trap gen
    df_tuples = Parallel(n_jobs=NUM_CPU)(delayed(process_file)(f, config) for f in tqdm(files, file=sys.stdout, desc='file #'))
    df_list = [i[0] for i in df_tuples]
    df0_list = [i[1] for i in df_tuples]

    # concatenate, sort, and reindex final dataframes
    df_traps = pd.concat(df_list, ignore_index=True)
    df_traps0 = pd.concat(df0_list, ignore_index=True)
    df_traps.sort_values(by=['run', 'subrun', 'evt'], inplace=True)
    df_traps0.sort_values(by=['run', 'subrun', 'evt'], inplace=True)
    df_traps.reset_index(drop=True, inplace=True)
    df_traps0.reset_index(drop=True, inplace=True)

    return df_traps, df_traps0

def save_pkl(df, filename):
    df.to_pickle(filename)


if __name__ == '__main__':
    t0 = time.time()

    check_dirs() # make sure output directories exist
    files_dict = find_ntuple_files()

    # loop through configurations
    for config, file_list in files_dict.items():
        print(f'Processing configuration: {config}, with # files: {len(file_list)}')
        # generate dfs
        df, df_gen = generate_traps_dataframes(config, file_list)
        # save pickles
        save_pkl(df, outdir+f'pickles/mu_bottle/{config}.trap.df.pkl')
        save_pkl(df_gen, outdir+f'pickles/mu_bottle/{config}.trap-gen.df.pkl')
    delta_t_seconds = time.time() - t0
    print(f'Finished processing! Total time: {delta_t_seconds} seconds = {delta_t_seconds/60} minutes')
