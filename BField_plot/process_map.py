import numpy as np
import pandas as pd

# globals to grab the right Bfield file
Bdir = '/cvmfs/mu2e.opensciencegrid.org/DataFiles/BFieldMaps/Mau13/'
pdir = 'data/'
Bfile_PS = Bdir + 'PSMap.txt'
pklfile_PS = pdir+'PSMap.pkl'
Bfile_DS = Bdir + 'DSMap.txt'
pklfile_DS = pdir+'DSMap.pkl'

def process_Bmap(Bfile, mm, gauss, xoffset, pklfile):
    '''
    Note: xoffset must be in mm, i.e. before scaling to m
    '''
    print(f'Processing File: {Bfile}')
    # read file
    header_names = ['X', 'Y', 'Z', 'Bx', 'By', 'Bz']
    df = pd.read_csv(Bfile, header=None, names=header_names, delim_whitespace=True, skiprows=4)
    print('Raw Dataframe:')
    print(df.head())

    # modifications
    # xoffset
    df.eval(f'X = X - {xoffset}', inplace=True)
    # Tesla to Gauss
    if gauss:
        df.eval('Bx = Bx * 10000.', inplace=True)
        df.eval('By = By * 10000.', inplace=True)
        df.eval('Bz = Bz * 10000.', inplace=True)
    # mm to m
    if not mm:
        df.eval('X = X / 1000.', inplace=True)
        df.eval('Y = Y / 1000.', inplace=True)
        df.eval('Z = Z / 1000.', inplace=True)
    # generate cylindrical coordinates
    df.eval('R = sqrt(X**2 + Y**2)', inplace=True)
    df.eval('Phi = arctan2(Y,X)', inplace=True)
    df.eval('Br = Bx*cos(Phi) + By*sin(Phi)', inplace=True)
    df.eval('Bphi = -Bx*sin(Phi) + By*cos(Phi)', inplace=True)

    print('Processed Dataframe:')
    print(df.head())

    print('Grid Values:')
    print(f'X: {df.X.unique()}')
    print(f'Y: {df.Y.unique()}')
    print(f'Z: {df.Z.unique()}')

    # save to pickle
    print(f'Saving to: {pklfile}')
    df.to_pickle(pklfile)

if __name__=='__main__':
    # PS
    process_Bmap(Bfile_PS, mm=False, gauss=True, xoffset=3904., pklfile=pklfile_PS)
    # DS
    process_Bmap(Bfile_DS, mm=False, gauss=True, xoffset=-3896., pklfile=pklfile_DS)
