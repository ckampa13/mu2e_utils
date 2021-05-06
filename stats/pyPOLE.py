import numpy as np

def readin(filename='data/pole1.0.ini', verbose=True):
    # read in initialization file. remove newline character and ignore comments (line starts with "!")
    with open(filename, 'r') as f:
        lines = [i.rstrip('\n').rstrip(' ') for i in f.readlines() if not i[0] in ['!', '#', '*', 'C']]#, 'c', ]]
    # loop through lines to set proper variables
    for line in lines:
        #print(line)
        ident, val = line.split(' ')
        if ident=='of':
            filehb = val
        elif ident=='ft':
            filltrue = int(val)
        elif ident=='ff':
            fluxfactor = float(val)
        elif ident=='ll':
            conflevel = float(val)
        elif ident=='fi':
            filetrue = int(val)
        elif ident=='fn':
            filein = val
        elif ident=='fo':
            fileout = val
        elif ident=='rf':
            fmax = float(val)
        elif ident=='sw':
            width = float(val)
        elif ident=='no':
            normtrue = int(val)
        elif ident=='fc':
            FCtrue = int(val)
        elif ident=='bg':
            background = float(val)
        elif ident=='nm':
            nobs = int(val)
        elif ident=='cc':
            ctrue = int(val)
        elif ident=='bu':
            sbg = float(val)
        elif ident=='pb':
            bkpar = int(val)
        elif ident=='eu':
            sac = float(val)
        elif ident=='pu':
            spar = int(val)
        elif ident=='eb':
            bsac = float(val)
        elif ident=='pe':
            bar = int(val)
    # calculate number of steps
    step_help = fmax/width
    steps = int(step_help)
    if verbose:
        # Tell User which parameters: 
        print('Performing Conf Belt Const :')
        print(f'Output hbfile:     {filehb}')  
        print(f'fill diagnostic histos {filltrue}')
        print(f'Confidence Level {conflevel:0.5f}')
        print(f'Condition (1=yes) {normtrue}')
        print(f'stepwidth : {width:0.3f}')
        print(f'Feldman Cousins    {FCtrue}')
        print(f'fluxfactor:        {fluxfactor:0.3f}')
        print(f'Read from file: {filein}')
        print(f'Write to file: {fileout}')
        print(f'Exp. BG events     {background:0.3f}')
        print(f'Measured events    {nobs}')
        print(' ')
        print('Used Paramterisation :')
        print('Gaussian = 1')
        print('flat = 2')
        print('log-normal = 3')
        print(' ')
        print(f'rel Aeff unc (sig): {sac:0.3f}')
        print(f'Parametrization: {spar}')
        print(f'rel Aeff unc (bck): {bsac:0.3f}')
        print(f'Parametrization: {bar}')
        print(f'rel bg unc:        {sbg:0.3f}')
        print(f'Parametrization: {bkpar}')
        print(f'max flux:          {fmax:0.3f}')
        print(f'Number of steps: {steps}')
        print('CAUTION: not bigger than 1000!')
    #return lines
    return filehb, filltrue, fluxfactor, conflevel, filetrue, filein, fileout, fmax, width,\
           normtrue, FCtrue, background, nobs, ctrue, sbg, bkpar, sac, spar, bsac, bar, steps

def read_grid(filename='data/test.in'):
    '''
    x = background
    y = number of observed events
    '''
    x = []; y = []
    # read in input file. remove newline character
    with open(filename, 'r') as f:
        lines = [i.rstrip('\n').rstrip(' ') for i in f.readlines()]
    # loop through lines to grab x, y values set proper variables
    for line in lines:
        y_, x_ = line.split(' ')
        x.append(float(x_))
        y.append(float(y_))
    x = np.array(x)
    y = np.array(y)
    ncalc = len(x)
    return x, y, ncalc
    
def fluxfactors(sac,bsac,sbg,ctrue,N_exp):
    sigfactor = np.zeros((3,N_exp)) # signal eff. unc.
    befactor = np.zeros((3,N_exp)) # bg eff. unc.
    bkfactor = np.zeros((3,N_exp)) # expected bg unc.
    # Gaussian distribution; mu=0, std=1
    a,b,r = np.random.normal(loc=0, scale=1, size=(3, N_exp))

    # Uniform distribution 
    #call ranlux(ranvec,3)
    ranvec = np.random.uniform(low=0, high=1, size=(3, N_exp))

    # Conrad
    sigfactor[0] = 1+sac*r
    sigfactor[1] = (1-sac)+2*ranvec[0]*sac
    sigfactor[2] = np.exp(sac*r - sac**2/2)

    # background efficiency factors
    if (ctrue == 1): # correlated
        befactor[0] =  1+bsac*r
        befactor[1] =  (1-bsac)+2*ranvec[0]*bsac
        # mean of logN will be 1
        befactor[2] = np.exp(bsac*r - bsac**2/2)
    else: # no correlation
        befactor[0] =  1+bsac*a
        befactor[1] =  (1-bsac)+2*ranvec[1]*bsac
        # mean of logN will be 1
        befactor[2] = np.exp(bsac*a - bsac**2/2)

    # background prediction factors.
    bkfactor[0] = (1+sbg*b)
    bkfactor[1] = (1-sbg)+2*ranvec[2]*sbg     
    bkfactor[2] = np.exp(sbg*b - sbg*bsac/2)
    
    return sigfactor, befactor, bkfactor

def fluxlim(trisigflu, sac, sbg, k, normtrue, used, fluxfactor,
            background, nobs, bsac, width, filltrue, ctrue, spar, bar, bkpar, conflevel,
            N_exp):
    # Take flux from our MC
    bckfluxnom = background/fluxfactor
    # Perform Pseudo Experiments to calculate Integrals
    sigfactor, befactor, bkfactor = fluxfactors(sac, bsac, sbg, ctrue, N_exp)
    # uncertainty in background flux    
    bckflux = bckfluxnom*bkfactor[bkpar-1]
    # Diagnostics histograms
    if (filltrue == 1): 
        #call hfill(k+2000,bkfactor(bkpar),1.,1.)
        # FIXME!
        pass
    # include flux uncertainty
    musignal = trisigflu*fluxfactor*sigfactor[spar-1]
    mubck  = bckflux*befactor[bar-1]
    # Truncation for Gaussian uncertainty
    mask_trunc = (musignal < 0.) | (mubck < 0.)
    musignal[mask_trunc] = 0.
    mubck[mask_trunc] = 0.
    # generate pseudo experiment results
    nsig = np.random.poisson(lam=musignal, size=N_exp)
    nbck = np.random.poisson(lam=mubck, size=N_exp)
    n_tot = nsig+nbck # total number observed
    fntot = n_tot.astype(np.float)
    # Diagnostics
    if (filltrue==1):
        # call hfill(k+100,fntot(nc),1.,1.)
        pass
    # truncate
    #mask_trunc = (musignal < 0.) | (mubck < 0.)
    fntot = fntot[~mask_trunc]
    # normalization / conditioning
    if normtrue != 0:
        mask_norm = (nbck < nobs)
        fntot = fntot[~mask_norm]
    # normalizations, FC
    # sort in ascending order
    fntot = np.sort(fntot)
    # limiting index for Neyman UL
    jlim = round((1.-conflevel)*len(fntot)) - 1 # integer. if you use int() it rounds down!
    intjlim = jlim # unnecessary--using round
    nlim= fntot[intjlim]

    # make histogram "N Hist" (new for python version)
    dist, _ = np.histogram(fntot, bins=np.arange(0,101,1)) # maybe 0, 102 so 100 is in different bin than 99
    noent = np.sum(dist) # number of entries in histogram

    # FIXME
    # needed for FC?
    #####call hfill(40,nlim,trisigflu,1.)

    # default value for checking later
    resflux = -1000 # can never be encountered for counting experiment
    # calculate Neyman upper limit for the passed in n0
    if (nlim == nobs+1):
        if (used == 0):
            resflux = trisigflu-width
            print(f'\nNeyman Upper Limit: {resflux}')
        used = 1

    return dist, noent, used, resflux, nlim

def FC(matrix,fluxarray,nobs,nent,steps,filltrue, conflevel):
    P_best = np.zeros(100)
    R = np.zeros((steps, 100))
    n_limit = np.zeros((2, steps))
    
    Philow=0
    Phihigh=0

    # for each n find mu_best                                           
    for j in range(100):
        mtemp = matrix[:,j] # cleaner
        mtemp = np.sort(mtemp) # sort in ascending order
        # CHECK IF TRUE
        P_best[j] = mtemp[-1] # best is mtemp with highest
        if (filltrue == 1):
            # FIXME! diagnostics
            #call hfill(31,P_best(j),fluxarray(i),1.)
            pass

    # for each flux calculate likelihood ratio for each n                
    for i in range(steps):
        for j in range(100):
            if (P_best[j] != 0) and (matrix[i,j] != 0):
                R[i,j] = matrix[i,j] / P_best[j]
            else:
                R[i,j] = 0

        # find i with highest R
        Rtemp = R[i,:] # cleaner
        index = np.argsort(Rtemp)[::-1]

        # add P(for that i)
        # until sum(P) = conflevel*100 %
        #j = 0
        adder = 0. # real
        dum = conflevel*nent[i] # real
        for j_ in range(100):
            j=j_ # CHECK
            if (adder >= dum):
                break
            adder += matrix[i, index[j_]]

        index_sorted = np.sort(index[:j]) # ascending sort
        n_limit[0,i] = index_sorted[0] # CHECK
        n_limit[1,i] = index_sorted[-1] + 1 # CHECK

        # find flux which has nobs as upper limit (Philow)
        # find flux which has nobs as lower limit (Phihigh) (shift due to indexing)             
        if (n_limit[0,i] == nobs):
            Phihigh = fluxarray[i]
        if (n_limit[1,i] == nobs):
            Philow = fluxarray[i] # I think this would give wrong result...not positive

        # just to be able to have a look at the construction 
        nlim1 = float(n_limit[0,i])
        nlim2 = float(n_limit[1,i])
        hbflux = fluxarray[i]
        # FIXME! Filling histogram ID=50
        #call hfill(50,nlim1,hbflux,1.)
        #call hfill(50,nlim2,hbflux,1.)
      
    print('\nexiting flux loop')
    print(f' FC upper limit: {Phihigh:0.5f}')
    print(f' FC lower limit: {Philow:0.5f}\n\n')

    return Philow, Phihigh, n_limit

def run_POLE(ini_file='data/pole1.0.ini', N_exp=100000):
    # read in steering file and print program configuration
    filehb, filltrue, fluxfactor, conflevel, filetrue, filein, fileout, fmax, width,\
        normtrue, FCtrue, background, nobs, ctrue, sbg, bkpar, sac, spar, bsac, bar, steps\
        = readin(filename=ini_file)
    if filetrue:
        x, y, ncalc = read_grid(filename='data/'+filein)
        print('-\n-\nMode: Read input from file\n-')
    else:
        x = np.array([background]) # check
        y = np.array([nobs]) # check
        ncalc = 1
        print('-\n-\nMode: single construction\n-')
    print(f'expected background (x): {x}')
    print(f'number of observed (y): {y}')
    
    # open output file
    fout = open(fileout, "w")
    # Loop over input nobs/BG pairs
    for p in range(ncalc):
        fluxarray = np.zeros(steps)
        matrix = np.zeros((steps, 100))
        nent = np.zeros(steps)
        Philow=0.
        Phihigh=0.
        # FIXME! Reset histogram IDs: 50, 40, 100
        nobs_ = y[p]
        background_ = x[p]
        # message without "RanLux" Statement
        print(f'-\nPerforming Construction for n0/bg: {nobs_}/{background_}\n-\n-\n')
        trisigflux= 0.0
        # Scan through flux space and perform Construction
        used = 0
        # FIXME! Reset histogram ID: 20
        # Loop through trisigflux to try
        mus = [] # TEST
        for i in range(steps):
            mus.append(trisigflux) # TEST
            # progress tracker
            # FIXME! Update to tqdm progress bar
            if (i % int(steps/20)) == 0:
                print('.')
            
            # INCREMENTS TWICE EACH LOOP--BAD!!!
            #trisigflux =trisigflux+width # this starts us above 0?
            #fluxarray[i] = trisigflux
            
            # Diagnostics histogramms
            if (filltrue == 1):
                # FIXME! diagnostics
                #nh = 100 + i
                #call hbook1(nh,'N Dist',100,0.,100.,0.)
                pass
            
            # call fluxlim
            dist, noent, used, resflux, nlim = fluxlim(trisigflux, sac, sbg, i, normtrue, used,
                                                       fluxfactor, background_, nobs_, bsac, width,
                                                       filltrue, ctrue, spar, bar, bkpar, conflevel,
                                                       N_exp)
            # fill matrix for Feldman Cousins
            #for l in range(100):
            #    matrix[i,l] = dist[l]
            matrix[i, :] = dist # cleaner
            nent[i] = noent
            # Diagnostics histogramms
            if (filltrue == 1):
                # FIXME! diagnostics
                # nh = 1100 + i
                #call hbook2(nh,'Rank Dist.',100,0.,50.,10,0.,1.,1.)
                pass
            fluxarray[i] = trisigflux
            trisigflux += width
        
        mus = np.array(mus) # TEST
        
        # Perform Likelihood Ratio construction 
        if (FCtrue == 1):
            Philow, Phihigh, n_limit = FC(matrix,fluxarray,nobs_,nent,steps,filltrue, conflevel)

        # Write output to file !
        fout.write(f'{nobs_}, {background_}, {Philow}, {Phihigh}\n')
    # outside nobs/background loop
    fout.close()
    
    print('Calculation complete!')
    
    return mus, Philow, Phihigh, n_limit, matrix, fluxarray, nent, steps

if __name__=='__main__':
    temp = run_POLE()
    mus, Philow, Phihigh, n_limit, matrix, fluxarray, nent, steps = temp
