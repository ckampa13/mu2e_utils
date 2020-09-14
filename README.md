# mu2e_utils
Some utility scripts for odds and ends in Mu2e.

## Install
- Use 'mu2e_utils/environment.yml' to create conda environment named 'mu2e_utils'
```bash
$ conda env create -f environment.yml
```

## Linear Gradient
- 'mu2e_utils/linear_gradient/'
- Fits linear gradient model that satisfies Maxwell's equations to Mu2e DS Tracker region magnetic field.

### Running this calculation
- Set user directory configurations in 'mu2e_utils/linear_gradient/user_conigs.py'
- From base directory ('mu2e_utils')
```bash
$ cd linear_gradient
$ conda activate mu2e_utils
$ python run.py
```
- Fit report will print to screen. The results and plots will be stored in 'outdir' set in user configurations
