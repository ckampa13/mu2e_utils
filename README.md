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

## Magnetic Bottle Visualization
- 'mu2e_utils/bottle_viz/'
- Visualize potential bottle regions in DS based on changing Br sign assumption.

### Making Plots
- Set user directory configurations in 'mu2e_utils/bottle_viz/user_conigs.py'
- From base directory ('mu2e_utils')
```bash
$ cd bottle_viz
$ conda activate mu2e_utils
$ python run.py
```
- Plots saved in 'outdir' set in user configurations. Need to change field map filename in user configurations to switch between field maps (e.g. Mau13 and shifted coil map).

## Muon Magnetic Bottle Analysis
- 'mu2e_utils/mu_bottle_analysis/'
- Analyze 'Time Virtual Detectors ntuple' (nttvd) ROOT files where t=0 ns and t=700 ns (if particle survives this long) are saved. Particles that survive until t=700 ns are called "trapped" and indicate magnetic bottle locations.

## Magnetic Field Plotting
- 'mu2e_utils/BField_plot/'
- Process magnetic field map file from CVMFS and make 3D surface plots in Plotly.
    1. Process map: can change passed in function parameters to change units
    ```bash
    $ cd BField_plot
    $ conda activate mu2e_utils
    $ python process_map.py
    ```
    which will save pickled `pandas.DataFrame` in `data` directory (not synced to git)
    2. Make plots: CAUTION -- value of "sliced" variable in query must be exact, and can be copied from the printed output of 'process_map.py'.
    ```bash
    ...
    $ python plot_B_slice.py
    ```
    which will output in the `plots` directory (not synced to git)
