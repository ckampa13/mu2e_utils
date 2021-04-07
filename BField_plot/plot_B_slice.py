import numpy as np
import pandas as pd
from plotly.offline import plot
import plotly.graph_objs as go

pdir='data/'
pklfile_PS = pdir+'PSMap.pkl'
pklfile_DS = pdir+'DSMap.pkl'

def load_df(pklfile):
    return pd.read_pickle(pklfile)

def make_plot(solname, df, X, Y, Z, query_str, save_name_base):
    title = f'{Z} vs. {X} and {Y} for {solname}<br>{query_str}'
    df_ = df.query(query_str)
    piv = df_.pivot(X, Y, Z)
    x = piv.index.values
    y = piv.columns.values
    z = piv.values.T
    XX, YY = np.meshgrid(x,y)
    surface = go.Surface(
        x=XX, y=YY, z=z,
        colorbar=go.surface.ColorBar(title='Gauss',
                                     titleside='right',
                                     titlefont=dict(size=25),
                                     tickfont=dict(size=18),
                                     ),
        colorscale='Viridis')
    data = [surface]
    layout = go.Layout(
        title=title,
        titlefont=dict(size=30),
        autosize=False,
        width=900,
        height=750,
        scene=dict(
            xaxis=dict(title=f'{X} (m)', titlefont=dict(size=25), tickfont=dict(size=16)),
            yaxis=dict(title=f'{Y} (m)', titlefont=dict(size=25), tickfont=dict(size=16)),
            zaxis=dict(title=f'{Z} (Gauss)', titlefont=dict(size=25), tickfont=dict(size=16)),
            aspectratio=dict(x=1, y=2, z=1),
            aspectmode='manual',
            camera=dict(
                center=dict(x=0,y=0,z=-0.3),
                eye=dict(x=3.45/1.6, y=2.49/1.6, z=1.59/1.6),
            ),
        ),
    )

    fig = go.Figure(data=data, layout=layout)
    fig.write_html(save_name_base+'.html')
    fig.write_image(save_name_base+'.pdf', engine='kaleido')
    fig.write_image(save_name_base+'.png', engine='kaleido')
    # plot(fig)


if __name__=='__main__':
    # PS plots
    yslice = 0.
    xlim = [-0.8, 0.8]
    zlim = [-7.9, 3.4]
    df = load_df(pklfile_PS)
    query_str = f'(Y == {yslice}) & ({xlim[0]} <= X <= {xlim[1]}) & ({zlim[0]} <= Z <= {zlim[1]})'
    make_plot('PS', df, 'X', 'Z', 'Bz', query_str, 'plots/PS_Bz_vs_XZ')

    # DS plots
    yslice = 0.
    xlim = [-1., 1.]
    zlim = [3.3, 14.]
    df = load_df(pklfile_DS)
    query_str = f'(Y == {yslice}) & ({xlim[0]} <= X <= {xlim[1]}) & ({zlim[0]} <= Z <= {zlim[1]})'
    make_plot('DS', df, 'X', 'Z', 'Bz', query_str, 'plots/DS_Bz_vs_XZ')
