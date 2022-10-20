
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.size"  : 22,
    "font.sans-serif": ["Helvetica"]})

def BarPlot( A , yerr=None, colors = ['r','b','g','y','r','b','g','y','r','b','g','y','r','b','g','y'], ax=None ):
    A = A.T
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot( 111, projection='3d' )
    
    lenx, leny = A.shape
    z = list(A)
    xs = np.arange(lenx)
    ys = np.arange(leny)
    
    for n in range(leny):
        zs = z[n]
        cs = colors[n]
        ax.bar( xs, A[n], n, zdir='y', color=cs, alpha=0.5)
        
        if yerr is not None:
            for i in xs:
                ax.plot( [i,i], [n, n], [A[n,i]+yerr[n,i], A[n,i]-yerr[n,i]], marker="_", color=cs)
        
    return ax
    
def Plot_Chois(  choi, error_choi=None, axes=None, z_lim = 0.1, shape = None, text=True  ) :
    
    N = len( choi )
    if shape is None:
        xx = 1
        yy = N
    else:
        xx = shape[0]
        yy = shape[1]
    
    if axes is None:
        fig = plt.figure(figsize=[yy*5,xx*5])
        axes = []
        for j in range(N):   
            axes.append( fig.add_subplot( xx, yy, j+1, projection='3d') )
            
    for j in range(N):
        ax = axes[j]
        if error_choi is None:
            ax = BarPlot( abs(choi[j]), None, ax=ax )
        else:
            ax = BarPlot( abs(choi[j]), error_choi[j], ax=ax )
        if text is True:
            ax.text( j*N+j, j*N+j, z_lim*1.1,  r'${}$'.format(np.round(abs(choi[j])[ j*N+j, j*N+j ],3))  )
        ax.set_zlim([0,z_lim])
    
    return axes


def set_ticks( axes, x_pos, x_ticks, y_pos, y_ticks, label_pos , Labels):

    j = 0
    for ax in axes:
        ax.set_xticks( x_pos )
        ax.set_xticklabels( x_ticks )
        ax.set_yticks( y_pos )
        ax.set_yticklabels( y_ticks )
        ax.text( label_pos[0], label_pos[1], label_pos[2], Labels[j], size = 28 )

        plt.setp( ax.xaxis.get_majorticklabels(), rotation=45, ha="right" , rotation_mode="anchor")
        plt.setp( ax.yaxis.get_majorticklabels(), rotation=-45, ha="left" , rotation_mode="anchor")
        ax.tick_params(axis='x', pad=-3)
        ax.tick_params(axis='y', pad=-3)
        ax.tick_params(axis='z', rotation=-15, pad=10)
        j += 1

    return axes