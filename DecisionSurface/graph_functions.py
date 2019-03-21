import numpy as np
import itertools
import math
import sys
import os as os
#import matplotlib.pyplot as plt
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
#import seaborn as sns



def target(csweep, typestr):
    ## below specify target ##

    clen = len(csweep)
    cmin = np.min(csweep)
    cmax = np.max(csweep)
    cstep = csweep[1] - csweep[0]
    dx, dy = cstep, cstep

    yd, xd = np.mgrid[slice(cmin, cmax + dy, dy),
                    slice(cmin, cmax + dx, dx)]


    zd= np.zeros([len(xd), len(yd)], dtype= float)
    mid = int(np.floor(clen/2))


    if typestr == 'RM':
        xgrid= np.linspace(0,0.5, num=len(xd))

        for i in range(len(xd)):
            zd[:,i] = np.linspace(xgrid[i], 1.0, num = len(xd))
        zd[0:mid+1, 0:mid+1] = 0.001
    elif typestr == 'AD':
        xgrid= np.linspace(0,1.0, num=len(xd))

        for i in range(len(xd)):
            zd[:,i] = np.linspace(xgrid[i], 1.0, num = len(xd))
        zd[0:mid+1, 0:mid+1] = 0.001

    elif typestr == 'IM':
        xbegin= np.linspace(0,1.0, num=len(xd))
        xend = np.linspace(1.0,0.5, num = len(xd))

        for i in range(len(xd)):
            zd[:,i] = np.linspace(xbegin[i], xend[i], num = len(xd))
        zd[0:mid+1, 0:mid+1] = 0.001
        zd = zd*0.9

    elif typestr == 'BL':
        xbegin= np.linspace(0,0.5, num=len(xd))
        xend = np.linspace(0.5,1.0, num = len(xd))

        for i in range(len(xd)):
            zd[:,i] = np.linspace(xbegin[i], xend[i], num = len(xd))
        zd[0:mid+1, 0:mid+1] = 0.001
        zd = zd*0.9
        
    return zd

def create_targets(csweep, tartype):

    # Below create all targets: zall
  
    return [target(csweep, tartype[0]), target(csweep, tartype[1])]

def create_inputs(beta, csweep, c0):
        

    return [(1.0+ np.exp(-beta[0]*(csweep-c0[1])))**-1, (1.0+ np.exp(-beta[1]*(csweep-c0[1])))**-1 ]

def create_Einit(Nlinks, emin = -8.0, emax = 2.0):
    thinit = (emax - emin) * np.random.random_sample(Nlinks) + emin

    return thinit

def evaluate_loss(Pnow, Ptar):
    loss = 0.0
    for i in range(0, len(Pnow)):
        loss += np.sum(np.abs(Pnow[i]- Ptar[i]))
    return loss

def return_tgype(index):

    if index == 1:
        return ['RM', 'RM']
    elif index == 2:
        return ['AD', 'AD']
    elif index == 3:
        return ['BL', 'BL']
    elif index == 4:
        return ['IM', 'IM']
    elif index == 5:
        return ['RM', 'AD']
    elif index == 6:
        return ['RM', 'BL']
    elif index == 7:
        return ['RM', 'IM']
    elif index == 8:
        return ['AD', 'BL']
    elif index == 9:
        return ['AD', 'IM']
    elif index == 10:
        return ['BL', 'IM']
    

def conditional_prob(type_str, PAstate, Ematrix, Wbind = 2.0, J = -2.0):

    num = 0.0
    den = 0.0

    if type_str == 'add':
        for j in range(len(Ematrix)):
            num += np.exp(-Ematrix[j])*PAstate[j]
            den += (np.exp(-Ematrix[j])*PAstate[j] + np.exp(-Wbind)*(1-PAstate[j]))
        
        return num/ (1.0+ den)

    elif type_str == 'AND':
        cooperation = PAstate[0]*(PAstate[1])* np.exp(-Ematrix[0]-Ematrix[1]-J)
        cooperation_un = PAstate[0]*(1-PAstate[1])*np.exp(-Ematrix[0]-Wbind)+ PAstate[1]*(1-PAstate[0])*np.exp(-Ematrix[1]-Wbind)
        activator = PAstate[0]* np.exp(-Ematrix[0])+  PAstate[1]* np.exp(-Ematrix[1])

        return cooperation / (1.0 + cooperation + cooperation_un + activator)

    elif type_str == 'rep':
        cooperation = PAstate[0]*(1- PAstate[1])* np.exp(-Ematrix[0]-Ematrix[1]-J)
        undesirablecoop = PAstate[0]*PAstate[1]* np.exp(-Wbind)
        activator = PAstate[0]*np.exp(-Ematrix[0])
        repressor_active = PAstate[1]*np.exp(-Wbind)
        repressor_inactive = (1-PAstate[1])*np.exp(-Ematrix[1])
        den = 1.0 + cooperation + undesirablecoop + activator + repressor_active + repressor_inactive

        return cooperation / den 


def create_name(ckttype, tgtype, ns):
    
	
    if len(tgtype) >= 1:
        str_r = ckttype + '_' + tgtype[0] + '_'+ tgtype[1] + '_ns' + str(ns)
    else:
        str_r = ckttype + '_' + tgtype + '_ns' + str(ns)
    return str_r


def plotDS_2d_multi(target, p_leanred, tgtype, ckt, ns):

    fs = 6 # font size
    fl = 10
    ft = 12
    ms = 6 # marker size
    lw = 5 # line width size

    

    fig, ax = plt.subplots(len(target), 3, figsize= (10, 5))
    # set up the ticks for heatmap
    # Start plotting 

    ## TARGET 1
    cmap = sns.color_palette("PuBu", n_colors = 200)
    sns.heatmap(target[0], cmap = cmap, square = True, ax = ax[0,0], vmin= 0.0)
    ax[0,0].invert_yaxis()
    ax[0,0].set_xticks((0, 6, 11))
    ax[0,0].set_yticks((0, 6, 11))
    ax[0,0].set_xticklabels(('0', '0.5', '1.0'), rotation = 0)
    ax[0,0].set_yticklabels(('0', '0.5', '1.0'), rotation = 0)
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize= fl)

    #ax[0,0].set_xlabel('probability of input 1', fontsize = fl)
    ax[0,0].set_ylabel('probability of input 2', fontsize = fl)
    ax[0,0].set_title('Target 1:' + tgtype[0], fontsize = ft, fontweight='bold' )
    #plt.show()


    ## TARGET 2
    sns.heatmap(target[1], cmap = cmap, square = True, ax = ax[1,0], vmin= 0.0)
    ax[1,0].invert_yaxis()
    ax[1,0].set_xticks((0, 6, 11))
    ax[1,0].set_yticks((0, 6, 11))
    ax[1,0].set_xticklabels(('0', '0.5', '1.0'), rotation = 0)
    ax[1,0].set_yticklabels(('0', '0.5', '1.0'), rotation = 0)
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize= fl)


    ax[1,0].set_xlabel('probability of input 1', fontsize = fl)
    ax[1,0].set_ylabel('probability of input 2', fontsize = fl)
    ax[1,0].set_title('Target 2:' + tgtype[1], fontsize = ft, fontweight='bold' )
    #plt.show()

    
    ### LEARNED 1
    cmap = sns.color_palette("BuGn", n_colors = 200)
    sns.heatmap(p_leanred[0], cmap = cmap, square = True, ax =ax[0,1], vmin = 0.0, vmax = 0.9)
    ax[0,1].invert_yaxis()
    ax[0,1].set_xticks((0, 6, 11))
    ax[0,1].set_yticks((0, 6, 11))
    ax[0,1].set_xticklabels(('0', '0.5', '1.0'), rotation = 0)
    ax[0,1].set_yticklabels(('0', '0.5', '1.0'), rotation = 0)
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize= fl)

    #ax[0,1].set_xlabel('probability of input 1', fontsize = fl)
    #ax[0,1].set_ylabel('probability of input 2', fontsize = fl)
    ax[0,1].set_title('Learned 1:' + tgtype[0], fontsize = ft, fontweight='bold' )
    #plt.show()


    
    cmap = sns.color_palette("BuGn", n_colors = 200)
    sns.heatmap(p_leanred[1], cmap = cmap, square = True, ax = ax[1,1], vmin = 0.0, vmax = 0.9)
    ax[1,1].invert_yaxis()
    ax[1,1].set_xticks((0, 6, 11))
    ax[1,1].set_yticks((0, 6, 11))
    ax[1,1].set_xticklabels(('0', '0.5', '1.0'), rotation = 0)
    ax[1,1].set_yticklabels(('0', '0.5', '1.0'), rotation = 0)
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize= fl)

    ax[1,1].set_xlabel('probability of input 1', fontsize = fl)
    #ax[1,1].set_ylabel('probability of input 2', fontsize = fl)
    ax[1,1].set_title('Learned 2:' + tgtype[1], fontsize = ft, fontweight='bold' )
    #plt.show()


    
    zderr = np.absolute(p_leanred[0]-target[0])
    cmap = sns.light_palette("red", n_colors = 50, reverse=False)
    sns.heatmap(zderr, cmap = cmap, square = True, vmin = 0, vmax = 0.6, ax = ax[0,2])
    ax[0,2].invert_yaxis()
    ax[0,2].set_xticks((0, 6, 11))
    ax[0,2].set_yticks((0, 6, 11))
    ax[0,2].set_xticklabels(('0', '0.5', '1.0'), rotation = 0)
    ax[0,2].set_yticklabels(('0', '0.5', '1.0'), rotation = 0)
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize= fl)

    #ax[0,2].set_xlabel('probability of input 1', fontsize = fl)
    #ax[0,2].set_ylabel('probability of input 2', fontsize = fl)
    ax[0,2].set_title('Error 1:' + tgtype[0], fontsize = ft, fontweight='bold' )

    #plt.show()
    
    
    

    zderr2 = np.absolute(p_leanred[1]-target[1])
    cmap = sns.light_palette("red", n_colors = 50, reverse=False)
    sns.heatmap(zderr2, cmap = cmap, square = True, vmin = 0, vmax = 0.6, ax = ax[1,2])
    ax[1,2].invert_yaxis()
    ax[1,2].set_xticks((0, 6, 11))
    ax[1,2].set_yticks((0, 6, 11))
    ax[1,2].set_xticklabels(('0', '0.5', '1.0'), rotation = 0)
    ax[1,2].set_yticklabels(('0', '0.5', '1.0'), rotation = 0)
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize= fl)

    ax[1,2].set_xlabel('probability of input 1', fontsize = fl)
    #ax[1,2].set_ylabel('probability of input 2', fontsize = fl)
    ax[1,2].set_title('Error 2:' + tgtype[1], fontsize = ft, fontweight='bold' )
    


    fig.tight_layout()
    fig_name = 'DS' + ckt + '_' + tgtype[0] + '_' + tgtype[1] + '_T'+ str(ns)
    #fig.savefig(fig_name+'.eps', bbox_inches='tight')
    #fig.savefig(fig_name+'.eps')
    #fig.savefig(fig_name+'.pdf', bbox_inches='tight')
    fig.savefig(fig_name+'.pdf')
    fig.savefig(fig_name+'.png')
    plt.close(fig)



def plotDS_err_evol(loss, tgtype, ckt):

    fs = 18 # font size
    fs_s = 16 # small text font size
    fl = 20
    ft = 18
    ms = 6 # marker size
    lw = 5 # line width size

    

    fig, ax = plt.subplots()
    title_str = tgtype[0] + '_'  + tgtype[1]
    t = np.linspace(1, len(loss), num = len(loss))
    ax.plot(t, loss, linewidth=1.5)
    ax.set_xlabel(r'annealing time, $t$', multialignment='left', fontweight='bold', fontsize=fs)
    ax.set_ylabel(r'Loss, $\mathcal{E}$', multialignment='left', fontweight='bold', fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.set_title(title_str, fontweight='bold', fontsize=fs)
    fig.tight_layout()
    fig_name = 'Loss' + ckt + '_tg_' + tgtype[0] + '_' + tgtype[1]
    #fig.savefig(fig_name+'.eps')
    fig.savefig(fig_name+'.pdf', bbox_inches='tight')
    fig.savefig(fig_name+'.png')
    plt.close(fig)


def plot_optparam(estar, tgtype, ckt, ns):

    elabel = range(1, len(estar)+1)
    elabels = map(str, elabel)

    plt.rcdefaults()
    fig, ax = plt.subplots()
    #fig.subplots_adjust(hspace=.35)
    fs = 12 # font size
    fs_s = 10 # small text font size
    ftick = 12 # tick font size
    titlesize = 12
    ms = 6 # marker size
    lw = 5 # line width size

    title_string = 'DS' + ckt 

    y_pos = np.arange(len(elabels))
    ax.barh(y_pos, estar, color='b', align='center', ecolor='black')

    ax.set_yticks(y_pos)
    ax.yaxis.set_tick_params(labelsize=ftick)
    ax.set_yticklabels(elabels,fontdict={'fontsize': ftick})
    ax.xaxis.set_tick_params(labelsize=ftick)
    ax.set_xlabel(r'$\theta_{ji}/(k_BT)$', fontsize=ftick)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_title(title_string, fontdict={'fontsize': titlesize})


    if len(tgtype)>=1:
        fig_str= 'paraDS'+ ckt +  '_'+ tgtype[0] + '_' + tgtype[1] + 'ns_' + str(ns)
    else:
        fig_str= 'paraDS'+ ckt +  '_'+ tgtype[0] + str(ns)

    
    #fig.savefig(fig_str+'.eps', bbox_inches='tight')
    fig.savefig(fig_str+'.pdf', bbox_inches='tight')
    fig.savefig(fig_str+'.png')
    plt.close(fig)

