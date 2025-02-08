from lumedit.params import root, img_dir
from lumedit.utils import load_image, cmpt_perceived_lightness, compute_luminance, adjust_lightness
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
#import glob

############################################
# This script is used to 
# - 1) calculate the average perceived lightness of a set of images -> target_Lstar, 
# - 2) adjust the perceived lightness of the images to this target value, 
# - 3) plot the distribution of the original and adjusted L* values.

if __name__ == '__main__':

    os.makedirs(os.path.join(root, 'step2'), exist_ok=True)
    img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')])
    #img_paths = sorted(glob.glob('/Users/pironn/Documents/PhD/experiment/crossmodal-sequences/data/input/stims/*/*.png'))
    # Calculate the average perceived lightness of the images
    Ls = []
    for path in img_paths:
        img = load_image(path)
        Lstar = cmpt_perceived_lightness(img)
        Ls.append(Lstar)
    target_Lstar = np.mean(Ls)
    print(f'Target L* value: {target_Lstar:.2f}')

    # Adjust the perceived lightness of the images to the target value
    Ls_adj = []
    for path in img_paths:
        img = load_image(path)
        img_adj = adjust_lightness(img, target_Lstar, tol=0.01, max_iter=1000 )
        r_lum, g_lum, b_lum = compute_luminance(img)
        Y = np.mean(r_lum + g_lum + b_lum)
        Lstar_adj = cmpt_perceived_lightness(img_adj)
        Ls_adj.append(Lstar_adj)
        print(f'Adjusted L* value: {Lstar_adj:.2f}')
        print(f'Average luminance: {Y:.2f}')  
        img_adj.save(os.path.join(root, 'step2', os.path.basename(path)))

    # Plot the distribution of L* values
    sns.set_context('talk')
    df = pd.DataFrame({'Original L*': Ls, 'Adjusted L*': Ls_adj})
    df_long = df.melt(var_name='Distribution', value_name='Lstar')

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    sns.histplot(Ls, bins=50, label='Original L*', color='blue', alpha=0.5, element='step', kde=True,  ax=ax)
    sns.histplot(Ls_adj, bins=50 , label='Adjusted L*', color='green', alpha=0.5, element='step', kde=True, ax=ax)
    ax.axvline(target_Lstar, color='red', linestyle='--', label='Target L*')
    ax.set_title('Distribution of L* values')
    ax.set_xlabel('Perceived lightness (L*)')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.tight_layout()
    sns.despine()
    viz_dir = os.path.join(root, 'step2', 'plots')
    os.makedirs(viz_dir, exist_ok=True)
    fig.savefig(os.path.join(viz_dir, 'Lstar_distribution.png'), dpi=300)

    print('Done!')