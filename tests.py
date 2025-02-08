from lumedit.utils import norm_rgb2img, normalize_rgb, scale_rgb, rgb2linear, linear2rgb, lum2Lstar, Lstar2lum, extract_rgb, load_image
from lumedit.params import root, img_dir
import os
import numpy as np

def forward_pipeline(r, g, b):
    r_norm, g_norm, b_norm = normalize_rgb(r, g, b)
    r_lin = rgb2linear(r_norm)
    g_lin = rgb2linear(g_norm)
    b_lin = rgb2linear(b_norm)
    r_Lstar = lum2Lstar(r_lin * 0.2126)
    g_Lstar = lum2Lstar(g_lin * 0.7152)
    b_Lstar = lum2Lstar(b_lin * 0.0722)
    return r_Lstar, g_Lstar, b_Lstar

def inverse_pipeline(r_Lstar, g_Lstar, b_Lstar):
    r_lin = Lstar2lum(r_Lstar) / 0.2126
    g_lin = Lstar2lum(g_Lstar) / 0.7152
    b_lin = Lstar2lum(b_Lstar) / 0.0722
    r_norm = linear2rgb(r_lin)
    g_norm = linear2rgb(g_lin)
    b_norm = linear2rgb(b_lin)
    img = norm_rgb2img(r_norm, g_norm, b_norm)
    r, g, b = scale_rgb(r_norm, g_norm, b_norm)
    return img, r, g, b

def test_conversions(chan_fwd, chan_inv):
    assert chan_fwd.shape == chan_inv.shape, 'Shapes do not match'
    assert chan_fwd.dtype == chan_inv.dtype, 'Dtypes do not match'
    # only 2 decimal places are considered
    assert np.allclose(chan_fwd, chan_inv, atol=1e-2), 'Values do not match'

if __name__ == '__main__':

    test_dir = os.path.join(root, 'step2', 'test')
    os.makedirs(test_dir, exist_ok=True)
    img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')])
    img = load_image(img_paths[0])
    r, g, b = extract_rgb(img)
    r_Lstar, g_Lstar, b_Lstar = forward_pipeline(r, g, b)
    img_inv, r_inv, g_inv, b_inv = inverse_pipeline(r_Lstar, g_Lstar, b_Lstar)
    test_conversions(r, r_inv)
    test_conversions(g, g_inv)
    test_conversions(b, b_inv)
    
    img.save(os.path.join(test_dir, 'original.png'))
    img_inv.save(os.path.join(test_dir, 'inverse.png'))