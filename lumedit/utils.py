from PIL import Image
import numpy as np

def adjust_lightness(img, target_Lstar, tol=0.5, max_iter=10):
    '''Adjust the perceived lightness of an image to a target value
    
    Parameters
    ----------
    img : PIL.Image
        Image object
    target_Lstar : float
        Target L* value
    tol : float
        Tolerance for the error in the L* value
    max_iter : int
        Maximum number of iterations to reach the target L* value
        
    Returns
    -------
    img_adj : PIL.Image
        Adjusted image
    '''
    r, g, b = extract_rgb(img)
    r_norm, g_norm, b_norm = normalize_rgb(r, g, b)
    # linearize RGB values 
    r_lin = rgb2linear(r_norm) 
    g_lin = rgb2linear(g_norm) 
    b_lin = rgb2linear(b_norm) 
    # apply luminance coefficients
    r_lum = r_lin * 0.2126
    g_lum = g_lin * 0.7152
    b_lum = b_lin * 0.0722
    Y_avg = np.mean((r_lum + g_lum + b_lum)) # average luminance
    Y_target = Lstar2lum(target_Lstar) # target luminance based on L* value
    factor = (Y_target / Y_avg) # scaling factor for the whole image

    for i in range(max_iter):
        # adjust linear RGB values
        r_lin_adj = np.clip(r_lin * factor, 0, 1)
        g_lin_adj = np.clip(g_lin * factor, 0, 1)
        b_lin_adj = np.clip(b_lin * factor, 0, 1)
        # compute new total luminance and its L*
        Y_adj = r_lin_adj * 0.2126 + g_lin_adj * 0.7152 + b_lin_adj * 0.0722
        Lstar_adj = np.mean(lum2Lstar(Y_adj))
        # Compute error:
        error = target_Lstar - Lstar_adj
        if abs(error) < tol:
            break
        # update factor by a small fraction of the relative error
        factor *= (1 + 0.5 * error / target_Lstar)

    # convert back to sRGB
    r_norm_adj = linear2rgb(r_lin_adj)
    g_norm_adj = linear2rgb(g_lin_adj)
    b_norm_adj = linear2rgb(b_lin_adj)
    return norm_rgb2img(r_norm_adj, g_norm_adj, b_norm_adj)

def cmpt_perceived_lightness(img):
    '''Calculate the average luminance of an image
    
    Parameters
    ----------
    img : PIL.Image
        Image object
        
    Returns
    -------
    Lstar : float
        Average perceived lightness
    '''
    r_lum, g_lum, b_lum = compute_luminance(img)
    Y = r_lum + g_lum + b_lum
    Lstar = lum2Lstar(Y)
    # average perceived lightness for each pixel
    return np.mean(Lstar)
    
def compute_luminance(img):
    '''Calculate the luminance of an image
    
    Parameters
    ----------
    img : PIL.Image
        Image object
        
    Returns
    -------
    r_lum, g_lum, b_lum : np.array
        Luminance values for each channel
    '''
    r, g, b = extract_rgb(img)
    r_norm, g_norm, b_norm = normalize_rgb(r, g, b)
    # linearize RGB values and apply luminance coefficients
    r_lum = rgb2linear(r_norm) * 0.2126
    g_lum = rgb2linear(g_norm) * 0.7152
    b_lum = rgb2linear(b_norm) * 0.0722
    # sum of the three channels to get the luminance
    return r_lum, g_lum, b_lum


def load_image(path):
    '''Loads an image from a filename and removes the alpha channel
    
    Parameters
    ----------
    path : str
        Path to the image file
            
    Returns
    -------
    img : PIL.Image
        Image object
    '''
    img = Image.open(path)
    img = img.convert('RGB')
    return img

##############################################
# Conversion functions 
# RGB <--> normRGB <--> Luminance <--> L*

def extract_rgb(img):
    '''Extract RGB values from an image. Returns three arrays.'''
    return [np.array(channel, dtype=np.float32) for channel in img.split()]

def norm_rgb2img(r_norm, g_norm, b_norm):
    '''Convert normalized RGB values to an image. Returns an Image object.'''
    r = (np.array(r_norm) * 255).astype(np.uint8)
    g = (np.array(g_norm) * 255).astype(np.uint8)
    b = (np.array(b_norm) * 255).astype(np.uint8)
    img = Image.merge(
        'RGB', (
            Image.fromarray(r), 
            Image.fromarray(g), 
            Image.fromarray(b)
            )
    )
    return img

def normalize_rgb(r_arr, g_arr, b_arr):
    '''Normalize RGB values to [0, 1]. Returns three arrays.'''
    return r_arr / 255, g_arr / 255, b_arr / 255

def scale_rgb(r_norm, g_norm, b_norm):
    '''Scale RGB values to [0, 255]. Returns three arrays.'''
    return r_norm * 255, g_norm * 255, b_norm * 255

def rgb2linear(x):
    '''Convert normalized sRGB values (in [0, 1]) to linear light. Returns an array or a scalar.'''
    x = np.array(x)
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return lin

def linear2rgb(x):
    '''Convert a linear light value (0-1) to normalized sRGB. Returns an array or a scalar.'''
    x = np.array(x)
    rgb = np.where(x <= 0.0031308, 12.92 * x, 1.055 * (x ** (1/2.4)) - 0.055)
    return rgb
    
def lum2Lstar(x):
    '''Converts luminance value sto L* space. Returns an array or a scalar.'''
    x = np.array(x)
    return np.where(x <= 0.008856, 903.3 * x, 116 * np.power(x, 1/3) - 16)

def Lstar2lum(x):
    '''Converts L* values to luminance space. Returns an array or a scalar.'''
    x = np.array(x)
    threshold = 903.3 * 0.008856  # approximately 8
    return np.where(x <= threshold, x / 903.3, np.power((x + 16) / 116, 3))
