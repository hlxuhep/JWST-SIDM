#Modify the path to a directory on your machine
import os
os.environ["CRDS_PATH"] = "/home/hailin/Documents/CRDS"
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"

# Packages that allow us to get information about objects:
import asdf
import copy
import shutil

# Numpy library:
import numpy as np

# For downloading data
import requests

# Astropy tools:
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.visualization import ImageNormalize, ManualInterval, LogStretch

import matplotlib.pyplot as plt
import matplotlib as mpl

def download_files(files, output_directory, force=False):
    """Given a tuple or list of tuples containing (URL, filename),
    download the given files into the current working directory.
    Downloading is done via astropy's download_file. A symbolic link
    is created in the specified output dirctory that points to the
    downloaded file.
    
    Parameters
    ----------
    files : tuple or list of tuples
        Each 2-tuple should contain (URL, filename), where
        URL is the URL from which to download the file, and
        filename will be the name of the symlink pointing to
        the downloaded file.
        
    output_directory : str
        Name of the directory in which to create the symbolic
        links to the downloaded files
        
    force : bool
        If True, the file will be downloaded regarless of whether
        it is already present or not.
                
    Returns
    -------
    filenames : list
        List of filenames corresponding to the symbolic links
        of the downloaded files
    """
    # In the case of a single input tuple, make it a
    # 1 element list, for consistency.
    filenames = []
    if isinstance(files, tuple):
        files = [files]
        
    for file in files:
        filenames.append(file[1])
        if force:
            print('Downloading {}...'.format(file[1]))
            demo_file = download_file(file[0], cache='update')
            # Make a symbolic link using a local name for convenience
            if not os.path.islink(os.path.join(output_directory, file[1])):
                os.symlink(demo_file, os.path.join(output_directory, file[1]))
        else:
            if not os.path.isfile(os.path.join(output_directory, file[1])):
                print('Downloading {}...'.format(file[1]))
                demo_file = download_file(file[0], cache=True)
                # Make a symbolic link using a local name for convenience
                os.symlink(demo_file, os.path.join(output_directory, file[1]))
            else:
                print('{} already exists, skipping download...'.format(file[1]))
                continue
    return filenames    

def plot_jump(signal, jump_group, xpixel=None, ypixel=None, slope=None):
    """Function to plot the signal up the ramp for a
    pixel and show the location of flagged jumps.
    
    Parameters
    ----------
    signal : numpy.ndarray
        1D array of signal values
        
    jump_group : list
        List of boolean values whether a jump is present or
        not in each group
        
    slope : numpy.ndarray
        1D array of signal values constructed from the slope
    """
    groups = np.arange(len(signal))
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot()

    plt.plot(groups, signal, marker='o', color='black')
    plt.plot(groups[jump_group], signal[jump_group], marker='o', color='red',
             label='Flagged Jump')
    
    if slope is not None:
        plt.plot(groups, slope, marker='o', color='blue', label='Data from slope')
        
    plt.legend(loc=2)

    plt.xlabel('Groups')
    plt.ylabel('Signal (DN)')
    fig.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    if xpixel and ypixel:
        plt.title('Pixel ('+str(xpixel)+','+str(ypixel)+')')

def plot_jumps(signals, jump_groups, pixel_loc, slopes=None):
    """Function to plot the ramp and show the jump location
    for several pixels. For simplicity, let's force the input
    number of pixels to be a square. 
    
    Parameters
    ----------
    signals : numpy.ndarray
        2D array (groups x pix) of signal values
        
    jump_groups : numpy.ndarray
        2D array containing boolean entries for each group of
        each pixel, describing where the jumps were found
        
    pixel_loc : list
        List of 2-tuples containing the (x, y)
        location of the pixels with the jumps
        
    slopes : numpy.ndarray
        2D array (groups x pix) of linear signal values
        If not None, these will be overplotted onto the
        plots of signals
    """
    num_group, num_pix = signals.shape
    root = np.sqrt(num_pix)
    if int(root + 0.5) ** 2 != num_pix:
        raise ValueError('Number of pixels input should be a square.')
    
    root = int(root)
    groups = np.arange(num_group)
    fig, axs = plt.subplots(root, root, figsize=(10, 10))

    for index in range(len(pixel_loc)):
        i = int(index % root)
        j = int(index / root)
        axs[i, j].plot(groups, signals[:, index], marker='o', color='black')
        j_grp = jump_groups[:, index]
        axs[i, j].plot(groups[j_grp], signals[j_grp, index],
                       marker='o', color='red')
        
        if slopes is not None:
            axs[i, j].plot(groups, slopes[:, index], marker='o', color='blue')
        
        axs[i, j].set_title('Pixel ({}, {})'.format(pixel_loc[index][1], pixel_loc[index][0]))
        
    plt.xlabel('Groups')
    plt.ylabel('Signal (DN)')
    fig.tight_layout()

def plot_ramp(groups, signal, xpixel=None, ypixel=None, title=None):
    """Function to plot the up the ramp signal for a pixel.
    
    Parameters
    ----------
    groups : numpy.ndarray
        1D array of group numbers. X-axis values.
        
    signal : numpy.ndarray
        1D array of pixel signal values.
        
    xpixel : int
        X-coordinate of the pixel being plotted. Used for legend only.
        
    ypixel : int
        Y-coordinate of the pixel being plotted. Used for legend only.
        
    title : str
        String to use for the plot title
    """    
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot()
    if xpixel and ypixel:
            plt.plot(groups, signal, marker='o',
                     label='Pixel ('+str(xpixel)+','+str(ypixel)+')') 
            plt.legend(loc=2)

    else:
        plt.plot(groups, signal, marker='o')
        
    plt.xlabel('Groups')
    plt.ylabel('Signal (DN)')
    fig.tight_layout()
    plt.subplots_adjust(left=0.15)
    
    if title:
        plt.title(title)

def plot_ramps(groups, signal1, signal2, label1=None, label2=None, title=None):
    """Function to plot the up the ramp signal for two pixels
    on a single plot.
    
    Parameters
    ----------
    groups : numpy.ndarray
        1D array of group numbers. X-axis values.
        
    signal1 : numpy.ndarray
        1D array of signal values for first pixel
        
    signal2 : numpy.ndarray
        1D array of signal values for second pixel
        
    label1 : str
        Label to place in the legend for pixel1
        
    label2 : str
        Label to place in the legend for pixel2
        
    title : str
        String to place in the title of the plot
    """    
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot()
    if label1:
        plt.plot(groups, signal1, marker='o', color='black', label=label1)
    else:
        plt.plot(groups, signal1, marker='o', color='black')
    if label2:
        plt.plot(groups, signal2, marker='o', color='red', label=label2)
    else:
        plt.plot(groups, signal2, marker='o', color='red')
    if label1 or label2:
        plt.legend(loc=2)
        
    plt.xlabel('Groups')
    plt.ylabel('Signal (DN)')
    fig.tight_layout()
    plt.subplots_adjust(left=0.15)
    plt.subplots_adjust(top=0.95)
    
    if title:
        plt.title(title)

def show_image(data_2d, vmin, vmax, xpixel=None, ypixel=None, title=None, unit = 'DN'):
    """Function to generate a 2D, log-scaled image of the data, 
    with an option to highlight a specific pixel (with a red dot).
    
    Parameters
    ----------
    data_2d : numpy.ndarray
        Image to be displayed
        
    vmin : float
        Minimum signal value to use for scaling
        
    vmax : float
        Maximum signal value to use for scaling
        
    xpixel : int
        X-coordinate of pixel to highlight
        
    ypixel : int
        Y-coordinate of pixel to highlight
        
    title : str
        String to use for the plot title
    """
    norm = ImageNormalize(data_2d, interval=ManualInterval(vmin=vmin, vmax=vmax),
                          stretch=LogStretch())
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(data_2d, origin='lower', norm=norm)
    
    if xpixel and ypixel:
        plt.plot(xpixel, ypixel, marker='o', color='red', label='Selected Pixel')

    fig.colorbar(im, label=unit)
    plt.xlabel('Pixel column')
    plt.ylabel('Pixel row')
    if title:
        plt.title(title)

def side_by_side(data1, data2, vmin, vmax, title1=None, title2=None, title=None, unit = 'DN'):
    """Show two images side by side for easy comparison. Optionally highlight
    a given pixel with a red dot.
    
    Parameters
    ----------
    data1 : numpy.ndarray
        First image to be displayed
        
    data2 : numpy.ndarray
        Second image to be displayed
        
    vmin : float
        Minimum signal value to use for scaling
        
    vmax : float
        Maximum signal value to use for scaling
            
    title1 : str
        Title to use for first (left) plot
        
    title2 : str
        Title to use for the second (right) plot

    title : str
        String to use for the plot title
    """
    norm = ImageNormalize(data1, interval=ManualInterval(vmin=vmin, vmax=vmax),
                          stretch=LogStretch())

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 8))
    im = axes[0].imshow(data1, origin='lower', norm=norm)
    im = axes[1].imshow(data2, origin='lower', norm=norm)
    
    axes[0].set_xlabel('Pixel column')
    axes[0].set_ylabel('Pixel row')
    axes[1].set_xlabel('Pixel column')
    
    if title1:
        axes[0].set_title(title1)
    if title2:
        axes[1].set_title(title2)
        
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax, label=unit)
    
    if title:
        fig.suptitle(title)    

