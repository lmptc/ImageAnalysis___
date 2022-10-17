# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 17:40:14 2022

@author: Sommer Lab
"""

import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.patches as patches
import lmfit
from lmfit import Parameters
import configparser  

class AndorZyla: # Andor Zyla 5.5  
    def __init__(self):
        self.quantum_eff = .62 #Andor Zyla 
        self.sensitivity = .45
        
class FLIRchameleon: #FLIR Chameleon3 CM3-U3-13S2M  
    def __init__(self):
        self.quantum_eff = .50 
        self.sensitivity = .45 # numbers need to be updated       
    
class ExperimentParams:
    def __init__(self, config, picturesPerIteration=1):
        """        
        Parameters
        ----------
        config : TYPE
            DESCRIPTION.
        picturesPerIteration : TYPE, optional
            picturesPerIteration : int
                How many pictures does Cicero take in each iteration? (typically 2 or 3) The default is 1.
        """
        #input parameters from config file
        self.config =  config
        self.picturesPerIteration = picturesPerIteration
        self.number_of_pics = int(config['Acquisition']['NumberinKineticSeries'])
        assert self.number_of_pics % picturesPerIteration == 0, "Number of pictures should be a multiple of picturesPerIteration" # checks for error
        self.number_of_iterations = int(self.number_of_pics / picturesPerIteration)
        
        self.height1 = int(config['FullImage']['VerticalEnd']) - int(config['FullImage']['VerticalStart']) + 1
        self.width1 = int(config['FullImage']['HorizontalEnd']) - int(config['FullImage']['HorizontalStart']) + 1 
        self.bin_horizontal = int(config['FullImage']['HorizontalBin'])
        self.bin_vertical = int(config['FullImage']['VerticalBin'])
        
        # image height, width and range after binning
        self.height = int(self.height1/self.bin_vertical)
        self.width = int(self.width1/self.bin_horizontal)
        self.xmin=int(0)  #origin placed at zero by python
        self.ymin=int(0)  #origin placed at zero by python
        self.xmax=self.width-1
        self.ymax=self.height-1 
        self.number_of_pixels = self.height*self.width
        
        self.data_type = np.int16
        self.ready_to_save = 'true'

        self.camera=AndorZyla()
        
        P_MOT_beam = 14e-3 #power per MOT beam, roughly 1/4 of what goes into octopus
        self.pixel_size = 1/(22.2e3) #obtained from measurement of magnification using ambient light
        r_beam = .01 #meters
        I1 = 2*P_MOT_beam/(np.pi*r_beam**2)
        I = 6*I1
        Isat = 25 #W/m^2
        self.s = I/Isat
        self.gamma = 36.898e6
        self.delta = 26e6*2*np.pi
        self.R_scat = self.gamma*.5*self.s/(1+self.s+(2*self.delta/self.gamma)**2)
        self.t_exp = float(config['Acquisition']['ExposureTime']) 
        aperture_radius =  6.73/2 #in mm, the radius of the iris placed at the lens directly after the chamber where the MOT starts to get blocked
        cos_theta = 150/np.sqrt(aperture_radius**2+150**2)
        self.solid_angle = 2*np.pi*(1-cos_theta)

def LoadConfigFile(dataFolder=".", configFileName='config.cfg'): 
        config_file = dataFolder + "//" + configFileName
        config = configparser.ConfigParser()
        config.read(config_file)
        return config
    
def LoadTOF(dataFolder='.', TOF_filename='TOF_list.txt', units_of_tof='ms'):    
    tof_list = np.loadtxt(dataFolder + "//" + TOF_filename)
    return tof_list, units_of_tof 
    
# to load a series of FLIR .pgm images into a 4D numpy array
def loadSeriesPGM(params, root_filename, number_of_pics=1, n_params=0, data_folder= "." , background_file_name= "background.pgm"):
# n_params is the number of embedded image information fields which are checked, vales between 0 to 10, default 0 
# zero is black, maxval is white
# maxval that our files show is 65536 because of the 2 byte packets but the ADC is 12 bit so
# i think in practice pixel values are between 0 and 4096 
# the standard binary .pgm file starts with string P5 \n width space height \n 65535 \n
    for x in range(number_of_pics): 
        filename = data_folder + "\\" + root_filename + str(x+1)+ ".pgm"  
        
        with open(filename, 'r') as f:
            filetype = f.readline()
            if filetype.strip() != "P2":
                raise Exception("wrong format, should be P2")
                return
            
            res = f.readline().split()
            cols = int(res[0])
            rows = int(res[1])
            pixel_number = rows*cols
            print("Resolution is: Columns=",cols," and Rows=", rows, ".")
            print("Total number of pixels is=", pixel_number)
            maxval = f.readline()
            
            datastrings = f.read().split()
            data = [x for x in map(int, datastrings)]
            
            data2D = np.reshape(data,(rows,cols))
            
            #Throw away first row or two containing the first 20 numbers
            
            # if maxval == 65535:
            #     maxval = 4096
               
            # if filetype == "P5\n":
            #     f.close
            #     with open(filename, 'br') as f:
                
                
            # nextline = f.readline()
            # print(nextline)    
            
            # header = f.read(18) #this number should be 18 when width is 3 digits and height is 4 digits
            # print(header)
            # meta = f.read(4*n_params+0)
            # print(meta)
            # img_tmp = f.read()
            # img = np.frombuffer(img_tmp, dtype=np.int16)
            # print("image size is:", 1288*964)
            # print(np.size(img))
            # print(img)
        
        
        # data_array_corrected = data_array - background_array #spool file that is background corrected
        # image_array[x*params.number_of_pixels: (x+1)*params.number_of_pixels] = data_array
        # print("max value before background subtraction = "+str(np.max(image_array)))
        # image_array_corrected[x*params.number_of_pixels: (x+1)*params.number_of_pixels] = data_array_corrected
            
        # reshape the total_image_array_corrected into a 4D array
        # outermost dimension's size is equal to the number of iterations, 
        # 2nd outer dimensions size is number of pictures per iteration
        # 3rd dimensions size is equal to the height of the images
        #print(params.number_of_iterations, params.picturesPerIteration, params.height, params.width)
    #images = np.reshape(image_array_corrected,(params.number_of_iterations, params.picturesPerIteration, params.height, params.width))
    #return images      

# to load a series of non-spooled Andor .dat images into a 4D numpy array
def LoadSeries(params, root_filename, data_folder= "." , background_file_name= "background.dat"):
        """
        Parameters
        ----------
        params : ExperimentParams object
            Contains config, number_of_pixels, and other parameters    
        data_folder : string
            path to the folder with the spooled series data, and the background image
        background_file_name : string
            name of background image, assumed to be in the data_folder
       
        Returns
        -------
        4D array of integers giving the background-subtracted camera counts in each pixel.
        Format: images[iterationNumber, pictureNumber, row, col]
    
        """
        background_array = np.zeros(params.number_of_pixels)
        #Load background image into background_array
        if background_file_name:
            background_img = data_folder + "//" + background_file_name
            file=open(background_img,"rb")
            content=file.read()
            background_array = np.frombuffer(content, dtype=params.data_type)
            background_array = background_array[0:params.number_of_pixels]
            file.close()

        #read the whole kinetic series, bg correct, and load all images into a numpy array called image-array_correcpted
        image_array = np.zeros(shape = (1, params.number_of_pixels * params.number_of_pics))[0] 
        image_array_corrected = np.zeros(shape = (1, params.number_of_pixels * params.number_of_pics))[0]
        for x in range(params.number_of_pics): 
            filename = data_folder + "\\" + root_filename + str(x+1)+ ".dat"    
            file = open(filename,"rb")
            content = file.read()
            data_array = np.frombuffer(content, dtype=params.data_type)
            data_array = data_array[0:params.number_of_pixels]
            data_array_corrected = data_array - background_array 
            image_array[x*params.number_of_pixels: (x+1)*params.number_of_pixels] = data_array
            print("max value before background subtraction = "+str(np.max(image_array)))
            image_array_corrected[x*params.number_of_pixels: (x+1)*params.number_of_pixels] = data_array_corrected
            #print("max value after background subtraction = "+str(np.max(image_array_corrected)))
            
        # reshape the total_image_array_corrected into a 4D array
        # outermost dimension's size is equal to the number of iterations, 
        # 2nd outer dimensions size is number of pictures per iteration
        # 3rd dimensions size is equal to the height of the images
        #print(params.number_of_iterations, params.picturesPerIteration, params.height, params.width)
        images = np.reshape(image_array_corrected,(params.number_of_iterations, params.picturesPerIteration, params.height, params.width))
        return images

def LoadSpooledSeries(params, data_folder= "." , background_file_name= "spool_background.dat"):
        """
        Parameters
        ----------
        params : ExperimentParams object
            Contains config, number_of_pixels, and other parameters    
        data_folder : string
            path to the folder with the spooled series data, and the background image
        background_file_name : string
            name of background image, assumed to be in the data_folder
       
        Returns
        -------
        4D array of integers giving the background-subtracted camera counts in each pixel.
        Format: images[iterationNumber, pictureNumber, row, col]
    
        """
        background_array = np.zeros(params.number_of_pixels)
        #Load background image into background_array
        if background_file_name:
            background_img = data_folder + "//" + background_file_name
            file=open(background_img,"rb")
            content=file.read()
            background_array = np.frombuffer(content, dtype=params.data_type)
            background_array = background_array[0:params.number_of_pixels]
            file.close()

        #read the whole kinetic series, bg correct, and load all images into a numpy array called image-array_correcpted
        image_array = np.zeros(shape = (1, params.number_of_pixels * params.number_of_pics))[0] 
        image_array_corrected = np.zeros(shape = (1, params.number_of_pixels * params.number_of_pics))[0]
        spool_number = '0000000000'
        for x in range(params.number_of_pics): 
            filename = data_folder + "\\"+ str(x)[::-1] + spool_number[0:(10-len(str(x)))]+"spool.dat"    
            file = open(filename,"rb")
            content = file.read()
            data_array = np.frombuffer(content, dtype=params.data_type)
            data_array = data_array[0:params.number_of_pixels] # a spool file that is not bg corrected
            data_array_corrected = data_array - background_array #spool file that is background corrected
            image_array[x*params.number_of_pixels: (x+1)*params.number_of_pixels] = data_array
            print("max value before background subtraction = "+str(np.max(image_array)))
            image_array_corrected[x*params.number_of_pixels: (x+1)*params.number_of_pixels] = data_array_corrected
            #print("max value after background subtraction = "+str(np.max(image_array_corrected)))
            
        # reshape the total_image_array_corrected into a 4D array
        # outermost dimension's size is equal to the number of iterations, 
        # 2nd outer dimensions size is number of pictures per iteration
        # 3rd dimensions size is equal to the height of the images
        #print(params.number_of_iterations, params.picturesPerIteration, params.height, params.width)
        images = np.reshape(image_array_corrected,(params.number_of_iterations, params.picturesPerIteration, params.height, params.width))
        return images

def CountsToAtoms(params, counts):
    """
    Convert counts to atom number for fluorescence images
    
    Parameters
    ----------
    params : ExperimentParams object
        
    counts : array or number
        Camera counts from fluorescence image
        
    Returns
    -------
    Atom number (per pixel) array in same shape as input counts array

    """
    return  (4*np.pi*counts*params.camera.sensitivity)/(params.camera.quantum_eff*params.R_scat*params.t_exp*params.solid_angle)
    

def ShowImages3d(images):
    """
    Draws a grid of images

    Parameters
    ----------
    images : 3d Array

    """
    iterations, height, width = np.shape(images)
    #print(iterations,picturesPerIteration)
    #imax = np.max(images)
    #imin = np.min(images)
    
    for it in range(iterations):
        print(it)
        ax = plt.subplot(iterations,1,1)
        ax.imshow(images[it,:,:],cmap="gray")#,vmin = imin, vmax=imax)
    plt.tight_layout()
    plt.show()

def ShowImages(images):
    """
    Draws a grid of images

    Parameters
    ----------
    images : 4d Array

    """
    iterations, picturesPerIteration, height, width = np.shape(images)
    #print(iterations,picturesPerIteration)
    #imax = np.max(images)
    #imin = np.min(images)
    
    for it in range(iterations):
        for pic in range(picturesPerIteration):
            print(it,pic)
            ax = plt.subplot(iterations, picturesPerIteration, it*picturesPerIteration + pic+1)
            ax.imshow(images[it,pic,:,:],cmap="gray")#,vmin = imin, vmax=imax)
    plt.tight_layout()
    plt.show()
    
def ShowImagesTranspose(images, autoscale=True):
    """
    Draws a grid of images

    Parameters
    ----------
    images : 4d Array
    
    autoscale: boolean
        True: scale each image independently

    """
    iterations, picturesPerIteration, height, width = np.shape(images)
    
    #print(iterations,picturesPerIteration)
    
    if not autoscale:
        imax = np.max(images)
        imin = np.min(images)
    
    for it in range(iterations):
        for pic in range(picturesPerIteration):
            print(it,pic)
            ax = plt.subplot(picturesPerIteration, iterations, pic*iterations + it+1)
            if autoscale:
                ax.imshow(images[it,pic,:,:],cmap="gray")#,vmin = imin, vmax=imax)
            else:
                ax.imshow(images[it,pic,:,:],cmap="gray",vmin = imin, vmax=imax)
    plt.tight_layout()
    plt.show()
 
 

# simple, no analysis, list of pics => normalized
def ImageTotals(images):
    """
    
    ----------
    images : 4D array of images
    
    Returns
    -------
    2D Array of sums over the images

    """
    
    shape1 = np.shape(images)
    assert len(shape1) == 4, "input array must be 4D"
    
    shape2 = shape1[:-2]
    totals = np.zeros(shape2)
    
    for i in range(shape2[0]):
        for j in range(shape2[1]):
            totals[i,j] = np.sum(images[i,j,:,:])
    return totals
    
def temp(images):    
    atoms_x = np.zeros((params.number_of_pics, params.width))
    atoms_y = np.zeros((params.number_of_pics, params.height))   
    
    #Sum the columns of the region of interest to get a line trace of atoms as a function of x position
    for i in range(params.number_of_iterations):
        for j in range(params.picturesPerIteration) :
            im_temp = images[i, j, params.ymin:params.ymax, params.xmin:params.xmax]
            count_x = np.sum(im_temp,axis = 0) #sum over y direction/columns 
            count_y = np.sum(im_temp,axis = 1) #sum over x direction/rows
            atoms_x[i] = (4*np.pi*count_x*params.sensitivity)/(params.quantum_eff*params.R_scat*params.t_exp*params.solid_angle)
            atoms_y[i] = (4*np.pi*count_y*params.sensitivity)/(params.quantum_eff*params.R_scat*params.t_exp*params.solid_angle)
            print("num_atoms_vs_x in frame" , i, "is: {:e}".format(np.sum(atoms_x[i])))
            print("num_atoms_vs_y in frame" , i, "is: {:e}".format(np.sum(atoms_y[i])))
    
    if atoms_x != atoms_y:
        print("atom count calculated along x and along y do NOT match")

    atoms_x_max = max(atoms_x)
    atoms_y_max = max(atoms_y)
    atoms_max = max(atoms_x_max, atoms_y_max)        
    
    return atoms_x, atoms_y, atoms_max        
        #  output_array = np.array((number_of_iteration, outputPicsPerIteration, height, width)


def absImaging(images):
    iterations, picturesPerIteration, height, width = np.shape(images)
    
    signal = np.zeros((iterations, 1, height, width))
    
    if picturesPerIteration==4:
        for i in range(iterations-1):
            # signal is column density along the imaging path
            signal[i,0,:,:] = (images[i,1,:,:] - images[i,3,:,:]) / (images[i,2,:,:] - images[i,3,:,:])
    else:
        print("This spooled series does not have the correct number of exposures per iteration for Absorption Imaging")        
        
    return signal


def Gaussian(x, a, mu, w0, c):
    return a*np.exp(-(x-mu)**2/(2*w0**2)) + c


#Gaussian_fit takes an array of the summed atom numbers. It outputs a gaussian width, a full fit report, and an x axis array
def Gaussian_fit(images, params, slice_array, tof, units_of_tof, dataFolder='.'):
    xposition = params.pixel_size*np.linspace(0, len(slice_array),len(slice_array))
    aguess = np.max(slice_array)
    muguess = params.pixel_size*np.where(slice_array == np.max(slice_array))[0][0]
    w0guess = params.pixel_size*len(slice_array)/4 #the standard dev. of the Gaussian
    cguess = np.min(slice_array)
    paramstemp = Parameters()
    paramstemp.add_many(
        ('a', aguess,True, None, None, None),
        ('mu', muguess, True, None, None, None),
        ('w0', w0guess, True, None, None, None),
        ('c', cguess, True, None, None, None),
        )
        
    model = lmfit.Model(Gaussian)
    result = model.fit(slice_array, x=xposition, params = paramstemp)
    gwidth = abs(result.params['w0'].value)
    return  gwidth, result, xposition


#Here I call the Gaussian_fit function on all of the expanding cloud pictures to output widths for all of them.
    half_of_pictures = int(params.number_of_pics/2)
    gaussian_widths_x = np.zeros(half_of_pictures)
    gaussian_widths_y = np.zeros(half_of_pictures)
    num_atoms_vs_x, num_atoms_vs_y, atoms_max = temp(slice_array)
 
    for i in range(half_of_pictures):
        fittemp_x = Gaussian_fit(num_atoms_vs_x[2*i+1,:])
        fittemp_y = Gaussian_fit(num_atoms_vs_y[2*i+1,:])
        gaussian_widths_x[i] = fittemp_x[0]
        gaussian_widths_y[i] = fittemp_y[0]
        
        
        if params.ready_to_save == 'true':
        
            #save Gaussian fit in x direction plot
            fit0_x = Gaussian_fit(num_atoms_vs_x[2*i+1,:])
            plt.figure()
            plt.rcParams.update({'font.size':9})
            plt.title('TOF = {}'.format(tof[i])+units_of_tof+' horizontal plot, standard dev. = {}m'.format(round(abs(fit0_x[0]), 5)))
            plt.xlabel("Position (m)")
            plt.ylabel("Number of atoms in MOT")
            plt.plot(fit0_x[2], num_atoms_vs_x[2*i+1,:], 'g.', label='Signal')
            plt.plot(fit0_x[2], fit0_x[1].best_fit, 'b', label='Fit')
            plt.legend()
            plt.tight_layout()
            plt.savefig(dataFolder +r'\TOF = {}'.format(tof[i])+units_of_tof+' horizontal plot.png', dpi = 300)
            plt.close()  
            
            #save Gaussian fit in y direction plot
            fit0_y = Gaussian_fit(num_atoms_vs_y[2*i+1,:])
            plt.figure()
            plt.title('TOF = {}'.format(tof[i])+units_of_tof+' vertical plot, standard dev. = {}m'.format(round(abs(fit0_y[0]), 5)))
            plt.xlabel("Position (m)")
            plt.ylabel("Number of atoms in MOT")
            plt.plot(fit0_y[2], num_atoms_vs_y[2*i+1,:], 'g.', label='Signal')
            plt.plot(fit0_y[2], fit0_y[1].best_fit, 'b', label='Fit')
            plt.legend()
            plt.tight_layout()
            plt.savefig(dataFolder+r'\TOF = {}'.format(tof[i])+units_of_tof+' vertical plot.png', dpi = 300)
            plt.close()
           
            #save the picture from Andor
            plt.figure()
            plt.title("Signal inside red rectangle")
            plt.imshow(images[2*i+1,params.ymin:params.ymax,params.xmin:params.xmax],cmap="gray", origin="lower",interpolation="nearest",vmin=np.min(images),vmax=np.max(images))
            plt.savefig(dataFolder+r'\TOF = {}'.format(tof[i])+units_of_tof+' signal inside red rectangle.png', dpi = 300)
            plt.close()
            

    gaussian_widths_x = np.flip(gaussian_widths_x)
    gaussian_widths_y = np.flip(gaussian_widths_y)


#Here we import the relevant TOF file and combine it with the gaussian widths
    widths_tof_x = np.zeros((len(gaussian_widths_x),2))
    widths_tof_y = np.zeros((len(gaussian_widths_y),2))

    for i in range(len(gaussian_widths_x)):
         widths_tof_x[i] = (gaussian_widths_x[i], tof[i])
         widths_tof_y[i] = (gaussian_widths_y[i], tof[i])

# save the data in a csv file
    if params.ready_to_save =='true':
        csvfilename_x = dataFolder+r"\widths_vs_tof_x.csv"
        csvfilename_y = dataFolder+r"\widths_vs_tof_y.csv"
        np.savetxt(csvfilename_x, widths_tof_x, delimiter = ",") 
        np.savetxt(csvfilename_y, widths_tof_y, delimiter = ",") 


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def exponential(x, m, t, b):
    return m * np.exp(-t * x) + b
    
# def fit_decay():
    
    #fit parameters
    # value = atom_max*np.exp(-1)
    # emin1 = find_nearest(array, value)
    # finder = np.where(N_atoms == emin1)
#     array_number = int(finder[0])
#     #print("array_number: ", array_number)
#     #######################################This is the time for the function to reach e**-1 of max value
#     emin1_time = Picture_Time[array_number]
#     atom_fraction = N_atoms/max(N_atoms)
    
    
    
#     p0 = (count_spooled.atom_max, 1/emin1_time, 0) # start with values near those we expect
#     params, cv = scipy.optimize.curve_fit(exp, Picture_Time, N_atoms, p0)
#     m, t, b = params
    
#     #Quality of fit
#     squaredDiffs = np.square(N_atoms - exp(Picture_Time, m, t, b))
#     squaredDiffsFromMean = np.square(N_atoms - np.mean(N_atoms))
#     rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
#     print(f"R² = {rSquared}")
#     print(f"Y = {m} * e^(-{t} * x) + {b}")
        
#     # plot the results
#     plt.plot(Picture_Time, N_atoms, '.', label="data")
#     plt.plot(Picture_Time, exp(Picture_Time, m, t, b), '--', label="fitted")
#     plt.title("Fitted Exponential Curve", fontsize = 18)
#     if m < 10**5:
#         pressure = t/(6.4*10**7)
#         print("It appears that this decay occurs in the low density limit.")
#         print("Based off of this assumption, the background pressure of the vacuum chamber appears to be {pressure} torr.")
    
    
# def fit_load():
#     p0 = (atom_max, (1-math.log(math.e-1))/emin1_time, atom_max) # start with values near those we expect
#     params, cv = scipy.optimize.curve_fit(exp, Picture_Time, N_atoms, p0)
#     m, t, b = params
    
#     #Quality of fit
#     squaredDiffs = np.square(N_atoms - exponential(Picture_Time, m, t, b))
#     squaredDiffsFromMean = np.square(N_atoms - np.mean(N_atoms))
#     rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
#     print(f"R² = {rSquared}")
#     print(f"Y = {m} * e^(-{t} * x) + {b}")
#     # plot the results
#     plt.plot(Picture_Time, N_atoms-min(N_atoms), '.', label="data")
#     plt.plot(Picture_Time, Load_Decay(Picture_Time, m, t, b)-min(N_atoms), '--', label="fitted")
#     plt.title("Atoms Loaded Over Time", fontsize = 20)    

if __name__ == "__main__":
    #TESTING Script:

    config = LoadConfigFile()
    params = ExperimentParams(config, picturesPerIteration=1)        

    # CountsToAtoms(params, images[3,4,:,:])

    loadSeriesPGM(params, root_filename="P2-nonchecked", number_of_pics=1, n_params=0, data_folder= "." , background_file_name= "")       
        

    
    # print("Number of iterations=",params.number_of_iterations)
    
    # images1 = LoadSpooledSeries(params, data_folder="abs img test 1_17", background_file_name="")
    # signal1 = absImaging(images1)
    # #atomsPerPixel = CountsToAtoms(params, counts)
    
    # #ShowImagesTranspose(atomsPerPixel)
    
    # print(np.shape(images1))
    # print(np.shape(signal1))
    
    # ShowImagesTranspose(images1, False)
    
    #atomNumbers = ImageTotals(atomsPerPixel)
    
    #print(atomNumbers)
    
    #number_of_pics = int(config['Acquisition']['NumberinKineticSeries'])
    #print(number_of_pics)
    
    # images = LoadSpooledSeries(config, data_folder= "." , background_file_name= "spool_background.dat", picturesPerIteration=3)
    # #images = LoadNonSpooledSeries(...)
    
    # atoms_per_pixel_images = GetCountsFromRawData(images,config)
    
    # #analyse it somehow:
    # #Find the total number of atoms at the end of each iteration
    # atom_numbers = GetTotalNumberofAtoms(atoms_per_pixel_images)
    
    # print("Number of atoms in 2nd picture of iteration 0:",atom_numbers[0][1])
    
    # #Do a fit:
    # result = DoExponentialFit(atom_numbers[:][1])
    
    # print(np.shape(images))
    
    
    
    

#     #here I am making a plot of the first gaussian with the fit for reference
#     fit0_x = Gaussian_fit(num_atoms_vs_x[PreviewIndex,:])
#     plt.figure()
#     plt.rcParams.update({'font.size':9})
#     plt.title('Atoms TOF horizontal example plot, standard dev. = {}m'.format(round(fit0_x[0], 5)))
#     plt.xlabel("Position (m)")
#     plt.ylabel("Number of atoms in MOT")
#     plt.plot(fit0_x[2], num_atoms_vs_x[PreviewIndex,:], 'g.', label='Signal')
#     plt.plot(fit0_x[2], fit0_x[1].best_fit, 'b', label='Fit')
#     # plt.xlim(fit0_x[2][0], fit0_x[2][-1])
#     plt.legend()
#     plt.tight_layout()
#     # plt.savefig(folder_name+r"\Horizontal Gaussian Example.png", dpi = 300)

#     fit0_y = Gaussian_fit(num_atoms_vs_y[PreviewIndex,:])
#     plt.figure()
#     plt.title('Atoms TOF vertical example plot, standard dev. = {}m'.format(round(fit0_y[0], 5)))
#     plt.xlabel("Position (m)")
#     plt.ylabel("Number of atoms in MOT")
#     plt.plot(fit0_y[2], num_atoms_vs_y[PreviewIndex,:], 'g.', label='Signal')
#     plt.plot(fit0_y[2], fit0_y[1].best_fit, 'b', label='Fit')
#     # plt.xlim(fit0_y[2][0], fit0_y[2][-1])
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#     # plt.savefig(folder_name+r"\Vertical Gaussian Example.png", dpi = 300)
    
