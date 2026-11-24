#importing all packages used within code
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

#reading in spectral data and skipping the first four rows as they do not contain useful numeric data
spectral_data = pd.read_csv("SpectralData_Hbeta.csv", skiprows = 4)

#setting plotting parameters for the size of all plots that will be produced
parameters = {"axes.labelsize": 14,
          "font.size": 13,
          "legend.fontsize": 13,
          "xtick.labelsize": 14,
          "ytick.labelsize": 14,
          "figure.figsize": [6,4]}
plt.rcParams.update(parameters)

#initialising counter variables to be used within the while loop
i = 0
frequency_column = 0
intensity_column = 1
observation_column = 0

#assigning constants to variables as they are used multiple times in calculation
lambda_e = 486.1e-9
c = 3e8

#creating an empty array to store all velocity associated data once it is calculated
velocities = []

#creating a function to calculate the parameters of a Gaussian linear fit line
def fit_func(frequency, a, mean, std, m, c):
    gaussian = a*np.exp(-(frequency - mean)**2/(2*std**2))
    line = m*frequency + c
    return gaussian + line

#setting up a while loop that will cycle through all thirty sets of data for intensity and frequency
while i < 30:

#assigning variables to store the columns of data for frequency and intensity, and also assigning a variable to the observation number for each frequency/ intensity data set
#these variables are looped so they store different data each loop cycle allowing thirty different plots to be generated
#frequency data is divided by 1*10^12 so that it is in THz
    frequency = spectral_data.iloc[:, frequency_column]/1e12
    intensity = spectral_data.iloc[:, intensity_column]
    observation_number = spectral_data.columns[observation_column]

#setting up initial guesses for the Gaussian linear fit, slope is estimated from the first and last data points for both frequency and intensity, intercept is estimated using y=mx+c,
#the mean and standard deviation are estimated using the numpy functions applied to the frequency data as the mean and standard deviation correspond to the x- axis
    slope = (intensity[999] - intensity[0])/(frequency[999] - frequency[0])
    intercept = intensity[0]-(slope*frequency[0])
    initial_guess = [20, np.mean(frequency), np.std(frequency), slope, intercept]

#assigning variables to store the array of fit parameters and the covariance matrix that is produced from fitting the Gaussian linear line to each plot
    po, po_cov = curve_fit(fit_func, frequency, intensity, initial_guess)

#outputting each Gaussian linear fit parameter with its uncertainty and unit for each plot
    print("Signal parameters:")
    print(f"Gaussian amplitude: {po[0]:.1f} A.U. +/- {np.sqrt(po_cov[0, 0]):.1f} A.U.")
    print(f"Observed frequency (mean): {po[1]:.1f} THz +/- {np.sqrt(po_cov[1, 1]):.1f} THz")
    print(f"Gaussian width (std) {po[2]:.1f} THz +/- {np.sqrt(po_cov[2, 2]):.1f} THz")
    print("Background estimate:")
    print(f"Slope: {po[3]:.2f} A.U./THz +/- {np.sqrt(po_cov[3, 3]):.2f} A.U./THz")
    print(f"Intercept: {po[4]:.0f} A.U. +/- {np.sqrt(po_cov[4, 4]):.0f} A.U.")

#plotting a graph of intensity against frequency for each data set and plotting the Gaussian linear fit on top of the data, adding axis labels, a title and a legend to each plot as well
    plt.plot(frequency, intensity, mew = 10, ms = 1, color = "blue", label = "data")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Intensity (A.U.)")
    plt.plot(frequency, fit_func(frequency, po[0], po[1], po[2], po[3], po[4]), color = "orange", label = "fit")
    plt.title(f"Observation {observation_number}")
    plt.legend()
    plt.show()

#calculating the observed wavelength using lambda=c/f for each data set where f is the observed frequency which is the mean of the Gaussian linear fit.
#also calculating the uncertainty in observed wavelength by propagating uncertainty in observed frequency, then outputting both values with their unit
    lambda_0 = c/(po[1]*1e12)
    lambda_0_uncertainty = c/((po[1]*1e12)**2)*(np.sqrt(po_cov[1, 1])*1e12)
    print(f"Observed wavelength: {lambda_0:.3e} m +/- {lambda_0_uncertainty:.3e} m")

#calculating the redshift velocity for each data set using v=c((x^2-1)/(x^2+1)) where x is observed wavelength divided by emitted wavelength which is 486.1 nm (the second line 
#in the Balmer series), the uncertainty in both x and x^2 are calculated through error propagation of the uncertainty in observed wavelength, the error in redshift velocity is then 
#calculated through error propagation of x^2, finally the redshift velocity and its associated uncertainty are outputted for each data set
    x = lambda_0/lambda_e
    x_uncertainty = (x/lambda_0)*lambda_0_uncertainty
    x_squared_uncertainty = 2*x*x_uncertainty
    redshift_velocity = c*((x**2-1)/(x**2+1))
    redshift_uncertainty = redshift_velocity*((2*((x_squared_uncertainty/(x**2))**2))**0.5)
    print(f"Redshift velocity: {redshift_velocity:.3e} m/s +/- {redshift_uncertainty:.3e} m/s")
    print("")

#each redshift velocity and its uncertainty are added to an array along with the observation number of the data that was used to calculate these values, creating a 2D array of 
#observation number, velocity and uncertainty for thirty data sets
    velocities.append([observation_number, redshift_velocity, redshift_uncertainty])

#increasing the counter variables to allow another cycle to be carried out, the variables for the columns of data are increased by two each time as in the data file
#every two columns are part of the same data set
    frequency_column += 2
    intensity_column += 2
    observation_column += 2
    i += 1

#converting observation numbers from strings to floats in the array of velocity data, observation numbers are stored in the first position of each 1D array within the 2D array
for elements in velocities:
    elements[0] = float(elements[0])

#reading in the distance data and skipping the first 5 rows as they do not contain useful numeric data, also noting that there are multiple whitespace tabs between some columns of data 
#so this is accounted for when reading the data
distance_data = pd.read_csv("dist_data.txt", skiprows = 5, sep = r"\s+")

#creating an empty array to store all distance associated data, observation numbers stored in column one, distances stored in column four and instrument responses stored in column five
#are then added to this array for each set of data, but in order to do this the columns of data must be converted to lists as they are read in as arrays and an array cannot be added 
#to an array, this created a 2D array of observation number, distance and instrument response for thirty data sets
distance_observations = []
distance_observations = distance_data.iloc[:,[2, 3, 4]].values.tolist()

#creating a new array that only contains the 1D arrays of the distance associated data that have acceptable instrument responses, if the instrument response which is stored in the third
#position in each 1D array is E (which denotes it is bad) then it is removed from the data, so a new 2D array is created with only twenty five data sets

distance_filtered = [elements for elements in distance_observations if elements[2] != "E"]

#creating an empty array to store all the final data needed to plot a graph of redshift velocity against distance with errorbars
plotting_data = []

#creating a dictionary of observation numbers, which are stored in position one in each 1D array within the 2D array of filtered distance data, and the corresponding distance, which is 
#stored in position two
observation_matching = {float(row[0]): float(row[1]) for row in distance_filtered}

#checks each 1D array in the 2D array of velocity data to see whether the observation number matches the observation number which is a key in the dictionary created of observation
#numbers and distances, if the observation numbers match then the observation number and its corresponding redshift velocity, redshift velocity uncertainty and distance are added 
#to the empty array of plotting data, this creates a 2D array of 25 data sets
for observation, velocity, uncertainty in velocities:
    if observation in observation_matching:
        plotting_data.append([observation, velocity, uncertainty, observation_matching[observation]])

#assigning all distances which are stored in position three of each 1D array in the 2D array of plotting data to the variable x as distance is plotted on the x- axis, assigning all
#redshift velocities which are stored in position one of each 1D array to the variable y as velocity is plotted on the y- axis, assigning all redshift velocity uncertainties which are 
#stored in position two of each 1D array to the variable error
x = [row[3] for row in plotting_data]
y = [row[1]/1e3 for row in plotting_data]
error = [row[2]/1e3 for row in plotting_data]

#assigning variables to store the array of fit parameters and the covariance matrix that is produced by fitting a weighted linear line to the plot of velocity against distance,
#the redshift velocity uncertainties are stored in the variable error as a list so must be converted to an array in order for calculations to be applied to them like reciprocating them
fit, cov = np.polyfit(x, y, 1, w=1/np.array(error), cov = True)

#calculating the error on the slope and intercept of the plot which is given by the square root of the diagonals of the covariance matrix respectively, since the covariance matrix
#diagonals represent variance for each value
sig_0 = np.sqrt(cov[0, 0])
sig_1 = np.sqrt(cov[1, 1])

#plotting a graph of redshift velocity against distance, and plotting a weighted linear line on top of the data points, adding axis labels, a title and a legend as well, the redshift
#velocity uncertainties are plotted as errorbars in the y direction on each data point
plt.plot(x, y, "x", mew = 10, ms = 1, color = "red", label = "data")
plt.errorbar(x, y, yerr = error, fmt = "*", color = "green", label = "error bar")
plt.plot(x, np.polyval(fit, x), label = "fit")
plt.xlabel("Distance (Mpc)")
plt.ylabel("Redshift velocity (km/s)")
plt.xticks()
plt.yticks()
plt.title("Redshift Velocity against Distance")
plt.legend()
plt.savefig("Redshift Velocity against Distance.png")
plt.show()

#outputting the values of the slope and intercept of the graph of velocity against distance and their associated uncertainties, Hubble's Constant is the gradient of this graph so its
#value has been calculated and so this value and its uncertainty is also outputted
print(f"Slope = {fit[0]:.0f} km/s/Mpc +/- {sig_0:.1f} km/s/Mpc")
print(f"Intercept = {fit[1]:.0f} km/s +/- {sig_1:.1f} km/s")
print(f"Therefore Hubble's Constant is {fit[0]:.0f} km/s/Mpc +/- {sig_0:.1f} km/s/Mpc")