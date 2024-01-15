import pandas as pd 

# read in weather measurements from file "weatherdata.csv" 

data = pd.read_csv('Assignment_MLBasics_WeatherData.csv')


# we consider each temp. measurement (=a row in dataframe data) as a separate data point 

#  print out first data point (first row)

print("first data point : \n", data.iloc[0])
print("\n ***** \n")
# here is another data point 

print("another data point : \n",data.iloc[13])


# as the label (quantity of interest) of a data point, we choose the temperature 
print("\n ***** \n")

print("label of first data point : ", data["temp"].iloc[0])
print("label of another data point : ", data["temp"].iloc[13])

# as feeatures of a data point we use lat,lon, year, mon, day, hour, minute (as float values)



