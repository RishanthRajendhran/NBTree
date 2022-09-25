from helper.imports.packageImports import np, csv 
#File operations 

#writeToCsvFile
#Input - x : Data matrix
#        x_path : Path to location of the csv file to be written into
#Output - NILL
#What it does - This function is used to write a data matrix row-by-row
#               into a csv file
#Assumption - NILL
def writeToCsvFile(x, x_path):
    if x.ndim == 1:
        x = x.reshape(x.shape[0], 1)
    with open(x_path, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(x)
        csv_file.close()