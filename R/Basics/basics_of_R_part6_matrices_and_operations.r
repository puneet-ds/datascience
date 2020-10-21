#============================================================================================================
#                                 Matrix
#============================================================================================================

# A two- dimensional collection of values that all have the same
# type. The values are arranged in rows and columns. Matrices are nothing but combination of vectors.
# Matrices are used to represent data in 2D (rows & colums.)

# You can create a 2D matrix by passing a single vector.The below command will create
# a 2D matrix (12 Rows & 1 Column) because by default the data is filled column wise in R.

mvector <- 1:12
my.matrix <- matrix(mvector)
print(my.matrix)
#===============================================================================
# The below command will create a 2D matrix but as we are passing byrow to TRUE
# and number of rows to 1. This time it will have 1 row & 12 columns.
my.matrix <- matrix(mvector, byrow=T, nrow=1)
print(my.matrix)
#============================================================================================================

# If you closely look at the output of above created matrices you will notice that 
# that the ROWS are having [1,] & columns are having [,1] it means in R matrices
# the first place in [1,1] notation means ROW and second place(after comma) represent
# column. This is usefull for slicing/extracting value from matrices based on ROW 
# & column.
# Lets create a new matrix to demonstrate it .(3  by 3 Matrix)
example.matrix <- matrix(1:9, byrow=T, nrow=3)
print(example.matrix)
# We want to extract the item in row 3 and 2 column(which is 8 in above example.matrix)
example.matrix[3,2] # This will return 8.
test.var <- example.matrix[3,2]
print(test.var)
#============================================================================================================
#                                                Arithmetic on Matrix
#============================================================================================================

test.mat <- matrix(1:9, byrow=T, nrow=3)
test.mat

# Adding scalar
test.mat + 2 # will add 2 to every element of matrix.
test.mat * 2 # will multiply each element by 2.This is element wise multiplication
test.mat * test.mat # This is element wise multiplication
test.mat/test.mat

# Matrix to Matrix multiplication. This is as per algebra standard.
test.mat %*% test.mat

#============================================================================================================
#                                             Giving names to Rows and Columns
#============================================================================================================

# If we want to change the row names and column names of our matrix there are functions available.
# Firstly we need to create vector having values which we want for our rows and columns 

names.of.row <- c("Petrol","Diesel","Gasoline")
names.of.columns <- c("Sun","Mon","Tues")
colnames(example.matrix) <- names.of.columns # To update the column names.
rownames(example.matrix) <- names.of.row # To update the row names
print(example.matrix)
# Output will be like below
#          Sun Mon Tues
#Petrol     1   2    3
#Diesel     4   5    6
#Gasoline   7   8    9

# Now we can extract the values by using the newly added row & column names.
# Remember [ROWS,COLUMN] NOTATION to filtering.

# Extract value of Petrol on Sunday
example.matrix["Petrol","Sun"] # Would return 1
# Extract value of Petrol for all days.
example.matrix["Petrol",] # We are interested in all columns(Days) but only row that contains "Petrol" data.

# Extract value of Petrol & Diesel for Sun & Mon.
example.matrix[c("Petrol","Diesel"),c("Sun","Mon")]

#============================================================================================================
#                                              Adding Rows and Columns
#============================================================================================================

# If we want to add a new ROW to a matrix. As the matrix can be seen as collection of vectors of same data type
# and length. We need to have a vector created first for the data which we need to add as ROW.
CNG <- c(10,11,12)
example.matrix <- rbind(example.matrix,CNG)
example.matrix

# Adding columns to existing matrix.
# Lets say we want to add a new column in above matrix which will store the mean of each row. Firstly we need
# to create a vector which will calculate the mean of each row and then we will add that vector as a new column.
AVG <- rowMeans(example.matrix)
AVG
example.matrix <- cbind(example.matrix,AVG)
example.matrix

