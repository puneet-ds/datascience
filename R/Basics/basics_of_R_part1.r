# Basic Operations in R.

# Defining a variable.
x <- 6
# To get the values or print we need to simply type the variable name..
x

# Defining string variable.
my_text = "Hello World From R.."
# To get the values or print we need to simply type the variable name..
my_text

# Basic Arithmetic Operations. 
1 + 9 # 10
10 - 1 # 9
20/2 # 10
2.9/2 # 1.45
# Standard Maths order. 
1 + (6/2) #4
10 * (6-2) # 40
10 * 6 - 2 # 58

# Arithmetic operation using variables..
x = 10
y = 15
x + y #25
x - y # -5
x / y # 0.66667
# Operation on string variables.
my_text1 = "Hello"
my_text2 = "World!"
final_text = paste(my_text1, my_text2)
final_text
# Splitting the strings.
strsplit(final_text," ")
test = strsplit(final_text," ")
test
#=======================================================================================================================================
# Basic Data Type in R.
# (1). Vectors
# (2). Factors
# (3). Matrices
# (4). DataFrames
# (5). Lists
#========================================================================================================================================

# (1): Vectors:- A collection of values that all have the same data type. The
# elements of a vector are all numbers, giving a numeric vector, or
# all character values, giving a character vector. A vector can be
# used to represent a single variable in a data set.
numeric_vector = c(1,2,3,4,5)
# Printing vector created above.
numeric_vector
#===============================
character_vector = c("Hi","From","R")
# Printing character vector created above
character_vector
# Vectors can have similar data type. Trying to create a vector with different data type(numeric + character)
test_vector = c(1,2,"text") # The vector is created . However the type of vector is character eventhough it contains numeric.
test_vector
typeof(test_vector)
#=========================================================================================================================================
# (2) Factors:- A collection of values that all come from a fixed set of possible 
# values. A factor is similar to a vector, except that the values within
# a factor are limited to a fixed set of possible values
# Creating a vector and then converting it to factor.
vehicles = c("car","bike","truck","bus")
vehicles_factored = factor(vehicles)
vehicles_factored
# Another Example
designation = as.factor(c("Manager", "Developer","Administrator", "Operator"))
designation
#==========================================================================================================================================
# (3). Matrices:- A two- dimensional collection of values that all have the same type. The values are arranged in rows and columns.
# A matrix can be created directly using the matrix() function. The following code creates a matrix from 9 values, with 3 columns and
# three rows; the values are used column-first
matrix(1:9, ncol=3)
# By default the numbers are filled column wise in the matrix, here by the use of byrow=TRUE argument, we fill the values row wise
matrix(1:9,ncol=3,byrow=TRUE)
#==========================================================================================================================================
# The array data structure extends the idea of a matrix to more than two dimensions. For example, a three-dimensional array
# corresponds to a data cube.
# The array() function can be used to create an array. In the following
# code, a two-by-two-by-two, three-dimensional array is created.
array(1:8,dim=c(2,2,2))
#==========================================================================================================================================
# (4). DataFrame:- A collection of vectors that all have the same length. This is like a
# matrix, except that each column can contain a different data type. A
# data frame can be used to represent an entire data set.
ids = c(1,2,3,4)
owner = c("Ram","John","Mark","Shyaam")
vehicle_count = c(1,2,5,3)
# Creating Data Frame from all the above individual vectors.
vehicle_details = data.frame(ids,owner,vehicle_count)
vehicle_details
#==========================================================================================================================================
# (5). Lists: A collection of data structures. The components of a list can be 
# simply vectors similar to a data frame, but with each column allowed
# to have a different length. However, a list can also be a much more
# complicated structure. This is a very flexible data structure. Lists can
# be used to store any combination of data values together.

#Creating List output by getting the details for VehicleDetails data frame
#We are using a function name dimnames, which will return the
#properties of rows and columns for the given data set.

dimnames(vehicle_details) # The dataframe has 4 Rows & 3 Columns(1:- Represent Row, 2:- Represent Column.)
#==========================================================================================================================================
