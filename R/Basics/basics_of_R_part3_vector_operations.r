#=============================================================================== 
#                              Vector Operations
#===============================================================================

# Creating Vectors:
numeric.vector1 <- c(1,2,3,4)
numeric.vector2 <- c(5,6,7,8)
# Adding two vectors: It performs element wise addition in case both vectors
# are of same length.
final.vector <- numeric.vector1 + numeric.vector2
print(final.vector) # 6  8 10 12
# Adding two vectors having different length. By default it recycles'short'
# vector to match the length of large vector.
vec1 <- c(1,2,3)
vec2 <- c(4,5,6,7,8,9)
rvec <- vec1 + vec2
print(rvec) # Gives output as 5 7 9 8 10 12 .
# The first 3 elements from each vectors are added and then the vec1 is recycled
# and added to remaining 3 elements of vec2. In background the 'short' vector
# was extended to match the length of 'long' vector by repeating the elements.
# vec1 <- c(1,2,3,1,2,3)
#==============================================================================
#                           Airthmetic Operations
#==============================================================================
division.example <- numeric.vector2 / numeric.vector1
print(division.example)
#==============================================================================
multi.example <- numeric.vector1 * numeric.vector2
print(multi.example)
#==============================================================================
substract.example <- numeric.vector2 - numeric.vector1
print(substract.example)
#==============================================================================
#                         Built-in functions
#==============================================================================
sqrt(vec2) # Gives square root of each element in the vector.
max(vec1) # Gives Maximum value in a specified vector.
sum(vec1) # Adding elements of a given vector
mean(vec2)
median(vec1)
prod(vec2) # Product of each element of a given vector.
#==============================================================================
#                           'Character' Vectors
#==============================================================================
text.vector1 <- c("Hello","From" ,"R")
text.vector2 <- c("Hi","From","Me")
combined.vector <- paste(text.vector1,text.vector2)
print(combined.vector) # Output : "Hello Hi"  "From From" "R Me"
# Slicing values using indexing. **Indexing in R starts from 1**
text.vector1[1]
text.vector1[1:3] # Output : "Hello" "From"  "R" **Upper bound is included in R**

#===============================================================================
#                          Naming Vectors
#===============================================================================
 # Lets say we have created a vector for storing temperature of 7 days.
temperature <- c(10,20,30,40,50,60,70)
# By creating vector above it would be difficult to identify the temperature on any particular day.
# Also by above it is not clear that first entry is for Monday or Sunday..to avoid such issues
# we can provide names to our vector

# Lets create a days vector first.
days_of_week <- c("Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday")

# Now lets give the names to our temperature vector. By using the builtin function names()
names(temperature) <- days_of_week
# Now lets print the modified 'temperature' vector.
print(temperature)

# The out put will be
# Sunday    Monday   Tuesday Wednesday  Thursday    Friday  Saturday 
# 10        20        30        40        50        60        70 

# Now it would be easy for us to extract the values from this vector based on 'day' by using slicing..
temperature["Sunday"]
temperature["Monday"]
temperature_sun <- temperature["Sunday"]
print(temperature_sun)
