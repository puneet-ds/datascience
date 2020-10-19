#===============================================================================
#                                Indexing
#===============================================================================
# Indexing helps us to extract items based on their index position.
# In R the index starts from 1.
index_example_vector <- c(1,2,3,4,5,6,7,8,9,10)
# Assume we want item at 7th index.
index_example_vector[7]
index_example_vector[1]
index_example_vector[-1] # This will give all the elements except the element at index 1.
# Lets say we want to extract elements which are at index 2 & 4  and 2 ,4 & 7 only.
index_example_vector[c(2,4)]
index_example_vector[c(2,4,7)]

# Indexing with character vectors
char_vector <- c("Hello","From", "R")
char_vector[c(1,3)]

# Updating value of a element by using index.
index_example_vector[4] <- 20

#===============================================================================
#                               Slicing
#===============================================================================
# If we want to take a 'slice' of existing vector.

#index_example_vector["Start_index":"End_index"]


index_example_vector[1:3]
# It will give the elements from index 1 to index 3. Index in R starts at 1 and
# upper bound is included.
index_example_vector[4:8]
# The below will add 10 to elements from index 4 till 8.
index_example_vector[4:8] + 10

# Lets create a named vector.
named.vector1 <- c(10,20,30,40)
names(named.vector1) <- c("TEN","TWENTY","THIRTY","FOURTY")
named.vector1["TEN"]
named.vector1[c("TEN","THIRTY")]

# Slicing the vector with a condition (Return elements which are greater than 10)
named.vector1[named.vector1 > 10]
