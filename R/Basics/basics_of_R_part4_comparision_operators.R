#===============================================================================
#                             Comparison Operators
#===============================================================================
# Equality:
2 == 2 # Returns TRUE or FALSE
3 == 5 
"text" == "text" # TRUE
"text" == "TEXT" # FALSE
#===============================================================================
# In-equality:
3 != 3 # FALSE
2 != 1 # TRUE
"text" != "TEXT" # TRUE
#===============================================================================
# Greater or Equal:
2 >= 1 #TRUE
1 >= 2 #FALSE
#===============================================================================
# Less or Equal:
1 <= 2 # TRUE
2 <= 1 # FALSE
#===============================================================================
# AND:
2 == 2 & 3 == 3 #TRUE
2 == 1 & 3 == 3 #FALSE
"text" == "text" & 3 == 3 #TRUE
#===============================================================================
# OR:
1 == 2 | 3 == 3 #TRUE
"TEST" == "text" | 1 < 2 #TRUE
1 > 2 | 1 > 3 #FALSE
#===============================================================================
#                               Comparison on Vectors
#===============================================================================
comp_vec1 <- c(1,2,3,4)
comp_vec2 <- c(5,6,7,8)
# The below comparisons will return vector of logical values element wise.
comp_vec1 == comp_vec2 # FALSE FALSE FALSE FALSE
comp_vec1 < 2
# Comparison on a particular element using indexing.
comp_vec1[1] > 2
# ** Give attention to "Assignment Operator (<-)". If you are comparing against any negative number
# then 'spacing' is important otherwise R will take that as assignment operator.

test.var <- 1
# Lets say we want to compare that if test.var is less than -2 
test.var <- 2 # As there is no spacing between < and - .R will take this as assignment operator and will update the value to 2.
test.var
# The correct spacing is required.
test.var < -2 # This is correct syntax and the result will be either TRUE or FALSE . 
#================================================================================
