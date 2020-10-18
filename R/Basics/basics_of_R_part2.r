# In this file we will see some more basic operations in R.
#========================================================================================================================================
# (1). Conditions: A condition is an expression that has a yes/no answer--for example,
# whether one data value is greater than, less than, or equal to another.
# The result of a condition is a logical value: either TRUE or FALSE.
# R has the standard operators for comparing values, plus
# operators for combining conditions
# ========================================================================================================================================
# (A). EQUALITY OPERATOR: "=="
2 == 2 # wHEN EXECUTED IN R it will return TRUE
2 == 1 # wHEN EXECUTED IN R it will return FALSE
# ========================================================================================================================================
# (B). Greater than (or equal to): ">" OR ">="
2 > 1 # wHEN EXECUTED IN R it will return TRUE
1 >=2 # wHEN EXECUTED IN R it will return FALSE
#=========================================================================================================================================
# (C). Less than (or equal to): "<" OR "<="
2 < 3  # wHEN EXECUTED IN R it will return TRUE
2 <= 1 # wHEN EXECUTED IN R it will return FALSE
#=========================================================================================================================================
# (D). Inequality: "!="
2 != 1 # wHEN EXECUTED IN R it will return TRUE
2 != 2 # wHEN EXECUTED IN R it will return FALSE
#=========================================================================================================================================
# (E). AND: "&"
(2 == 2 ) & (1 < 3) # wHEN EXECUTED IN R it will return TRUE as both conditions are satisfied.
(2 <= 1) & (3 == 3) # wHEN EXECUTED IN R it will return FALSE as one condition is FALSE.
#=========================================================================================================================================
# (F). OR: "|"
(2 == 2 ) | (1 < 3) # wHEN EXECUTED IN R it will return TRUE as one condition is TRUE
(2 <= 1) | (3 < 1) # wHEN EXECUTED IN R it will return FALSE as both conditions are FALSE

#=========================================================================================================================================
#                                            Functions                                                                                  
#=========================================================================================================================================

# The most common and most useful type of R expression is a
# function call. Function calls are very important because they are
# how we use R to perform any non-trivial task.
# A function call is essentially a complex instruction, and there are
# thousands of different R functions that perform different tasks. This
# section just looks at the basic structure of a function call; we will
# meet some important specific functions for data manipulation in
# later sections. A function call consists of the function name followed
# by, within parentheses and separated from each other by commas,
# expressions called arguments that provide necessary information for
# the function to perform its task.

x = 8
y = 10
sum(x,y) # Inbuilt function to perform addition.
max(x,y) # Getting maximum value among input values
min(x,y) 
sqrt(10)
test_numeric_vector = c(4,6,8)
sqrt(test_numeric_vector) # Returns square root of each element. [2.000000 2.449490 2.828427]
#=========================================================================================================================================
#                                                   Symbols and assignment
#=========================================================================================================================================
#Anything that we type that starts with a letter, and which is not one
#of the special R keywords, is interpreted by R as a symbol (or name). A symbol is a label for an object that is currently stored in
# RAM. When R encounters a symbol, it extracts from memory the
# value that has been stored with that label.
# R automatically loads some predefined values into memory, with
# associated symbols. One ex-ample is the predefined symbol pi,
# which is a label for the mathematical constant π.
pi # Returns 3.141593
#=========================================================================================================================================

# Keywords:
# Some symbols are used to represent special values. These
# predefined symbols cannot be re-assigned.
# "NA" : This symbol represents a missing or unknown value
# "Inf" : This symbol is used to represent infinity (as the result of an arithmetic expression).
1/0
# "NaN": This symbol is used to represent an arithmetic result that is undefined (Not A Number).
0/0
# "NULL": This symbol is used to represent an empty result. Some R functions do not produce a result, so they return NULL
# TRUE and FALSE: These symbols represent the logical values “true” and “false”. The result of a condition is a logical value
3 == 3
3 >= 5
#==========================================================================================================================================
#                                                          Naming variables
#==========================================================================================================================================
# When writing R code, because we are constantly assigning
# intermediate values to symbols, we are forced to come up with lots
# of different symbol names. It is important that we choose sensible
# symbol names for several reasons:
#   Good symbol names are a form of documentation in themselves. A name
# like dateOfBirth tells the reader a lot more about what value has been
# assigned to the symbol than a name like d, or dob, or even date.
# Short or convenient symbol names, such as x, or xx, or xxx should
# be avoided because it too easy to create errors by reusing the
# same name for two different purposes.
# Anyone with children will know how difficult it can be to come up
# with even one good name, let alone a constant supply. However,
# unlike children, symbols usually have a specific purpose, so the
# symbol name naturally arises from a description of that purpose.
# A good symbol name should fully and accurately represent
# the information that has been assigned to that symbol.
# One problem that arises is that a good symbol name is often a
# combination of two or more words. One approach to making such
# symbols readable is to use a mixture of lowercase and up-percase
# letters when typing the name; treat the symbol name like a
# sentence and start each new word with a capital letter (e.g.,dateOfBirth). This naming mechanism is called “camelCase” 
# (the uppercase letters form humps like the back of a camel).