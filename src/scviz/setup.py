# functions to add here
# check for installed packages?

# consider inbuilt error handling
class PairLengthError(Exception):
    """Raised when a pair in test_pairs does not contain exactly two groups"""
    pass

class GroupLengthError(Exception):
    """Raised when a group in a pair in test_pairs does not match the length of test_variables"""
    pass