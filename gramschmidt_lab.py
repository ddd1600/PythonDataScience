#Gram-Schmidt Process Lab Quiz

import numpy as np
import numpy.linalg as la

verySmallNumber = 1e-14 # That's 1×10⁻¹⁴ = 0.00000000000001

# Our first function will perform the Gram-Schmidt procedure for 4 basis vectors.
# We'll take this list of vectors as the columns of a matrix, A.
# We'll then go through the vectors one at a time and set them to be orthogonal
# to all the vectors that came before it. Before normalising.
# Follow the instructions inside the function at each comment.
# You will be told where to add code to complete the function.
def gsBasis4(A) :
    B = np.array(A, dtype=np.float_) # Make B as a copy of A, since we're going to alter it's values.
    # The zeroth column is easy, since it has no other vectors to make it normal to.
    # All that needs to be done is to normalise it. I.e. divide by its modulus, or norm.
    B[:, 0] = B[:, 0] / la.norm(B[:, 0])
    # For the first column, we need to subtract any overlap with our new zeroth vector.
    B[:, 1] = B[:, 1] - B[:, 1] @ B[:, 0] * B[:, 0]
    # If there's anything left after that subtraction, then B[:, 1] is linearly independant of B[:, 0]
    # If this is the case, we can normalise it. Otherwise we'll set that vector to zero.
    if la.norm(B[:, 1]) > verySmallNumber :
        B[:, 1] = B[:, 1] / la.norm(B[:, 1])
    else :
        B[:, 1] = np.zeros_like(B[:, 1])
    # Now we need to repeat the process for column 2.
    # Insert two lines of code, the first to subtract the overlap with the zeroth vector,
    # and the second to subtract the overlap with the first.
    B[:,2] = B[:,2] - B[:,2] @ B[:,0] * B[:,0]
    B[:,2] = B[:,2] - B[:,2] @ B[:,1] * B[:,1]
    # Again we'll need to normalise our new vector.
    # Copy and adapt the normalisation fragment from above to column 2.
    if la.norm(B[:,2]) > verySmallNumber :
        B[:,2] = B[:,2] / la.norm(B[:,2])
    else :
        B[:,2] = np.zeros_like(B[:,2])
    
    # Finally, column three:
    # Insert code to subtract the overlap with the first three vectors.
    B[:,3] = B[:,3] - B[:,3] @ B[:,0] * B[:,0]
    B[:,3] = B[:,3] - B[:,3] @ B[:,1] * B[:,1]    
    B[:,3] = B[:,3] - B[:,3] @ B[:,2] * B[:,2]        
    # Now normalise if possible
    if la.norm(B[:,3]) > verySmallNumber :
        B[:,3] = B[:,3] / la.norm(B[:,3])
    else :
        B[:,] = np.zeros_like(B[:,3])
    
    
    
    # Finally, we return the result:
    return B
