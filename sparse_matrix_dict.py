import numpy as np
import math

NEG_INF = -100000

class SparseMatrixDict(object):
    """Utility class for Sparse Matrix, implemented by a dictionary. Assume there will not be much col. Matrix is for log probability"""
    def __init__(self, shape, log_result=False, laplace_smoothing=False):
        super(SparseMatrixDict, self).__init__()
        self.height, self.width = shape
        self.laplace_smoothing = laplace_smoothing  #start with all 1's in the matrix
        self.log_result = log_result    #decide if the getitem will return a log prob, or raw accumulations
        self.matrix_dict = {}
        
        self.row_sums = np.zeros(self.height)   #there are height num of rows, assume not many cols



    def __getitem__(self, pos):
        x,y = pos   #hack!
        
        if self.matrix_dict.has_key(pos):
            if self.log_result:

                if self.laplace_smoothing:
                    return math.log( (1.0+self.matrix_dict[pos])/(self.row_sums[x]+1.0*(self.width)))
                else:
                    return math.log(self.matrix_dict[pos] / self.row_sums[x])
            
            else:
                return self.matrix_dict[pos]

        else:
            if self.log_result:

                if self.laplace_smoothing:
                    return math.log(1.0/(self.row_sums[x] + 1.0*(self.width)))

                else:
                    return NEG_INF    #????

            else:
                return 0.0


    def __setitem__(self, pos, value):
        x,y = pos

        # Update sums
        self.row_sums[x] += float(value)
        self.matrix_dict[pos] = float(value)

    # Change the state of sparse matrix to return log likelihood
    def change_state(self):
        self.log_result = not self.log_result


    def print_stats(self):
        print 'Total number of non-zeros: ', len(self.matrix_dict)
        print 'Sums for each row: ', self.row_sums


if __name__ == '__main__':
    sm1 = SparseMatrixDict((3,4), log_result=False)
    sm1[0,0] = 0.0
    sm1[0,1] = 1.0
    sm1[0,2] = 2.0
    sm1[0,2] = sm1[0,2] + 1.0

    print sm1[1,1]

    print sm1.matrix_dict