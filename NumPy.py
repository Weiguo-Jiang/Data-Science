import numpy as np

def createArray():
    # Create array from Python list
    npArray = np.array([1, 2, 3])
    print(npArray)

    # NumPy will upcast if possible, in this case
    # it upcasts to float64
    npArray = np.array([1.2, 2, 3])
    print(npArray)

    # You can also specify the data type of the array
    npArray = np.array([1, 2, 3], dtype=np.float64)
    print(npArray)

    # Multidimensional array
    npArray = np.array([[1, 2, 3], [4, 5, 6]])
    print(npArray)

    # Pre-filled arrays
    npArray = np.zeros(10, dtype=np.int64)
    print(npArray)

    npArray = np.ones((3, 5), dtype=np.float64)
    print(npArray)

    npArray = np.full((3, 5), 3.14)
    print(npArray)

    # Linear sequence with stepsize specified
    npArray = np.arange(0, 20, 2)
    print(npArray)

    # Linear space with value evenly distributed
    npArray = np.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1]
    print(npArray)

    # 3*3 matrix with random values in [0, 1]
    npArray = np.random.random((3, 3))
    print(npArray)

    # 3*3 matrix with random values from normal distribution
    # with mean 0 and std 1
    npArray = np.random.normal(0, 1, (3, 3))
    print(npArray)

    # 3*3 matrix with random integers in the interval [0, 10)
    npArray = np.random.randint(0, 10, (3, 3))
    print(npArray)

    # 3*3 identity matrix
    npArray = np.eye(3)
    print(npArray)


def arrayBasics():
    # set seed for reproducibility
    np.random.seed(0)

    # array, (row)
    npArray = np.random.randint(10, size=6)
    print(npArray)

    # 2D array, (row, column)
    npArray2D = np.random.randint(10, size=(3, 4))
    print(npArray2D)

    # 3D array, (# of matrices, row, column)
    npArray3D = np.random.randint(10, size=(3, 4, 5))
    print(npArray3D)

    # ndim for number of matrices
    # shape for dimension of each matrix
    # size for total number of elements
    print("ndim: ", npArray3D.ndim)
    print("shape: ", npArray3D.shape)
    print("size: ", npArray3D.size)

    # dtype for data type of array
    print("dtype: ", npArray3D.dtype)

    # itemsize for size of each element in bytes
    # nbytes for total size of array in bytes
    print("itemsize: ", npArray3D.itemsize, "bytes")
    print("nbytes: ", npArray3D.nbytes, "bytes")

    # Access elements via index
    npArray = np.array([[1, 2, 3], [4, 5, 6]])

    # value will be truncated to int
    npArray[0, 0] = 3.14
    print(npArray)

    # slicing array has the syntax:
    # array[start:stop:step]
    # start is inclusive, stop is exclusive
    # by default we have start=0, stop=size of dimension, step=1
    npArray = np.arange(10)
    print(npArray)

    # every other element, starting from index 1
    print(npArray[1::2])

    # multi-dimensional array slicing
    npArray = np.random.randint(10, size=(3, 4))
    print(npArray)

    print(npArray[:2, :3])
    print(npArray[::-1, ::-1])

    # access a single column
    print(npArray[:, 0])

    # in Python when we slice we create a copy,
    # but in NumPy we create a view, i.e., a reference
    # to the original array
    npArray = np.arange(10)
    print(npArray)
    npArraySlice = npArray[:5]
    npArraySlice[0] = 11
    print(npArray)

    # if we want to create a copy of the array
    # we can use the copy() method
    npArray = np.arange(10)
    print(npArray)
    npArraySlice = npArray[:5].copy()
    npArraySlice[0] = 11
    print(npArray)

    # Reshaping
    # array will be filled with numbers in the
    # range, note that size of initial array
    # must match the size of the reshaped array
    # reshape() specifies the dimensions
    npArray = np.arange(1, 10).reshape((3, 3))
    print(npArray)

    # Concatenation
    npArray = np.array([[1, 2, 3], [4, 5, 6]])
    npArray = np.concatenate([npArray, npArray, npArray], axis=1)
    print(npArray)

    # Concatenation of arrays of different dimensions
    npArray = np.array([1, 2, 3])
    npArray2 = np.array([[9, 8, 7], [6, 5, 4]])
    # stack vertically / row-wise
    print(np.vstack([npArray, npArray2]))
    # stack horizontally / column-wise
    print(np.hstack([npArray2, [[99], [99]]]))

    # Splitting
    npArray = np.arange(10)
    # [3, 7] means we break at index 3 and 7
    npArray1, npArray2, npArray3 = np.split(npArray, [3, 7])
    print(npArray1, npArray2, npArray3)

    # split vertically / row-wise
    npArray = np.arange(16).reshape((4, 4))
    npArray1, npArray2 = np.vsplit(npArray, [2])
    print(npArray1)
    print(npArray2)

    # split horizontally / column-wise
    npArray = np.arange(16).reshape((4, 4))
    npArray1, npArray2 = np.hsplit(npArray, [2])
    print(npArray1)
    print(npArray2)


def UFuncs():
    # A universal function (or ufunc for short) is a function that
    # operates on ndarrays in an element-by-element fashion, supporting
    # array broadcasting, type casting, and several other standard
    # features. That is, an ufunc is a “vectorized” wrapper for a function
    # that takes a fixed number of specific inputs and produces a fixed
    # number of specific outputs.

    # Element-wise array arithmetic
    npArray = np.arange(4)
    print(npArray)

    # all the ops are wrappers of functions
    print(npArray + 5)  # == np.add(npArray, 5)
    print(npArray - 5)  # == np.subtract(npArray, 5)
    print(npArray * 5)  # == np.multiply(npArray, 5)
    print(npArray / 5)  # == np.divide(npArray, 5)
    print(npArray // 5) # == np.floor_divide(npArray, 5)
    print(npArray ** 2) # == np.power(npArray, 2)
    print(npArray % 5)  # == np.mod(npArray, 5)
    print(abs(npArray)) # == np.absolute(npArray)

    # for complex nums, abs() returns magnitude
    npArray = np.array([-2 - 1j, -1 + 1j, 1 + 1j, 2 - 1j])
    print(abs(npArray))

    # trigs
    npArray = np.linspace(0, np.pi, 3)  # [0, pi/2, pi]
    print(np.sin(npArray))
    print(np.cos(npArray))
    print(np.tan(npArray))

    # inverse trigs
    npArray = [-1, 0, 1]
    print(np.arcsin(npArray))
    print(np.arccos(npArray))
    print(np.arctan(npArray))

    # exp & log
    npArray = [1, 2, 3]
    # only have exp() & exp2(), no exp3()
    # we use power() instead
    print(np.exp(npArray))
    print(np.power(3, npArray))

    # log(), log2(), log10()
    print(np.log(npArray))
    print(np.log2(npArray))
    print(np.log10(npArray))

    # more specialized uFuncs can be found in SciPy,
    # SciPy means Scientific Python and is built upon NumPy,
    # NumPy means Numerical Python

    # Aggregates
    npArray = np.arange(1, 6)
    print(np.add.reduce(npArray))   # == np.sum(npArray)
    print(np.multiply.reduce(npArray))  # == np.prod(npArray)
    print(np.add.accumulate(npArray))   # == np.cumsum(npArray)
    print(np.multiply.accumulate(npArray))  # == np.cumprod(npArray)


def aggregations():
    npArray = np.random.random(100)
    # np.sum() is faster than Python's sum()
    print(np.sum(npArray)) # == npArray.sum()

    # min() & max()
    print(np.min(npArray)) # == npArray.min()
    print(np.max(npArray))  # == npArray.max()

    # multi-dimensional arrays aggregation
    npArray = np.random.random((3, 4))
    # sum all elements
    print(np.sum(npArray))

    # The axis keyword specifies the dimension of the array that will be collapsed,
    # rather than the dimension that will be returned. So specifying axis=0 means
    # that the first axis will be collapsed: for two-dimensional arrays, this means
    # that values within each column will be aggregated.
    print(np.min(npArray, axis=0))  # min of each column
    print(np.max(npArray, axis=1))  # max of each row


def broadcasting():
    # Broadcasting rules:
    # 1. If the two arrays differ in their number of dimensions, the shape of the
    #    one with fewer dimensions is padded with ones on its leading (left) side.
    # 2. If the shape of the two arrays does not match in any dimension, the array
    #    with shape equal to 1 in that dimension is stretched to match the other shape.
    # 3. If in any dimension the sizes disagree and neither is equal to 1, an error is raised.

    # Naive example
    npArray = np.arange(3)
    print(npArray+3)    # 3 is reshaped and broadcasted

    # Example 1
    M = np.ones((2, 3)) # M.shape = (2, 3)
    a = np.arange(3)   # a.shape = (3,)
    # Based on rule 1 we pad the shape of a with ones: a.shape -> (1, 3)
    # Based on rule 2 we stretch this axis to match M's shape: a.shape -> (2, 3)
    # Result: M.shape -> (2, 3), a.shape -> (2, 3)
    print(M + a)

    # Example 2
    a = np.arange(3).reshape((3, 1)) # a.shape = (3, 1)
    b = np.arange(3)   # b.shape = (3,)
    # Based on rule 1 we pad the shape of b with ones: b.shape -> (1, 3)
    # Based on rule 2 we stretch this axis to match a's shape: b.shape -> (3, 3)
    # Result: a.shape -> (3, 1), b.shape -> (3, 3)
    # Based on rule 2 we stretch the first axis of a to match b's shape: a.shape -> (3, 3)
    # Result: a.shape -> (3, 3), b.shape -> (3, 3)
    print(a + b)

    # Example 3
    M = np.ones((3, 2)) # M.shape = (3, 2)
    a = np.arange(3)   # a.shape = (3,)
    # Based on rule 1 we pad the shape of a with ones: a.shape -> (1, 3)
    # Based on rule 2 we stretch this axis to match M's shape: a.shape -> (3, 3)
    # Result: M.shape -> (3, 2), a.shape -> (3, 3)
    # We don't have any 1s and we still have a mismatch, so in this case we get an error:
    try:
        print(M + a)
    except ValueError:
        print("ValueError: operands could not be broadcast together with shapes (3,2) (3,)")


def comparisons():
    npArray = np.arange(5)
    # internally, comparisons are done using uFuncs
    print(npArray < 3)  # == np.less(npArray, 3)
                        # [True, True, True, False, False]

    # 2D array
    npArray = np.random.randint(10, size=(3, 4))
    print(npArray)
    # count # of True values
    print(np.count_nonzero(npArray < 6))
    # another way is np.sum where False is 0 and True is 1
    print(np.sum(npArray < 6))
    # check if any or all values are True
    print(np.any(npArray > 8))
    print(np.all(npArray < 10))
    # np.all() & np.any() can be used along specific axes
    print(np.any(npArray > 8, axis=1))

    # boolean operators
    print((npArray > 0) & (npArray < 5))    # == np.bitwise_and(npArray > 0, npArray < 5)

    # masking
    print(npArray[npArray < 5]) # prints all values < 5


def fancyIndexing():
    npArray = np.random.randint(0, 100, 10)
    print(npArray)
    ind = [3, 7, 4]
    print(npArray[ind]) # prints values at indices 3, 7, 4

    # fancy indexing

    # the shape of the result reflects the shape of the index arrays
    # rather than the shape of the array being indexed:
    ind = np.array([[3, 7], [4, 5]])
    print(npArray[ind])

    npArray = np.arange(12).reshape((3, 4))
    row = np.array([0, 1, 2])
    col = np.array([2, 1, 3])
    print(npArray[row, col])    # prints values at (0, 2), (1, 1), (2, 3)

    # pairing follows broadcasting rules
    # row is 3x1, col is 1x3, so the result is 3x3
    print(npArray[row[:, np.newaxis], col])

    print(npArray[2, [2, 0, 1]])    # prints values at (2, 2), (2, 0), (2, 1), 2 == [2]
    print(npArray[1:, [2, 0, 1]])   # prints values at (1, 2), (1, 0), (1, 1), (2, 2), (2, 0), (2, 1)

    # fancy indexing with masking
    mask = np.array([1, 0, 1, 0], dtype=bool)
    print(npArray[row[:, np.newaxis], mask]) # print from columns 0 & 2

    # we can also modify values using fancy indexing
    npArray = np.arange(5)
    i = np.array([0, 1])
    npArray[i] += 99
    print(npArray)  # [99, 100, 2, 3, 4]


def sorting():
    npArray = np.random.randint(0, 100, 10)
    print(np.sort(npArray)) # returns a sorted copy of the array
    # npArray.sort() sorts the array in-place

    # argsort() returns the indices of the sorted elemente
    npArray = np.random.randint(0, 100, 10)
    # The first element of this result gives the index of the smallest element, the second
    # value gives the index of the second smallest, and so on.
    ind = np.argsort(npArray)
    print(ind)
    # we can use these indices to reconstruct the sorted array
    print(npArray[ind])

    # sorting along rows or columns
    npArray = np.random.randint(0, 100, (3, 5))
    print(np.sort(npArray, axis=0)) # sort each column of npArray
    print(np.sort(npArray, axis=1)) # sort each row of npArray

    # partial sorting
    npArray = np.random.randint(0, 100, 10)
    print(np.partition(npArray, 3)) # first 3 elements are smallest,
                                    # however they are not sorted

    npArray = np.random.randint(0, 100, (3, 5))
    print(np.partition(npArray, 2, axis=1)) # first 2 elements of each row are smallest,
                                            # however they are not sorted


def structuredArrays():
    name = ['Alice', 'Bob', 'Cathy', 'Doug']
    age = [25, 45, 37, 19]
    weight = [55.0, 85.5, 68.0, 61.5]

    # compound data types for structured arrays
    data = np.zeros(4, dtype={'names': ('name', 'age', 'weight'),
                                'formats': ('U10', 'i4', 'f8')})
    # U10 means Unicode string of maximum length 10
    # i4 means 4-byte (i.e., 32 bit) integer
    # f8 means 8-byte (i.e., 64 bit) float

    data['name'] = name
    data['age'] = age
    data['weight'] = weight
    print(data)

    # get all names
    print(data['name'])
    # get first row of data
    print(data[0])
    # get the name from the last row
    print(data[-1]['name'])

    # get names where age is under 30
    print(data[data['age'] < 30]['name'])

    # alternative to U10, i4 and f8 is to use the Python types directly
    data = np.zeros(4, dtype={'names': ('name', 'age', 'weight'),
                                'formats': ((np.str_, 10), int, np.float32)})


def main():
    # NOTE: NumPy arrays contains homogenous datatypes.
    #       If you try to initialize an array with different
    #       datatypes, NumPy cast all to one single
    #       datatype if possible.
    # https://stackoverflow.com/questions/49751000/how-does-numpy-determine-the-array-data-type-when-it-contains-multiple-dtypes

    # createArray()
    # arrayBasics()
    # UFuncs()
    # aggregations()
    # broadcasting()
    # comparisons()
    # fancyIndexing()
    # sorting()
    # structuredArrays()
    pass

if __name__ == "__main__":
    main()