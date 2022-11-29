import sys

#hamed@talebian@tuni.fi 
#Read command line arguments and convert to a list of integers
arr = sys.argv[1].split(',')
my_numbers = [None]*len(arr)
for idx, arr_val in enumerate(arr):
    my_numbers[idx] = int(arr_val)
print(f'Before sorting {my_numbers}')



# My sorting code (Hoare sorting method - modified from this source (copy with pride!) 
# https://towardsdatascience.com/an-overview-of-quicksort-algorithm-b9144e314a72)

def find_pviot_index(A,start,end):
    """a function that define the pivot indec in the array and increment/swap the lower bound in each itteration
        input: array, int: 0 index of array, int: end index of array 
        return: int, pivot_index for next iteration"""
    pivot=A[end]
    p_index=start
    for iter in range(start,end):
        if A[iter] <= pivot:
            A[p_index],A[iter]=A[iter],A[p_index]
            p_index+=1
    A[p_index],A[end]=pivot,A[p_index]
    return p_index     

#main sorting function
def quick_sort(A,start,end):
    """a function that implement the Hoare sortin algorithm, itteratively
        input: array, input numbers, int: 0 index, int: end-index
        return: none (the sorted array automatically return to the main sort func.""" 
    
    if start < end:
        pivot_index=find_pviot_index(A,start,end)
        quick_sort(A,start,pivot_index-1)
        quick_sort(A,pivot_index+1,end)

def sort(array):
    """a function that initiate the Hare sorting algorithm
        param: list, input array by terminal
        return: list, sored list 
    """

    quick_sort(array, 0, len(array) - 1)
    return array

output = sort(my_numbers)
print(f'after sorting {output}')