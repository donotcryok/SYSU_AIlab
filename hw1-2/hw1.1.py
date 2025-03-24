def binarySearch(nums, target):
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        mid_val = nums[mid]
        
        if mid_val == target:
            return mid
        elif mid_val < target:
            left = mid + 1
        else:
            right = mid - 1
            
    return -1

nums = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
target = 23
print(binarySearch(nums, target))  

target = 3
print(binarySearch(nums, target))  