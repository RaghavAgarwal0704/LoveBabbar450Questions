## 1. Find first and last positions of an element in a sorted array

naive: o(n)

o(log n):
```cpp
int first(int arr[], int x, int n)
{
    int low = 0, high = n - 1, res = -1;
    while (low <= high) {
        // Normal Binary Search Logic
        int mid = (low + high) / 2;
        if (arr[mid] > x)
            high = mid - 1;
        else if (arr[mid] < x)
            low = mid + 1;
 
        // If arr[mid] is same as x, we
        // update res and move to the left
        // half.
        else {
            res = mid;
            high = mid - 1;
        }
    }
    return res;
}
 
/* if x is present in arr[] then returns the index of
LAST occurrence of x in arr[0..n-1], otherwise
returns -1 */
int last(int arr[], int x, int n)
{
    int low = 0, high = n - 1, res = -1;
    while (low <= high) {
        // Normal Binary Search Logic
        int mid = (low + high) / 2;
        if (arr[mid] > x)
            high = mid - 1;
        else if (arr[mid] < x)
            low = mid + 1;
 
        // If arr[mid] is same as x, we
        // update res and move to the right
        // half.
        else {
            res = mid;
            low = mid + 1;
        }
    }
    return res;
}
```

---------------------------------------------------------------
## 2. Find a Fixed Point (Value equal to index) in a given array
```cpp
vector<int> valueEqualToIndex(int arr[], int n) {
    vector<int>vec;
    for(int i=0;i<n;i++){
        if(arr[i]==i+1)vec.push_back(i+1);
    }
    return vec;
}
```

---------------------------------------------------------------
## 3. Search in a rotated sorted array
 
naive: linear search

efficient: binary search


---------------------------------------------------------------
## 4. square root of an integer

---------------------------------------------------------------
## 5. Maximum and minimum of an array using minimum number of comparisons

---------------------------------------------------------------
## 6. Optimum location of point to minimize total distance

---------------------------------------------------------------
## 7. Find the repeating and the missing

https://www.geeksforgeeks.org/find-a-repeating-and-a-missing-number/

---------------------------------------------------------------
## 8. find majority element
o(n^2) o(1) => o(nlogn) o(1) => o(n) o(n) 
 
best => o(n) o(1) ---Moore's Voting algorithm-----https://www.youtube.com/watch?v=kRKZ0s2TjJw
```cpp
int findCandidate(int a[], int size)
{
    int maj_index = 0, count = 1;
    for (int i = 1; i < size; i++) {
        if (a[maj_index] == a[i])
            count++;
        else
            count--;
        if (count == 0) {
            maj_index = i;
            count = 1;
        }
    }
    return a[maj_index];
}

/* Function to check if the candidate
   occurs more than n/2 times */
bool isMajority(int a[], int size, int cand)
{
    int count = 0;
    for (int i = 0; i < size; i++)

        if (a[i] == cand)
            count++;

    if (count > size / 2)
        return 1;

    else
        return 0;
}

/* Function to print Majority Element */
void printMajority(int a[], int size)
{
    /* Find the candidate for Majority*/
    int cand = findCandidate(a, size);

    /* Print the candidate if it is Majority*/
    if (isMajority(a, size, cand))
        cout << " " << cand << " ";

    else
        cout << "No Majority Element";
}
``` 

---------------------------------------------------------------
## 9. Searching in an array where adjacent differ by at most k



---------------------------------------------------------------
## 10. find a pair with a given difference

naive: o(n^2)
efficient: o(nlogn)
```cpp
bool findPair(int arr[], int size, int n){
    sort(arr,arr+size);
    int i=0,j=1;
    while(i<size && j<size){
        if(arr[j]-arr[i]==n)return true;
        else if(arr[j]-arr[i]<n)j++;
        else i++;
    }
    return false;
}
```

---------------------------------------------------------------
## 11. find four elements that sum to a given value

```cpp
vector<vector<int>> fourSum(vector<int> &arr, int k) {
    sort(arr.begin(), arr.end());
    int n = arr.size();
    vector<vector<int>> ans;

    for (int i = 0; i < n - 2; i++) {
        for (int j = i + 1; j < n - 1; j++) {
            int reqd = k - arr[i] - arr[j];
            int l = j + 1, r = n - 1;
            while (l < r) {
                if (arr[l] + arr[r] == reqd) ans.push_back({arr[i], arr[j], arr[l], arr[r]});
                if (arr[l] + arr[r] < reqd)
                    l++;
                else
                    r--;
            }
        }
    }
    set<vector<int>> s(ans.begin(), ans.end());
    return vector<vector<int>>(s.begin(), s.end());
}
```

---------------------------------------------------------------
## 12. maximum sum such that no 2 elements are adjacent

---------------------------------------------------------------
## 13. Count triplet with sum smaller than a given value

---------------------------------------------------------------
## 14. merge 2 sorted arrays

Time : O(n*m); Space : O(1)
```cpp
void merge(int arr1[], int arr2[], int n, int m) {
    
    
    int i = 0, j = 0;
    for (i = m - 1; i >= 0; i--) {
        int k = upper_bound(arr1, arr1 + n, arr2[i]) - arr1;
        if (k < n) {
            int temp = arr1[n - 1];
            for (j = n - 1; j > k; j--) arr1[j] = arr1[j - 1];
            arr1[k] = arr2[i];
            arr2[i] = temp;
        }
    }
}
```

Time : O(nlogn + mlogm) 
```cpp
void merge(int arr1[], int arr2[], int n, int m) {
    for (int i = 0; i < min(n, m); i++)
        if (arr1[n - i - 1] > arr2[i]) swap(arr1[n - i - 1], arr2[i]);
    sort(arr1, arr1 + n);
    sort(arr2, arr2 + m);
}
```

---------------------------------------------------------------
## 15. print all subarrays with 0 sum

---------------------------------------------------------------
## 16. Product array Puzzle
```cpp
vector<long long int> productExceptSelf(vector<long long int>& nums, int n) {
    vector<long long int> temp(n);
    if (n == 1) return vector<long long int>({1});

    temp[0] = nums[0];
    for (int i = 1; i < n; i++) temp[i] = temp[i - 1] * nums[i];

    vector<long long int> temp2(n);
    temp2[n - 1] = nums[n - 1];
    for (int i = n - 2; i >= 0; i--) temp2[i] = temp2[i + 1] * nums[i];
    
    nums[0] = temp2[1];
    nums[n - 1] = temp[n - 2];
    for (int i = 1; i < n - 1; i++) nums[i] = temp[i - 1] * temp2[i + 1];
    return nums;
}
```

---------------------------------------------------------------
## 17. Sort array according to count of set bits

---------------------------------------------------------------
## 18. minimum no. of swaps required to sort the array

---------------------------------------------------------------
## 19. Bishu and Soldiers

---------------------------------------------------------------
## 20. Rasta and Kheshtak

---------------------------------------------------------------
## 21. Kth smallest number again

---------------------------------------------------------------
## 22. Find pivot element in a sorted array

---------------------------------------------------------------
## 23. K-th Element of Two Sorted Arrays

---------------------------------------------------------------
## 24. Aggressive cows

---------------------------------------------------------------
## 25. Book Allocation Problem

---------------------------------------------------------------
## 26. EKOSPOJ:

---------------------------------------------------------------
## 27. Job Scheduling Algo

---------------------------------------------------------------
## 28. Missing Number in AP

---------------------------------------------------------------
## 29. Smallest number with atleastn trailing zeroes infactorial

---------------------------------------------------------------
## 30. Painters Partition Problem:

---------------------------------------------------------------
## 31. ROTI-Prata SPOJ

---------------------------------------------------------------
## 32. DoubleHelix SPOJ

---------------------------------------------------------------
## 33. Subset Sums

---------------------------------------------------------------
## 34. Findthe inversion count

---------------------------------------------------------------
## 35. Implement Merge-sort in-place

---------------------------------------------------------------
## 36. Partitioning and Sorting Arrays with Many Repeated Entries

---------------------------------------------------------------
## 37. Bitonic Point 
```cpp
int findMaximum(int arr[], int n) {
        int l = 0;
        int r = n - 1;
        int ans = 0;
        while (l <= r) {
            int mid = (l + r) / 2;
            if (mid <= 0) l = mid + 1;
            else if (mid >= n - 1) r = mid - 1;
            else {
                if (arr[mid] < arr[mid - 1]) r = mid - 1;
                else l = mid + 1;
            }
            ans = max(ans, arr[mid]);
        }
        return ans;
    }
```