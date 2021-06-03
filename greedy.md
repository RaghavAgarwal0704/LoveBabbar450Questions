## 1. Activity Selection Problem
sort activities based on finish time, then include those meetings whose start time is after the finish time of previously selected meeting
 
here sorting using lambda of sort ---- can also do pairsort algorithm

```cpp
int maxMeetings(int start[], int end[], int n) {
    vector<pair<int, int>> v;
    for (int i = 0; i < n; i++) v.push_back({start[i], end[i]});
    sort(v.begin(), v.end(), [](const pair<int, int>& a, const pair<int, int>& b) {
        return a.second < b.second;
    });
    
    int count = 1;//since first meeting is always included
    int j=0;//stores the previously included meeting
    for (int i = 1; i < n; i++)
        if (v[i].first > v[j].second) {
            count++;
            j=i;
        }
    return count;
}
```

pairsort algorithm://sorting done based on the first element
```cpp
void pairsort(int start[],int end[],int n){
        pair<int,int>p[n];
        for(int i=0;i<n;i++){
            p[i].first=end[i];
            p[i].second=start[i];
        }
        sort(p,p+n);
        for(int i=0;i<n;i++){
            start[i]=p[i].second;
            end[i]=p[i].first;
        }
}
``` 

-----------------------------------------------------
## 2. Job SequencingProblem

-----------------------------------------------------
## 3. Huffman Coding

-----------------------------------------------------
## 4. Water Connection Problem
```cpp
vector<vector<int>> solve(int n, int p, vector<int> a, vector<int> b, vector<int> d) {
    vector<vector<int>> ans;
    vector<int> temp(20, 0);
    for (int i : a) temp[i]++;
    for (int i : b) temp[i]++;
    unordered_map<int, pair<int, int>> m;
    for (int i = 0; i < p; i++) m[a[i]] = {b[i], d[i]};
    for (int i = 1; i <= 20; i++) {
        if (temp[i] == 1 && m.find(i) != m.end()) {
            int x = i;
            int dia = INT_MAX;
            while (m.find(x) != m.end()) {
                dia = min(dia, m[x].second);
                x = m[x].first;
                m[i] = {x, dia};
            }
            ans.push_back({i, m[i].first, m[i].second});
        }
    }
    return ans;
}
```

-----------------------------------------------------
## 5. Fractional Knapsack Problem
O(N*logN) time, O(1) space : Greedy on value per unit weight.
```cpp
double fractionalKnapsack(int W, Item arr[], int n)
{
    sort(arr,arr+n,[](const Item& a,const Item& b){
        return (double)a.value/a.weight > (double)b.value/b.weight;
    });
    int i=0;
    double ans=0;
    while(i<n && W){
        ans+=min(arr[i].weight,W)*(double)arr[i].value/arr[i].weight;
        W-=min(arr[i++].weight,W);
    }
    return ans;
}
```

-----------------------------------------------------
## 6. Greedy Algorithm to find Minimum number of Coins

-----------------------------------------------------
## 7. Maximum trains for which stoppage can be provided

-----------------------------------------------------
## 8. Minimum Platforms Problem
```cpp
typedef pair<int, int> pi;
int findPlatform(int arr[], int dep[], int n) {
    vector<pi> v;
    for (int i = 0; i < n; i++) v.push_back(pi(arr[i], dep[i]));
    sort(v.begin(), v.end(), [](const pi& a, const pi& b) {
            return a.second < b.second;
    });
    int ans = 0;
    for (int i = 0; i < n; i++) {
        int temp = 1;
        for (int j = i + 1; j < n; j++) {
            if (v[j].first <= v[i].second) temp++;
        }
        ans = max(ans, temp);
        if (ans == n) return ans;
    }
    return ans;
}
```

-----------------------------------------------------
## 9. Buy Maximum Stocks if i stocks can be bought on i-th day

-----------------------------------------------------
## 10. Find the minimum and maximum amount to buy all N candies
```cpp
vector<int> candyStore(int candies[], int N, int k)
{
    sort(candies,candies+N);
    int min=0,max=0;
    int n=N-1,i=0;
    while(i<=n){
        min+=candies[i++];
        n-=k;
    }
    i=N-1,n=0;
    while(i>=n){
        max+=candies[i--];
        n+=k;
    }
    return vector<int>({min,max});
}
```

-----------------------------------------------------
## 11. Minimize Cash Flow among a given set of friends who have borrowed money from each other

-----------------------------------------------------
## 12. Minimum Cost to cut a board into squares

-----------------------------------------------------
## 13. Check if it is possible to survive on Island

-----------------------------------------------------
## 14. Find maximum meetings in one room

-----------------------------------------------------
## 15. Maximum product subset of an array

-----------------------------------------------------
## 16. Maximize array sum after K negations
```cpp
long long int maximizeSum(long long int a[], int n, int k)
{
    sort(a,a+n);
    long long int ans=accumulate(a,a+n,0LL);
    long long int mi=INT_MAX;
    int i=0;
    while(k--){
        if(a[i]<0) {
            mi=min(mi,a[i]*-1);
            ans-=2*a[i++];
        }
        else {
            mi=min(a[i],mi);
            ans-=2*mi;
            mi*=-1;
        }
    }
    return ans;
}
```

-----------------------------------------------------
## 17. Maximize the sum of arr[i]*i
```cpp
int mod = 1e9 + 7;
int Maximize(int a[], int n) {
    sort(a, a + n);
    long long int ans = 0;
    for (int i = 0; i < n; i++)
        ans += (i * (long long int)a[i]) % mod;

    return ans % mod;
}
```

-----------------------------------------------------
## 18. Maximum sum of absolute difference of an array

-----------------------------------------------------
## 19. Maximize sum of consecutive differences in a circular array
```cpp
long long int maxSum(int arr[], int n)
{
    long long int ans=0;
    sort(arr,arr+n);
    int k=ceil(n/2);
    int t=0;
    for(int i=0;i<k;i++){
        ans+=abs(arr[n-1-t]-arr[i]);
        ans+=abs(arr[n-1-t]-arr[(i+1)%k]);
        t++;
    }
    return ans;
}
```

-----------------------------------------------------
## 20. Minimum sum of absolute difference of pairs of two arrays

-----------------------------------------------------
## 21. Program for Shortest Job First (or SJF) CPU Scheduling

-----------------------------------------------------
## 22. Program for Least Recently Used (LRU) Page Replacement algorithm
```cpp
int pageFaults(int N, int C, int pages[]) {
    unordered_map<int, list<int>::iterator> m;
    list<int> mem;
    int ans = 0;
    for (int i = 0; i < N; i++) {
        if (m.find(pages[i]) != m.end())
            mem.erase(m[pages[i]]);
        else if (C)
            ans++, C--;
        else {
            m.erase(mem.front());
            mem.pop_front();
            ans++;
        }
        mem.push_back(pages[i]);
        m[pages[i]] = --mem.end();
    }
    return ans;
}
```

-----------------------------------------------------
## 23. Smallest subset with sum greater than all other elements

-----------------------------------------------------
## 24. Chocolate Distribution Problem
```cpp
long long findMinDiff(vector<long long> a, long long n, long long m){
    sort(a.begin(),a.end());
    long long ans=LONG_LONG_MAX;
    for(int i=m-1;i<n;i++){
        ans=min(ans,a[i]-a[i-m+1]);
    }
    return ans;
}
```
24.1 chocolate distribution 2--> gfg (first left,then right)
```cpp
long candies(int n, vector<int> arr) {
    int candy[n];
    for(int i=0;i<n;i++)candy[i]=1;
    for(int i=1;i<n;i++){
        if(arr[i]>arr[i-1])candy[i]=candy[i-1]+1;
    }
    for(int i=n-2;i>=0;i--){
        if(arr[i]>arr[i+1])candy[i]=max(candy[i],candy[i+1]+1);
    }
    long sum=0;
    for(int i=0;i<n;i++)sum+=candy[i];
    return sum;
}
```

-----------------------------------------------------
## 25. DEFKIN -Defense of a Kingdom

-----------------------------------------------------
## 26. DIEHARD -DIE HARD

-----------------------------------------------------
## 27. GERGOVIA -Wine trading in Gergovia

-----------------------------------------------------
## 28. Picking Up Chicks

-----------------------------------------------------
## 29. CHOCOLA –Chocolate

-----------------------------------------------------
## 30. ARRANGE -Arranging Amplifiers

-----------------------------------------------------
## 31. K Centers Problem

-----------------------------------------------------
## 32. Minimum Cost of ropes
```cpp
long long minCost(long long arr[], long long n) {
    long long int ans=0;
    priority_queue<long long int, vector<long long int>, greater<long long int> >q;
    for(int i=0;i<n;i++) q.push(arr[i]);
    while(q.size()!=1){
        long long int x=q.top();
        q.pop();
        long long int y=q.top();
        q.pop();
        q.push(x+y);
        ans+=x+y;
    }
    return ans;
}
```

-----------------------------------------------------
## 33. Find smallest number with given number of digits and sum of digits
```cpp
string smallestNumber(int S, int D){
    string ans="";
    while(D--){
        if(S>9) S-=9,ans="9"+ans;
        else {
            if(D) ans=(char)(S-1 +'0')+ans,S=1;
            else ans=(char)(S +'0')+ans,S=0;
        }
    }
    return (S!=0) ?"-1":ans;
}
```

-----------------------------------------------------
## 34. Rearrange characters in a string such that no two adjacent are same

-----------------------------------------------------
## 35. Find maximum sum possible equal sum of three stacks

-----------------------------------------------------
