## 0. fibonnaci

1) using recursion-->exponential

2) using dp (bottom up)-->tabulation

3) using dp (top down)-->memoization
```cpp
class Solution {
  public:
    long long int dp[1000];
    long long int fib(long long int n){
        long long int first;
        long long int second;
        if(n<=1)return n;
        if(dp[n-1]!=-1)first=dp[n-1];
        else first=fib(n-1);
        if(dp[n-2]!=-1)second=dp[n-2];
        else second=fib(n-2);
        return dp[n]=(first+second)%cons;
    }
    long long int nthFibonacci(long long int n){
        memset(dp,-1,sizeof(dp));
        return fib(n);
    }
};
```

4) direct formula--> constant space and time
Fn = {[(√5 + 1)/2] ^ n} / √5 
```cpp
int fib(int n) {
  double phi = (1 + sqrt(5)) / 2;
  return round(pow(phi, n) / sqrt(5));
}
```

------------------------------
## 1. Coin ChangeProblem

1) Naīve Recursive method : 
```cpp
long long int count( int S[], int m, int n ){
    if(n==0 || m==0) return 0;
    if(S[m-1]<=n) {
        if(S[m-1]==n) return count(S,m-1,n)+1LL;
        return count(S,m,n-S[m-1])+count(S,m-1,n);
    }
    return count(S,m-1,n);
}
```

2) Recursive with memoization : O(m*n) space, O(n*m) time;
```cpp
long long int util(int S[], int m, int n) {
    if (n == 0 || m == 0) return 0;
    if (dp[n][m] != -1) return dp[n][m];
    if (S[m - 1] <= n) {
        if (S[m - 1] == n) dp[n][m] = util(S, m - 1, n) + 1LL;
        else dp[n][m] = util(S, m, n - S[m - 1]) + util(S, m - 1, n);
    } else
        dp[n][m] = util(S, m - 1, n);
    return dp[n][m];
}
long long int count(int S[], int m, int n) {
    dp = vector<vector<long long>>(n + 1, vector<long long>(m + 1, -1));
    return util(S, m, n);
}
```

3) Iterative(top-down) : O(m*n) space, O(n*m) time;
```cpp
vector<vector<long long>> dp;
long long int count(int S[], int m, int n) {
    vector<vector<long long>> dp(n + 1, vector<long long>(m + 1, 0));
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            dp[i][j] = dp[i][j - 1];
            if (S[j - 1] == i)
                dp[i][j] = dp[i][j - 1] + 1LL;
            else if (S[j - 1] < i)
                dp[i][j] = dp[i - S[j - 1]][j] + dp[i][j - 1];
        }
    }
    return dp[n][m];
}
```

----------------------------------------------
## 2. Knapsack Problem

1) Naīve Recursive method : 
```cpp
int knapSack(int W, int wt[], int val[], int n) {
    if (n <= 0 || W <= 0) return 0;
    if (wt[n - 1] <= W)
        return max(val[n - 1] + util(W - wt[n - 1], wt, val, n - 1), util(W, wt, val, n - 1));
    return util(W, wt, val, n - 1);
}
```

2) Recursive method with memoization : O(n*W) space, O(n*W) time
```cpp
vector<vector<int>> dp;
int util(int W, int wt[], int val[], int n) {
    if (n <= 0 || W <= 0) return 0;
    if (dp[W][n] != -1) return dp[W][n];
    if (wt[n - 1] <= W)
        dp[W][n] = max(val[n - 1] + util(W - wt[n - 1], wt, val, n - 1), util(W, wt, val, n - 1));
    else
        dp[W][n] = util(W, wt, val, n - 1);
    return dp[W][n];
}

int knapSack(int W, int wt[], int val[], int n) {
    dp = vector<vector<int>>(W + 1, vector<int>(n + 1, -1));
    return util(W, wt, val, n);
}
```

3) Iterative method : O(n*W) space, O(n*W) time
```cpp
int knapSack(int W, int wt[], int val[], int n) {
    vector<vector<int>> dp(W + 1, vector<int>(n + 1, 0));
    for (int i = 1; i <= W; i++)
        for (int j = 1; j <= n; j++) {
            if (i >= wt[j - 1])
                dp[i][j] = max(val[j - 1] + dp[i - wt[j - 1]][j - 1], dp[i][j - 1]);
            else
                dp[i][j] = dp[i][j - 1];
        }
    return dp[W][n];
}
```

4) tabulation+space optimized space=o(w) store only 2 rows
```cpp
int KnapSack(int val[], int wt[], int n, int W)
{
    // matrix to store final result
    int mat[2][W+1];
    memset(mat, 0, sizeof(mat));
 
    // iterate through all items
    int i = 0;
    while (i < n) // one by one traverse each element
    {
        int j = 0; // traverse all weights j <= W
 
        // if i is odd that mean till now we have odd
        // number of elements so we store result in 1th
        // indexed row
        if (i%2!=0)
        {
            while (++j <= W) // check for each value
            {
                if (wt[i] <= j) // include element
                    mat[1][j] = max(val[i] + mat[0][j-wt[i]],
                                    mat[0][j] );
                else           // exclude element
                    mat[1][j] = mat[0][j];
            }
 
        }
 
        // if i is even that mean till now we have even number
        // of elements so we store result in 0th indexed row
        else
        {
            while(++j <= W)
            {
                if (wt[i] <= j)
                    mat[0][j] = max(val[i] + mat[1][j-wt[i]],
                                     mat[1][j]);
                else
                    mat[0][j] = mat[1][j];
            }
        }
        i++;
    }
 
    // Return mat[0][W] if n is odd, else mat[1][W]
    return (n%2 != 0)? mat[0][W] : mat[1][W];
}
```

----------------------------------------------
## 3. Binomial CoefficientProblem

1) Naīve method : O(N) time O(N) space
```cpp
int mod=1e9+7;
vector<long long int>facts;
long long int fact(int n){
    if(n==0) return 1;
    if(facts[n]!=-1) return facts[n];
    return facts[n]=(fact(n-1)*n);
}
int nCr(int n, int r){
    if(r>n) return 0;
    else if(r==n) return 1;
    facts=vector<long long int>(n+1,-1);
    return (int)(fact(n)/fact(r)*fact(n-r))%mod;
}
```

2) Better appraoch using formula : 

C(n, r) = C(n-1, r-1) + C(n-1, r)
C(n, 0) = C(n, n) = 1
```cpp
vector<vector<int>>dp;
int util(int n,int r){
    if(r==0 || n==r) return dp[n][r]=1;
    if(dp[n][r]!=-1) return dp[n][r];
    return dp[n][r]=(util(n-1,r-1)+util(n-1,r))%mod;
}
int nCr(int n, int r){
    if(r>n) return 0;
    else if(r==n) return 1;
    dp=vector<vector<int>>(n+1,vector<int>(r+1,-1));
    return util(n,r);
}
```

----------------------------------------------
## 4. Permutation CoefficientProblem

P(n, r) = P(n-1, r) + r* P(n-1, r-1) 
P(n, 0) = 1
```cpp
vector<vector<int>>dp;
int util(int n,int r){
    if(r==0) return dp[n][r]=1;
    if(dp[n][r]!=-1) return dp[n][r];
    return dp[n][r]=(r*util(n-1,r-1)+util(n-1,r))%mod;
}
int nPr(int n, int r){
    if(r>n) return 0;
    else if(r==n) return 1;
    dp=vector<vector<int>>(n+1,vector<int>(r+1,-1));
    return util(n,r);
}
```

----------------------------------------------
## 5. Program for nth Catalan Number

C(5)= C(0)*C(4) + C(1)*C(3) + C(2)*C(2) + C(3)*C(2) + C(4)*C(1)
1) O(N) space , O(N^2) time
```cpp
cpp_int findCatalan(int n) 
{
    vector<cpp_int>dp(n+2,0);
    dp[0]=1;
    dp[1]=1;
    for(int i=2;i<=n+1;i++){
        for(int j=0;j<=i/2;j++){
            if(j!=(i-j)) dp[i]+=2*dp[j]*dp[i-j];
            else dp[i]+=dp[j]*dp[i-j];
        }
    }
    return dp[n+1];
}
```

2) O(N) time and O(1) space
```cpp
cpp_int findCatalan(int n) 
{
    cpp_int ans=1;
    for(int i=1;i<=n;i++) ans = (((2*i)*(2*i-1)*ans)/(i*(i+1)));
    return ans;    
}
```

----------------------------------------------
## 6. Matrix Chain Multiplication 
```cpp
int util(int arr[], int i, int j) {
    if (i >= j) return 0;
    if (dp[i][j] != -1) return dp[i][j];
    int ans = INT_MAX;
    for (int k = i; k < j; k++) {
        ans = min(ans, (arr[i - 1] * arr[k] * arr[j]) + util(arr, i, k) + util(arr, k + 1, j));
    }
    return dp[i][j] = ans;
}
vector<vector<int>> dp;
int matrixMultiplication(int N, int arr[]) {
    dp = vector<vector<int>>(N + 1, vector<int>(N + 1, -1));
    return util(arr, 1, N - 1);
}
```

----------------------------------------------
## 7. Edit Distance

1) Recursive with memoization : 
```cpp
vector<vector<int>> dp;
int util(string s, string t, int x, int y) {
    if (x == 0 && y == 0) return 0;
    if (x == 0) return y;
    if (y == 0) return x;
    if (dp[x][y] != -1) return dp[x][y];
    if (s[x - 1] == t[y - 1])
        dp[x][y] = util(s, t, x - 1, y - 1);
    else
        dp[x][y] = 1 + min(util(s, t, x, y - 1), min(util(s, t, x - 1, y), util(s, t, x - 1, y - 1)));
    return dp[x][y];
}
int editDistance(string s, string t) {
    // Code here
    int x = s.length();
    int y = t.length();
    dp = vector<vector<int>>(x + 1, vector<int>(y + 1, -1));
    return util(s, t, x, y);
}
```

2) Iterative :
```cpp
int editDistance(string s, string t) {
    int x = s.length();
    int y = t.length();
    vector<vector<int>> dp(x + 1, vector<int>(y + 1, -1));
    dp[0][0] = 0;
    for (int i = 0; i <= x; i++) dp[i][0] = i;
    for (int i = 0; i <= y; i++) dp[0][i] = i;
    for (int i = 1; i <= x; i++)
        for (int j = 1; j <= y; j++) {
            if (s[i - 1] == t[j - 1])
                dp[i][j] = dp[i - 1][j - 1];
            else
                dp[i][j] = 1 + min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i][j - 1]));
        }
    return dp[x][y];
}
```

----------------------------------------------
## 8. Subset Sum Problem-->equal sum partition
```cpp
int equalPartition(int N, int arr[])
    {
        int s=0;
        for(int i=0;i<N;i++)s+=arr[i];
        if(s%2!=0)return 0;
        int sum=s/2;
        bool dp[N+1][sum+1];
        for(int i=0;i<=N;i++)dp[i][0]=true;
        for(int i=1;i<=sum;i++)dp[0][i]=false;
        for(int i=1;i<=N;i++){
            for(int j=1;j<=sum;j++){
                if(arr[i-1]>j)dp[i][j]=dp[i-1][j];
                else dp[i][j]=dp[i-1][j]||dp[i-1][j-arr[i-1]];
            }
        }
        return dp[N][sum];
    }
```
    ----------OR-----------------

1) Naīve Recursive method :
```cpp
int util(int N,int arr[],int sum){
    if(N==0 ) return sum==0;
    if(N==0 && sum!=0) return 0;
    if(arr[N-1]<=sum) 
        return util(N-1,arr,sum-arr[N-1]) || util(N-1,arr,sum);
    return util(N-1,arr,sum);
}

int equalPartition(int N, int arr[])
{
    int sum=0;
    for(int i=0;i<N;i++) sum+=arr[i];
    if(sum&1 || N<=1 ) return 0;
    return util(N,arr,sum/2);
}
```

2) Recursive method with memoization : O(N*sum/2) space, O(N*sum/2) time
```cpp
vector<vector<int>> dp;
int util(int N, int arr[], int sum) {
    if (N == 0) return sum == 0;
    if (N == 0 && sum != 0) return 0;
    if (dp[sum][N] != -1) return dp[sum][N];
    if (arr[N - 1] <= sum) {
        dp[sum][N] = util(N - 1, arr, sum - arr[N - 1]) || util(N - 1, arr, sum);
    } else
        dp[sum][N] = util(N - 1, arr, sum);
    return dp[sum][N];
}
int equalPartition(int N, int arr[]) {
    int sum = 0;
    for (int i = 0; i < N; i++) sum += arr[i];
    if (sum & 1 || N <= 1) return 0;
    dp = vector<vector<int>>(sum / 2 + 1, vector<int>(N + 1, -1));
    return util(N, arr, sum / 2);
}
```

3) Iterative(TOP Down) : O(N*sum/2) space, O(N*sum/2) time
```cpp
int equalPartition(int N, int arr[]) {
    int sum = 0;
    for (int i = 0; i < N; i++) sum += arr[i];
    if (sum & 1 || N <= 1) return 0;
    vector<vector<int>>dp(sum / 2 + 1, vector<int>(N + 1, -1));
    for (int i = 1; i <= sum / 2; i++) dp[i][0] = 0;
    for (int i = 0; i <= N; i++) dp[0][i] = 1;
    for (int i = 1; i <= sum / 2; i++)
        for (int j = 1; j <= N; j++) {
            if (i >= arr[j - 1])
                dp[i][j] = dp[i - arr[j - 1]][j - 1] || dp[i][j - 1];
            else
                dp[i][j] = dp[i][j - 1];
        }
    return dp[sum / 2][N];
}
```

----------------------------------------------
## 9. Friends Pairing Problem

1) O(N) space O(N) time
```cpp
int countFriendsPairings(int n) 
{ 
    int mod=1e9+7;
    vector<int>dp(n+1,-1);
    dp[0]=0;
    dp[1]=1;
    dp[2]=2;
    for(int i=3;i<=n;i++)
        dp[i]=(dp[i-1]+((long long int)(i-1)*dp[i-2])%mod)%mod;
    return dp[n];
    
}
```

2) O(1) space O(N) time
```cpp
int countFriendsPairings(int n) 
{ 
    int mod=1e9+7;
    if(n==0) return 0;
    int a=1,b=2;
    if(n==1) return a;
    if(n==2) return b;
    for(int i=3;i<=n;i++){
        int temp=b;
        b=(b+((long long int)(i-1)*a)%mod)%mod;
        a=temp;
    }
    return b;
}
```

----------------------------------------------
## 10. Gold Mine Problem

1) Naīve Recursive :
```cpp
int maxGold(int n, int m, vector<vector<int>> M) {
    if (n > M.size()) return 0;
    if (m == 0 || n == 0) return 0;
    return M[n - 1][m - 1] + max(maxGold(n, m - 1, M), max(maxGold(n - 1, m - 1, M), maxGold(n + 1, m - 1, M)));
}
```

2) Recursive with memoization : O(m*n) space, O(n*m) time;
```cpp
vector<vector<int>> dp;
int util(int n, int m, vector<vector<int>> M) {
    if (n > M.size() || m > M[0].size()) return 0;
    if (m == 0 || n == 0) return 0;
    if (dp[n][m] != -1) return dp[n][m];
    dp[n][m] = M[n - 1][m - 1] + max(util(n, m + 1, M), max(util(n - 1, m + 1, M), util(n + 1, m + 1, M)));
    return dp[n][m];
}
int maxGold(int n, int m, vector<vector<int>> M) {
    dp = vector<vector<int>>(n + 1, vector<int>(m + 1, -1));
    int ans = INT_MIN;
    for (int i = 1; i <= n; i++) {
        ans = max(ans, util(i, 1, M));
    }
    return ans;
}
```

3) Iterative :  O(m*n) space, O(n*m) time;
```cpp
int maxGold(int n, int m, vector<vector<int>> M) {
    dp = vector<vector<int>>(n + 1, vector<int>(m + 1, -1));
    for (int i = 0; i < n; i++) dp[i][0] = 0;
    for (int i = 0; i < m; i++) dp[0][i] = 0;
    int ans = INT_MIN;
    for (int j = 1; j <= m; j++)
        for (int i = 1; i <= n; i++) {
            dp[i][j] = M[i - 1][j - 1] + max(dp[i][j - 1], max((i == n) ? 0 : dp[i + 1][j - 1], dp[i - 1][j - 1]));
            ans = max(ans, dp[i][j]);
        }
    return ans;
}
```

----------------------------------------------
## 11. Assembly Line SchedulingProblem

----------------------------------------------
## 12. Painting the Fenceproblem

----------------------------------------------
## 13. Maximize The Cut Segments

1) Naīve Recursive :
```cpp
int maximizeTheCuts(int n, int x, int y, int z) {
    if (n == 0) return 0;
    int ans = -1;
    if (x <= n) ans = max(ans, maximizeTheCuts(n - x, x, y, z));
    if (y <= n) ans = max(ans, maximizeTheCuts(n - y, x, y, z));
    if (z <= n) ans = max(ans, maximizeTheCuts(n - z, x, y, z));
    return ans + 1;
}
```

2) Recursive with memoization : O(N) time, O(N) space
```cpp
vector<int> dp;
int util(int n, int x, int y, int z) {
    if (n == 0) return 0;
    int ans = -1;
    if (dp[n] != -1) return dp[n];
    if (x <= n) ans = max(ans, util(n - x, x, y, z));
    if (y <= n) ans = max(ans, util(n - y, x, y, z));
    if (z <= n) ans = max(ans, util(n - z, x, y, z));
    dp[n] = ans == -1 ? INT_MIN : ans + 1;
    return dp[n];
}
int maximizeTheCuts(int n, int x, int y, int z) {
    dp = vector<int>(n, -1);
    int ans = util(n, x, y, z);
    return ans < 0 ? 0 : ans;
}
```

3) Iterative : 
```cpp
int maximizeTheCuts(int n, int x, int y, int z) {
    dp = vector<int>(n + 1, -1);
    dp[0] = 0;

    for (int i = 1; i <= n; i++) {
        dp[i] = -1;
        if (x <= i) dp[i] = max(dp[i], dp[i - x]);
        if (y <= i) dp[i] = max(dp[i], dp[i - y]);
        if (z <= i) dp[i] = max(dp[i], dp[i - z]);
        if (dp[i] != -1) dp[i]++;
    }
    return dp[n] == -1 ? 0 : dp[n];
}
```

----------------------------------------------
## 14. Longest Common Subsequence

in copy

---------------------OR-----------------------
1) Recursive with memoization :
```cpp
vector<vector<int>> dp;
int util(int x, int y, string s1, string s2) {
    if (x > s1.length() || y > s2.length()) return 0;
    if (dp[x][y] != -1) return dp[x][y];
    if (s1[x - 1] == s2[y - 1])
        dp[x][y] = 1 + util(x + 1, y + 1, s1, s2);
    else
        dp[x][y] = max(util(x, y + 1, s1, s2), util(x + 1, y, s1, s2));
    return dp[x][y];
}
int lcs(int x, int y, string s1, string s2) {
    dp = vector<vector<int>>(x + 1, vector<int>(y + 1, -1));
    return util(1, 1, s1, s2);
}
```

2) Iterative :
```cpp
int lcs(int x, int y, string s1, string s2) {
    auto dp = vector<vector<int>>(x + 1, vector<int>(y + 1, -1));
    for (int i = 0; i <= x; i++) dp[i][0] = 0;
    for (int i = 0; i <= y; i++) dp[0][i] = 0;
    for (int i = 1; i <= x; i++)
        for (int j = 1; j <= y; j++) {
            if (s1[i - 1] == s2[j - 1])
                dp[i][j] = 1 + dp[i - 1][j - 1];
            else
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
        }
    return dp[x][y];
}
```

----------------------------------------------
## 15. Longest Repeated Subsequence
```cpp
int find(string a,string b,int m,int n){
    int dp[m+1][n+1];
    for(int i=0;i<=m;i++)dp[i][0]=0;
    for(int j=0;j<=n;j++)dp[0][j]=0;
    for(int i=1;i<=m;i++){
        for(int j=1;j<=n;j++){
            if(a[i-1]==b[j-1]&&i!=j)dp[i][j]=1+dp[i-1][j-1];
            else dp[i][j]=max(dp[i-1][j],dp[i][j-1]);
        }
    }
    return dp[m][n];
}
int LongestRepeatingSubsequence(string str){
    return find(str,str,str.size(),str.size());
}
```
----------------------------------------------
## 16. Longest Increasing Subsequence

1) Naīve approach : O(N^2) time and O(N) space
```cpp
int longestSubsequence(int n, int a[])
{
    vector<int>dp(n+1,1);
    for(int i=1;i<n;i++){
        int m=INT_MIN;
        for(int j=0;j<i;j++) if(a[i]>a[j])m=max(m,dp[j]);
        dp[i]=(m==INT_MIN)?1:m+1;
    }
    return *max_element(dp.begin(),dp.end());
}
```

2) Better method : O(N*logN) time and O(N) space
```cpp
int longestSubsequence(int n, int a[])
{
    
    vector<int>ans({a[0]});
    for(int i=1;i<n;i++){
        int ind=lower_bound(ans.begin(),ans.end(),a[i])-ans.begin();   
        if(ind==ans.size()) ans.push_back(a[i]);
        else ans[ind]=a[i];
    }
    return ans.size();
}
```

----------------------------------------------
## 17. Space Optimized Solution of LCS

----------------------------------------------
## 18. LCS (Longest Common Subsequence) of three strings

1) Recursive with memoization : O(n1*n2*n3) space,O(n1*n2*n3) time
```cpp
vector<vector<vector<int>>> dp;
int util(string s1, string s2, string s3, int x, int y, int z) {
    if (x > s1.length() || y > s2.length() || z > s3.length()) return 0;
    if (dp[x][y][z] != -1) return dp[x][y][z];
    if (s1[x - 1] == s2[y - 1] && s2[y - 1] == s3[z - 1])
        dp[x][y][z] = 1 + util(s1, s2, s3, x + 1, y + 1, z + 1);
    else
        dp[x][y][z] = max(util(s1, s2, s3, x, y, z + 1), max(util(s1, s2, s3, x, y + 1, z), util(s1, s2, s3, x + 1, y, z)));
    return dp[x][y][z];
}
int LCSof3(string s1, string s2, string s3, int x, int y, int z) {
    dp = vector<vector<vector<int>>>(x + 1, vector<vector<int>>(y + 1, vector<int>(z + 1, -1)));
    return util(s1, s2, s3, 1, 1, 1);
}
```

2) Iterative : O(n1*n2*n3) space,O(n1*n2*n3) time
```cpp
int LCSof3(string s1, string s2, string s3, int x, int y, int z) {
    vector<vector<vector<int>>> dp(x + 1, vector<vector<int>>(y + 1, vector<int>(z + 1, -1)));
    for (int i = 0; i <= x; i++)
        for (int j = 0; j <= y; j++)
            for (int k = 0; k <= z; k++)
                if (i == 0 || j == 0 || k == 0) dp[i][j][k] = 0;

    for (int i = 1; i <= x; i++)
        for (int j = 1; j <= y; j++)
            for (int k = 1; k <= z; k++) {
                if (s1[i - 1] == s2[j - 1] && s1[i - 1] == s3[k - 1])
                    dp[i][j][k] = 1 + dp[i - 1][j - 1][k - 1];
                else {
                    dp[i][j][k] = max(dp[i][j][k], dp[i][j][k - 1]);
                    dp[i][j][k] = max(dp[i][j][k], dp[i][j - 1][k]);
                    dp[i][j][k] = max(dp[i][j][k], dp[i - 1][j][k]);
                }
            }
    return dp[x][y][z] == -1 ? 0 : dp[x][y][z];
}
```

----------------------------------------------
## 19. Maximum Sum Increasing Subsequence

----------------------------------------------
## 20. Count all subsequences having product less than K

----------------------------------------------
## 21. Longest subsequence such that difference between adjacent is one

----------------------------------------------
## 22. Maximum subsequence sum such that no three are consecutive

----------------------------------------------
## 23. Egg Dropping Problem
```cpp
int util(int i, int j, int e) {
    if (j - i + 1 == 0 || j - i + 1 == 1) return j - i + 1;
    if (e == 1) return j - i + 1;
    if (e == 0) return 0;
    if (dp[e][j - i + 1] != -1) return dp[e][j - i + 1];
    int ans = INT_MAX;
    for (int k = i; k <= j; k++) {
        ans = min(ans, 1 + max(util(i, k - 1, e - 1), util(k + 1, j, e)));
    }
    return dp[e][j - i + 1] = ans;
}
int dp[202][202];
int eggDrop(int n, int k) {
    memset(dp, -1, sizeof(dp));
    return util(1, k, n);
}
```

----------------------------------------------
## 24. Maximum Length Chain of Pairs

```cpp
int maxChainLen(struct val p[], int n) {
    sort(p, p + n, [](struct val &a, struct val &b) {
        return a.second < b.second;
    });
    vector<int> dp(n + 1, 1);
    dp[0] = 0;
    int pre = INT_MIN;
    int ans = INT_MIN;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j < i; j++)
            if (p[i - 1].first > p[j - 1].second)
                dp[i] = max(dp[i], dp[j] + 1);
        ;
        ans = max(ans, dp[i]);
    }
    return ans;
}
```

----------------------------------------------
## 25. Maximum size square sub-matrix with all 1s

O(M*N) time , O(M*N) space : 
```cpp
int maxSquare(int n, int m, vector<vector<int>> mat){
    vector<vector<int>>dp(n+1,vector<int>(m+1,0));
    int ans=0;
    for(int i=1;i<=n;i++){
        for(int j=1;j<=m;j++){
            if(mat[i-1][j-1]==1) dp[i][j]=min(dp[i-1][j],min(dp[i][j-1],dp[i-1][j-1]))+1;
            ans=max(ans,dp[i][j]);
        }
    }
    return ans;
}
```

----------------------------------------------
## 26. Maximum sum of pairs with specific difference

----------------------------------------------
## 27. Min Cost PathProblem

----------------------------------------------
## 28. Maximum difference of zeros and ones in binary string
```cpp
int maxSubstring(string S)
{
    int ans=0;int curr=0;
    for(int i=0;i<S.length();i++){
        curr=max(curr+(S[i]=='1'?-1:1),0);
        ans=max(ans,curr);
    }
    return ans==0?-1:ans;
}
```

----------------------------------------------
## 29. Minimum number of jumps to reach end
```cpp
int minJumps(int nums[], int n){
    if(n<=1) return 0;
    int i=0,j=0;
    for(int k=0;k<n;k++){
        int far=0;
        for(int w=i;w<=j;w++){
            if(w+nums[w]>=n-1) return k+1;
            far=max(far,nums[w]+w);
        }
        i=j+1;
        j=far;
    }
    return -1;      
}
```

----------------------------------------------
## 30. Minimum cost to fill given weight in a bag

----------------------------------------------
## 31. Minimum removals from array to make max –min <= K

----------------------------------------------
## 32. Longest Common Substring
```cpp
int longestCommonSubstr(string s1, string s2, int x, int y) {
    vector<vector<int>> dp(x + 1, vector<int>(y + 1, 0));
    int ans = 0;
    for (int i = 1; i <= x; i++)
        for (int j = 1; j <= y; j++) {
            if (s1[i - 1] == s2[j - 1]) dp[i][j] = 1 + dp[i - 1][j - 1];
            ans = max(ans, dp[i][j]);
        }
    return ans;
}
```

----------------------------------------------
## 33. Count number of ways to reach a given score in a game

1) Naīve Recursive method : 
```cpp
long long int util( int S[], int m,long long int n ){
    if(n==0 || m==0) return 0;
    if(S[m-1]<=n) {
        if(S[m-1]==n) return util(S,m-1,n)+1LL;
        return util(S,m,n-S[m-1])+util(S,m-1,n);
    }
    return util(S,m-1,n);
}

long long int count(long long int n)
{
    int S[]={3,5,10};
    return util(S,3,n);
}
```

2) Recursive with memoization : O(n) space, O(n) time;
```cpp
vector<vector<long long>> dp;
long long int util(int S[], int m, int n) {
    if (n == 0 || m == 0) return 0;
    if (dp[n][m] != -1) return dp[n][m];
    if (S[m - 1] <= n) {
        if (S[m - 1] == n) dp[n][m] = util(S, m - 1, n) + 1LL;
        else dp[n][m] = util(S, m, n - S[m - 1]) + util(S, m - 1, n);
    } else
        dp[n][m] = util(S, m - 1, n);
    return dp[n][m];
}
long long int count(long long int n) {
    dp = vector<vector<long long>>(n + 1, vector<long long>(4, -1));
    int S[]={3,5,10};
    return util(S, 3, n);
}
```

3) Iterative(top-down) : O(n) space, O(n) time;
```cpp
long long int count(long long int n) {
    vector<vector<long long>> dp(n + 1, vector<long long>(4, 0));
    int S[]={3,5,10};
    int m=3;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            dp[i][j] = dp[i][j - 1];
            if (S[j - 1] == i)
                dp[i][j] = dp[i][j - 1] + 1LL;
            else if (S[j - 1] < i)
                dp[i][j] = dp[i - S[j - 1]][j] + dp[i][j - 1];
        }
    }
    return dp[n][m];
}
```

----------------------------------------------
## 34. Count Balanced Binary Trees of Height h

----------------------------------------------
## 35. LargestSum Contiguous Subarray [V>V>V>V IMP ]

----------------------------------------------
## 36. Smallest sum contiguous subarray

----------------------------------------------
## 37. Unbounded Knapsack (Repetition of items allowed)

```cpp
int knapSack(int n, int W, int val[], int wt[]) {
    // code here
    vector<int> dp(W + 1, 0);
    for (int i = 0; i < n; i++)
        for (int j = 1; j <= W; j++)
            if (wt[i] <= j)
                dp[j] = max(val[i] + dp[j - wt[i]], dp[j]);

    return dp[W];
}
```

----------------------------------------------
## 38. Word Break Problem

----------------------------------------------
## 39. Largest Independent Set Problem

----------------------------------------------
## 40. Partition problem

----------------------------------------------
## 41. Longest Palindromic Subsequence (Length + String)

https://leetcode.com/problems/longest-palindromic-subsequence/

```cpp
int longestPalindromeSubseq(string s) {
    string p = s;
    reverse(p.begin(), p.end());
    int n = s.length();
    vector<vector<int>> dp(n + 1, vector<int>(n + 1, 0));
    // ans= LCS(s,reverse(s))
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            if (s[i - 1] == p[j - 1])
                dp[i][j] = 1 + dp[i - 1][j - 1];
            else
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
        }
    }
    int i = n, j = n;
    string ans = "";
    while (i > 0 && j > 0 && dp[i][j] != 0) {
        if (s[i - 1] == p[j - 1]) {
            ans = s[i - 1] + ans;
            i--;
            j--;
        } else {
            if (dp[i - 1][j] > dp[i][j - 1])
                i--;
            else
                j--;
        }
    }
    cout << ans << endl;
    return dp[n][n];
}
```

----------------------------------------------
## 42. Count All Palindromic Subsequence in a given String

----------------------------------------------
## 43. Longest Palindromic Substring

----------------------------------------------
## 44. Longest alternating subsequence

----------------------------------------------
## 45. Weighted Job Scheduling

----------------------------------------------
## 46. Coin game winner where every player has three choices

----------------------------------------------
## 47. Count Derangements (Permutation such that no element appears in its original position) [ IMPORTANT ]

----------------------------------------------
## 48. Maximum profit by buying and selling a share at most twice [ IMP ]

----------------------------------------------
## 49. Optimal Strategy for a Game
```cpp
vector<vector<int>>dp;
long long util(int arr[], int i,int j){
    if(i>j) return 0;
    if(i==j) return dp[i][j]=arr[i];
    if(j-i==1) return dp[i][j]=max(arr[i],arr[j]);
    if(dp[i][j]!=-1) return dp[i][j];
    return dp[i][j]=max(
        arr[i]+min(util(arr,i+2,j),util(arr,i+1,j-1)),
        arr[j]+min(util(arr,i+1,j-1),util(arr,i,j-2))
    );
}

long long maximumAmount(int arr[], int n) 
{
    dp=vector<vector<int>>(n+1,vector<int>(n+1,-1));
    return util(arr,0,n-1);
}
```

----------------------------------------------
## 50. Optimal Binary Search Tree

----------------------------------------------
## 51. Palindrome Partitioning Problem
```cpp
bool isPalindrome(string &s, int i, int j) {
    while (i < j)
        if (s[i++] != s[j--]) return false;
    return true;
}
int util(string &s, int i, int j) {
    if (i >= j) return 0;
    if (dp[i][j] != -1) return dp[i][j];
    if (isPalindrome(s, i, j)) return dp[i][j] = 0;
    int ans = INT_MAX;
    for (int k = i; k < j; k++) ans = min(ans, 1 + util(s, i, k) + util(s, k + 1, j));

    return dp[i][j] = ans;
}
vector<vector<int>> dp;
int palindromicPartition(string str) {
    // code here
    int N = str.length();
    dp = vector<vector<int>>(N + 1, vector<int>(N + 1, -1));
    return util(str, 0, str.size() - 1);
}
```

----------------------------------------------
## 52. Word Wrap Problem

----------------------------------------------
## 53. Mobile Numeric Keypad Problem [ IMP ]

----------------------------------------------
## 54. Boolean Parenthesization Problem
```cpp
int util(string &s, int i, int j, bool b) {
    if (i > j) return 0;
    if (dp[i][j][b] != -1) return dp[i][j][b];
    if (i == j) return dp[i][j][b] = (b ? (s[i] == 'T') : (s[i] == 'F'));

    long long int ans = 0;
    for (int k = i + 1; k < j; k += 2) {
        int lT = util(s, i, k - 1, true);
        int lF = util(s, i, k - 1, false);
        int rT = util(s, k + 1, j, true);
        int rF = util(s, k + 1, j, false);
        if (s[k] == '&')
            ans += (b ? (lT * rT) : ((lT * rF) + (lF * rT) + (lF * rF)));
        else if (s[k] == '|')
            ans += (b ? ((lT * rT) + (lT * rF) + (lF * rT)) : (lF * rF));
        else if (s[k] == '^')
            ans += (b ? ((lT * rF) + (lF * rT)) : ((lT * rT) + (lF * rF)));
    }
    return dp[i][j][b] = ans % 1003;
}
int dp[202][202][2];
int countWays(int N, string S) {
    // use memset instead of vector
    // dp=vector<vector<vector<int>>>(202,vector<vector<int>>(202,vector<int>(2,-1)));
    memset(dp, -1, sizeof(dp));
    return util(S, 0, N - 1, true);
}
```

----------------------------------------------
## 55. Largest rectangular sub-matrix whose sum is 0

----------------------------------------------
## 56. Largest area rectangular sub-matrix with equal number of 1’s and 0’s [ IMP ]

----------------------------------------------
## 57. Maximum sum rectangle in a 2D matrix

----------------------------------------------
## 58. Maximum profit by buying and selling a share at most k times

----------------------------------------------
## 59. Find if a string is interleaved of two other strings

----------------------------------------------
## 60. Maximum Length of Pair Chain
```cpp
int findLongestChain(vector<vector<int>>& p) {
    sort(p.begin(), p.end(), [](auto &a, auto &b) {
        return a[1] < b[1];
    });
    int n=p.size();
    vector<int> dp(n + 1, 1);
    dp[0] = 0;
    int ans = INT_MIN;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j < i; j++)
            if (p[i - 1][0] > p[j - 1][1])
                dp[i] = max(dp[i], dp[j] + 1);
        ans = max(ans, dp[i]);
    }
    return ans;
}
```

----------------------------------------------
## 61. shortest common supersequence--> m+n-lcs(str a,str b,int m,int n)

https://practice.geeksforgeeks.org/problems/shortest-common-supersequence0322/1
```cpp
int lcs(string x,string y,int m,int n){
        if(m==0||n==0)return 0;
        int dp[m+1][n+1];
        for(int i=0;i<=m;i++){
            for(int j=0;j<=n;j++){
                if(i==0||j==0){dp[i][j]=0;continue;}
                if(x[i-1]==y[j-1])dp[i][j]=1+dp[i-1][j-1];
                else dp[i][j]=max(dp[i][j-1],dp[i-1][j]);
            }
        }
        return dp[m][n];
    }
    int shortestCommonSupersequence(char* X, char* Y, int m, int n)
    {
        return m+n-lcs(X,Y,m,n);
    }
```

or
```cpp
int shortestCommonSupersequence(string X, string Y, int m, int n) {
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int i = 0; i <= n; i++) dp[0][i] = i;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (X[i - 1] == Y[j - 1])
                dp[i][j] = 1 + dp[i - 1][j - 1];
            else
                dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j]);
        }
    }
    //printing the superseq
    int i = m, j = n;
    string ans = "";
    while (i > 0 && j > 0) {
        if (X[i - 1] == Y[j - 1]) {
            ans = X[i - 1] + ans;
            i--;
            j--;
        } else {
            if (dp[i - 1][j] < dp[i][j - 1])
                ans = X[--i] + ans;
            else
                ans = Y[--j] + ans;
        }
    }
    while (i > 0) ans = X[--i] + ans;
    while (j > 0) ans = Y[--j] + ans;
    cout << ans << endl;
    return dp[m][n];//returning the length
}
```
-----------------------------------------------
## 62. min no of insertions and deletions to convert a string into other string 
del=m-len_lcs,  ins=n-len_lcs, total = del+ins

https://practice.geeksforgeeks.org/problems/minimum-number-of-deletions-and-insertions0209/1
```cpp
int lcs(string x,string y,int m,int n){
        if(m==0||n==0)return 0;
        int dp[m+1][n+1];
        for(int i=0;i<=m;i++){
            for(int j=0;j<=n;j++){
                if(i==0||j==0){dp[i][j]=0;continue;}
                if(x[i-1]==y[j-1])dp[i][j]=1+dp[i-1][j-1];
                else dp[i][j]=max(dp[i][j-1],dp[i-1][j]);
            }
        }
        return dp[m][n];
    }
	int minOperations(string str1, string str2) 
	{ 
	    int m=str1.length();
	    int n=str2.length();
	    int lcs_len=lcs(str1,str2,m,n);
	    int del=m-lcs_len;
	    int ins=n-lcs_len;
	    return del+ins;
	}
```

-------------------------------------------------------

## 63. Buying Vegetables 

https://practice.geeksforgeeks.org/problems/buying-vegetables0016/1

Recursive 
```cpp
int util(int N, vector<int> cost[], int ind) {
    if (N == 0) return 0;
    int c = INT_MAX;
    for (int i = 0; i < 3; i++)
        if (ind != i)
            c = min(c, util(N - 1, cost, i) + cost[N - 1][i]);
    return c;
}
```

Iterative
```cpp
int minCost(int N, vector<int> cost[]) {
    // return util(N,cost,0);
    vector<vector<int>> dp(N + 1, vector<int>(3, 0));
    for (int i = 1; i <= N; i++)
        for (int j = 0; j < 3; j++)
            dp[i][j] = cost[i - 1][j] + min(dp[i - 1][(j + 1) % 3], dp[i - 1][(j + 2) % 3]);

    return *min_element(dp[N].begin(), dp[N].end());
}
```

-------------------------------------------------------

## 64. Count ways to reach the n'th stair

https://practice.geeksforgeeks.org/problems/count-ways-to-reach-the-nth-stair-1587115620/1
```cpp
int countWays(int n) {
    int mod = 1e9 + 7;
    if (n == 1) return 1;
    if (n == 2) return 2;
    int a = 1, b = 2;
    for (int i = 3; i <= n; i++) {
        int temp = b;
        b = (b + a) % mod;
        a = temp;
    }
    return b % mod;
}
```

-------------------------------------------------------

## 65. Skip the work

https://practice.geeksforgeeks.org/problems/skip-the-work5752/1
```cpp
int minAmount(int A[], int N) {
    int a = 0;
    if (N == 0) return a;
    int b = A[0];
    if (N == 1) return b;
    for (int i = 2; i <= N; i++) {
        int temp = b;
        b = min(b, a) + A[i - 1];
        a = temp;
    }
    return min(b, a);
}
```

-------------------------------------------------------

## 66. Max possible amount

https://practice.geeksforgeeks.org/problems/max-possible-amount2717/1
```cpp
int util(int arr[],int st,int end){
    if(st>end) return 0;
    if(st==end) return arr[st];
    if(dp[st][end]!=-1) return dp[st][end];
    dp[st][end]=max(
        min(util(arr,st+2,end),util(arr,st+1,end-1))+arr[st],
        min(util(arr,st,end-2),util(arr,st+1,end-1)+arr[end])
    );
    return dp[st][end];
}
vector<vector<int>>dp;
int maxAmount(int arr[], int n)
{
    dp=vector<vector<int>>(n+1,vector<int>(n+1,-1));
    return util(arr,0,n-1);
}
```

-------------------------------------------------------

## 67. Maximum calorie

https://practice.geeksforgeeks.org/problems/maximum-calorie0056/1

1) O(N) space O(N) time
```cpp
int maxCalories(int arr[], int n) {

    vector<int> sum(n + 1, 0);
    sum[0] = arr[0];
    sum[1] = arr[0] + arr[1];
    sum[2] = max(arr[0] + arr[1], max(arr[0] + arr[2], arr[2] + arr[1]));
    for (int i = 3; i < n; i++)
        sum[i] = max(sum[i - 1], max(sum[i - 2] + arr[i], sum[i - 3] + arr[i] + arr[i - 1]));
    return sum[n - 1];
}
```

2) O(1) space O(N) time
```cpp
int maxCalories(int arr[], int n) {
    int a = arr[0];
    if (n == 1) return a;
    int b = arr[0] + arr[1];
    if (n == 2) return b;
    int c = max(arr[0] + arr[1], max(arr[0] + arr[2], arr[2] + arr[1]));
    if (n == 3) return c;
    for (int i = 3; i < n; i++) {
        int temp = c;
        c = max(c, max(b + arr[i], a + arr[i] + arr[i - 1]));
        a = b;
        b = temp;
    }
    return c;
}
```

-------------------------------------------------------

## 68. Maximum path sum in matrix 

https://practice.geeksforgeeks.org/problems/path-in-matrix3805/1
```cpp
int maximumPath(int N, vector<vector<int>> Matrix)
{
    int ans=0;
    vector<vector<int>> dp(N+1,vector<int>(N+2,0));
    
    for(int i=1;i<=N;i++){
        for(int j=1;j<=N;j++){
            dp[i][j]=max(dp[i-1][j],max(dp[i-1][j+1],dp[i-1][j-1]))+Matrix[i-1][j-1];
            if(i==N) ans=max(ans,dp[N][j]);
        }
    }
    return ans;
}
```
-------------------------------------------------------

## 69. Stickler Thief

https://practice.geeksforgeeks.org/problems/stickler-theif-1587115621/1
```cpp
typedef long long int ll;
ll FindMaxSum(ll arr[], ll n) {
    ll a = arr[0];
    if (n == 1) return a;
    ll b = max(arr[0], arr[1]);
    for (int i = 2; i < n; i++) {
        ll temp = b;
        b = max(arr[i] + a, b);
        a = temp;
    }
    return b;
}
```

-------------------------------------------------------

## 70. Minimum Cost To Make Two Strings Identical

https://practice.geeksforgeeks.org/problems/minimum-cost-to-make-two-strings-identical1107/1

Recursive
```cpp
int util(string X, string Y, int costX, int costY, int x, int y) {
    if (x == 0) return y * costY;
    if (y == 0) return x * costX;
    if (dp[x][y] != -1) return dp[x][y];
    if (X[x - 1] == Y[y - 1]) return dp[x][y] = util(X, Y, costX, costY, x - 1, y - 1);
    return dp[x][y] = min(costY + util(X, Y, costX, costY, x, y - 1), costX + util(X, Y, costX, costY, x - 1, y));
}
```

Iterative
```cpp
vector<vector<int>> dp;
int findMinCost(string X, string Y, int costX, int costY) {
    int x = X.length();
    int y = Y.length();
    dp = vector<vector<int>>(x + 1, vector<int>(y + 1, 0));
    for (int i = 0; i < x; i++) dp[i + 1][0] += (i + 1) * costX;
    for (int i = 0; i < y; i++) dp[0][i + 1] += (i + 1) * costY;
    if (X[0] == Y[0]) dp[1][1] = 0;
    for (int i = 1; i <= X.length(); i++) {
        for (int j = 1; j <= Y.length(); j++) {
            if (X[i - 1] == Y[j - 1])
                dp[i][j] = dp[i - 1][j - 1];
            else
                dp[i][j] = min(costX + dp[i - 1][j], costY + dp[i][j - 1]);
        }
    }
    return dp[x][y];
}
```

-------------------------------------------------------

## 71. Number of Unique Paths

https://practice.geeksforgeeks.org/problems/number-of-unique-paths5339/1
```cpp
int NumberOfPath(int a, int b) {
    vector<vector<int>> dp(a + 1, vector<int>(b + 1, 0));
    for (int i = 1; i <= a; i++) dp[i][1] = 1;
    for (int i = 1; i <= b; i++) dp[1][i] = 1;
    for (int i = 2; i <= a; i++)
        for (int j = 2; j <= b; j++)
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
    return dp[a][b];
}
```
-------------------------------------------------------

## 72. Count number of hops

https://practice.geeksforgeeks.org/problems/count-number-of-hops-1587115620/1
```cpp
long long countWays(int n) {
    int mod = 1000000007;
    int a = 1, b = 2, c = 4;
    if (n == 1) return 1;
    if (n == 2) return 2;
    if (n == 3) return 4;
    for (int i = 4; i <= n; i++) {
        int temp = c;
        c = ((long long int)c + a + b) % mod;
        int temp2 = b;
        b = temp;
        a = temp2;
    }
    return c;
}
```