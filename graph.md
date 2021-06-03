## 1. Create a Graph, print it

----------------------------------------------------------
## 2. Implement BFS algorithm 
```cpp
void bfs(int s, vector<int> g[], vector<bool> &vis, int N) {
    queue<int> q;
    q.push(s);
    while (!q.empty()) {
        int x = q.front();
        q.pop();
        cout << x << " ";
        for (int i = 0; i < g[x].size(); i++) {
            if (!vis[g[x][i]]) {
                q.push(g[x][i]);
                vis[g[x][i]] = true;
            }
        }
    }
}
```
----------------------------------------------------------
## 3. Implement DFS Algo 
```cpp
void dfs(int s, vector<int> g[], bool vis[]) {
    if (vis[s]) return;
    cout << s << " ";
    vis[s] = true;
    for (int i = 0; i < g[s].size(); i++) {
        if (!vis[g[s][i]])
            dfs(g[s][i], g, vis);
    }
}
```

----------------------------------------------------------
## 4. Detect Cycle in Directed Graph using BFS/DFS Algo 

1) Using DFS
```cpp
bool util(int v, vector<int> adj[], vector<bool> visited, vector<bool> done) {
    visited[v] = true;
    done[v] = true;

    for (int i : adj[v]) {
        if (!visited[i] && util(i, adj, visited, done)) {
            return true;
        } else if (done[i])
            return true;
    }
    done[v] = false;
    return false;
}
bool isCyclic(int V, vector<int> adj[]) {
    vector<bool> visited(V, false);
    vector<bool> done(V, false);
    for (int i = 0; i < V; i++)
        if (util(i, adj, visited, done)) return true;
    return false;
}
```

----------------------------------------------------------
## 5. Detect Cycle in UnDirected Graph using BFS/DFS Algo 

1) Using DFS
```cpp
bool util(int v, vector<int> adj[], vector<bool>& visited, int parent) {
    visited[v] = true;
    for (int i : adj[v]) {
        if (visited[i] && i != parent) return true;
        if (!visited[i] && util(i, adj, visited, v)) return true;
    }
    return false;
}
bool isCycle(int V, vector<int> adj[]) {
    // Code here
    vector<bool> visited(V + 1, false);
    for (int i = 0; i < V; i++)
        if (!visited[i] && util(i, adj, visited, -1)) return true;
    return false;
}
```

----------------------------------------------------------
## 6. Search in a Maze

----------------------------------------------------------
## 7. Minimum Step by Knight

----------------------------------------------------------
## 8. flood fill algo
```cpp
vector<int>dir{1,0,-1,0,1};
vector<vector<int>> util(vector<vector<int>>& image, int sr, int sc, int newColor,int prev) {
    if(sr<0 || sc<0 || sr>=image.size() || sc>=image[0].size() || image[sr][sc]!=prev) return image;
    if(image[sr][sc]==newColor) return image;
    
    int x=image[sr][sc];
    image[sr][sc]=newColor;
    for(int k=0;k<4;k++)
        image=util(image,sr+dir[k],sc+dir[k+1],newColor,x);

    // image=util(image,sr+1,sc,newColor,x);
    // image=util(image,sr,sc-1,newColor,x);
    // image=util(image,sr-1,sc,newColor,x);
    // image=util(image,sr,sc-1,newColor,x);

    return image; 
}

vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int newColor) {
    return util(image,sr,sc,newColor,image[sr][sc]); 
}
```

----------------------------------------------------------
## 9. Clone a graph

----------------------------------------------------------
## 10. Making wired Connections
```cpp
void dfs(unordered_map<int,vector<int>>&m,int s,vector<int>&vis){
    if(vis[s]) return;
    vis[s]=1;
    for(int i:m[s])
        if(!vis[i]) dfs(m,i,vis);
}

int makeConnected(int n, vector<vector<int>>& connections) {
    if(n>connections.size()+1) return -1;
    unordered_map<int,vector<int>>m;
    for(vector<int>v:connections){
        m[v[0]].push_back(v[1]);
        m[v[1]].push_back(v[0]);
    }
    int ans=-1;
    vector<int>vis(n,0);
    for(int i=0;i<n;i++)
        if(!vis[i]) dfs(m,i,vis),ans++;
    
    return ans;
}
```

----------------------------------------------------------
## 11. word Ladder 

----------------------------------------------------------
## 12. Dijkstra algo
```cpp
vector<int> dijkstra(int V, vector<vector<int>> adj[], int S) {
    vector<int> dist(V + 1, INT_MAX);
    dist[S] = 0;
    vector<bool> vis(V, false);
    for (int a = 0; a < V; a++) {
        int x = S;
        int m = INT_MAX;
        for (int j = 0; j < V; j++)
            if (!vis[j] && dist[j] < m)
                m = dist[j], x = j;

        vis[x] = true;
        for (auto i : adj[x]) {
            if (!vis[i[0]] && dist[x] + i[1] < dist[i[0]])
                dist[i[0]] = dist[x] + i[1];
        }
    }
    return dist;
}
```

----------------------------------------------------------
## 13. Implement Topological Sort 
```cpp
vector<int> ans;
void util(int v, vector<int> adj[], vector<bool>& visited) {
    visited[v] = true;
    for (int i : adj[v]) {
        if (!visited[i]) util(i, adj, visited);
    }
    ans.push_back(v);
}

vector<int> topoSort(int V, vector<int> adj[]) {
    vector<bool> visited(V, false);
    for (int i = 0; i < V; i++)
        if (!visited[i]) util(i, adj, visited);
    reverse(ans.begin(), ans.end());
    return ans;
}
```

----------------------------------------------------------
## 14. Minimum time taken by each job to be completed given by a Directed Acyclic Graph

----------------------------------------------------------
## 15. Find whether it is possible to finish all tasks or not from given dependencies

----------------------------------------------------------
## 16. Find the no. of Islands
```cpp
vector<int> dir{-1, 0, 1, 0, -1, 1, 1, -1, -1};
void util(int i, int j, vector<vector<char>>& grid) {
    if (i < 0 || j < 0 || i >= grid.size() || j >= grid[0].size() || grid[i][j] == '0') return;
    grid[i][j] = '0';
    for (int k = 0; k <= 7; k++) util(i + dir[k], j + dir[k + 1], grid);
    // util(i-1,j,grid);
    // util(i,j+1,grid);
    // util(i+1,j,grid);
    // util(i,j-1,grid);
    // util(i-1,j+1,grid);
    // util(i+1,j+1,grid);
    // util(i+1,j-1,grid);
    // util(i-1,j-1,grid);
}
int numIslands(vector<vector<char>>& grid) {
    int ans = 0;
    for (int i = 0; i < grid.size(); i++)
        for (int j = 0; j < grid[0].size(); j++)
            if (grid[i][j] == '1') {
                ans++;
                util(i, j, grid);
            }
    return ans;
}
```

----------------------------------------------------------
## 17. Given a sorted Dictionary of an Alien Language, find order of characters

----------------------------------------------------------
## 18. Implement Kruksal’sAlgorithm

----------------------------------------------------------
## 19. Implement Prim’s Algorithm

----------------------------------------------------------
## 20. Total no. of Spanning tree in a graph

----------------------------------------------------------
## 21. Implement Bellman Ford Algorithm

----------------------------------------------------------
## 22. Implement Floyd warshallAlgorithm

----------------------------------------------------------
## 23. Travelling Salesman Problem

----------------------------------------------------------
## 24. Graph ColouringProblem

----------------------------------------------------------
## 25. Snake and Ladders Problem

----------------------------------------------------------
## 26. Find bridge in a graph

----------------------------------------------------------
## 27. Count Strongly connected Components(Kosaraju Algo)

----------------------------------------------------------
## 28. Check whether a graph is Bipartite or Not

----------------------------------------------------------
## 29. Detect Negative cycle in a graph

----------------------------------------------------------
## 30. Longest path in a Directed Acyclic Graph

----------------------------------------------------------
## 31. Journey to the Moon

----------------------------------------------------------
## 32. Cheapest Flights Within K Stops

----------------------------------------------------------
## 33. Oliver and the Game

----------------------------------------------------------
## 34. Water Jug problem using BFS

----------------------------------------------------------
## 35. Water Jug problem using BFS

----------------------------------------------------------
## 36. Find if there is a path of more thank length from a source

----------------------------------------------------------
## 37. M-ColouringProblem

----------------------------------------------------------
## 38. Minimum edges to reverse o make path from source to destination

----------------------------------------------------------
## 39. Paths to travel each nodes using each edge(Seven Bridges)

----------------------------------------------------------
## 40. Vertex Cover Problem

----------------------------------------------------------
## 41. Chinese Postman or Route Inspection

----------------------------------------------------------
## 42. Number of Triangles in a Directed and Undirected Graph

----------------------------------------------------------
## 43. Minimise the cashflow among a given set of friends who have borrowed money from each other

----------------------------------------------------------
## 44. Two Clique Problem

----------------------------------------------------------