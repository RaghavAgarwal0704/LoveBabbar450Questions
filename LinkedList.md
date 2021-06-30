## 1. Write a Program to reverse the Linked List. (Both Iterative and recursive)

first solve using stack but then it uses extra data structure, next solve using recursion but here also extra space of stack call, finally solve iteratively.
ITERATIVE:(solved using 3 pointers, practice using 2 pointers using xor)
```cpp
struct Node* reverseList(struct Node *head)
{
      struct Node* curr=head,*next=NULL,*prev=NULL;
      while(curr!=NULL)
      {
            next=curr->next;
            curr->next=prev;
            prev=curr;
            curr=next;
      }
      return prev;
}
```
RECURSIVE: pass 2 nodes to function prev and current node and make current->next = prev. Then recursive call the function with current node as prev node. And if current node is NULL return prev node
--same as iterative 
```cpp
Node* reverseUtil(Node* prev,Node* curr){
    if(!curr) return prev;
    Node* next=curr->next;
    curr->next=prev;
    return reverseUtil(curr,next);
}
Node *reverseList(Node *head)
{
    return reverseUtil(NULL,head);
}
```

----------------------------------------------------------------------
## 2. Reverse a Linked List in group of Given Size. [Very Imp]

1) Method 1 : using stack,,time=o(n) space=o(k)
```cpp
struct node *reverse (struct node *head, int k)
{ 
    stack<node*>st;
    struct node* prev=NULL;
    struct node * curr=head;
    while(curr){
        int count=0;
        while(curr && count!=k){
            st.push(curr);
            curr=curr->next;
            count++;
        }
        while(!st.empty()){
            if(prev==NULL){
                prev=st.top();
                st.pop();
                head=prev;
            }
            else{
                prev->next=st.top();
                st.pop();
                prev=prev->next;
            }
        }
    }
    prev->next=NULL;
    return head;
}
    
```
2) Method 2 : efficient: time=o(n) space=o(1)
```cpp
Node* reverse(struct Node* head, int k)
{
    Node* prev = NULL;
    Node* curr = head;
    Node* next = NULL;
    Node* tail = NULL;
    Node* newHead = NULL;
    Node* join = NULL;
    int t = 0;
  
    // Traverse till the end of the linked list
    while (curr) {
        t = k;
        join = curr;
        prev = NULL;
  
        // Reverse group of k nodes of the linked list
        while (curr && t--) {
            next = curr->next;
            curr->next = prev;
            prev = curr;
            curr = next;
        }
  
        // Sets the new head of the input list
        if (!newHead)
            newHead = prev;
  
        /* Tail pointer keeps track of the last node 
        of the k-reversed linked list. We join the 
        tail pointer with the head of the next 
        k-reversed linked list's head */
        if (tail)
            tail->next = prev;
  
        /* The tail is then updated to the last node 
        of the next k-reverse linked list */
        tail = join;
    }
  
    /* newHead is new head of the input list */
    return newHead;
}
```
-----------OR--------------------
```cpp
struct node *reverseList(struct node *head, int k) {
    struct node *pre = NULL, *ptr = head, *temp = head;
    while (k-- && ptr) {
        temp = ptr->next;
        ptr->next = pre;
        pre = ptr;
        ptr = temp;
    }
    head->next = ptr;
    head = pre;
    return head;
}

struct node *reverse(struct node *head, int k) {
    struct node *temp = head;
    struct node *t1, *t2 = NULL;
    while (temp) {
        t1 = temp;
        temp = reverseList(temp, k);
        t2 ? t2->next = temp : head = temp;
        temp = t1->next;
        t2 = t1;
    }
    return head;
}
```

----------------------------------------------------------------------
## 3. Write a program to Detect loop in a linked list.

1) Method 1(naive) : time=space=n-->using hashmap
```cpp
bool detectLoop(Node* head){
    set<Node*>s;
    Node* curr=head;
    while(curr){
        if(s.find(curr)!=s.end())return true;
        s.insert(curr);
        curr=curr->next;
    }
    return false;
}
``` 
    
2) Method 2 :(but org list is modified here) whichever node is visited just make its next point to a temp node--> time=n space=1
```cpp
bool detectLoop(Node* head)
{
 
    // Create a temporary node
    Node* temp = new Node;
    while (head != NULL) {
 
        // This condition is for the case
        // when there is no loop
        if (head->next == NULL) {
            return false;
        }
 
        // Check if next is already
        // pointing to temp
        if (head->next == temp) {
            return true;
        }
 
        // Store the pointer to the next node
        // in order to get to it in the next step
        Node* nex = head->next;
 
        // Make next point to temp
        head->next = temp;
 
        // Get to the next node in the list
        head = nex;
    }
 
    return false;
}
``` 
 
3) Method 3 (efficient) : floyds cycle detecttion-->time=n space=1
```cpp
int detectloop(Node *head) {
    Node *slow = head;
    Node *fast = head;
    while (fast && fast->next) {
        fast = fast->next->next;
        slow = slow->next;
        if (fast == slow)
            return 1;
    }
    return 0;
}
```

----------------------------------------------------------------------
## 4. Write a program to Delete loop in a linked list.
1) Method 1 (naive) : using hashmap-->time=o(n) space=o(n)
```cpp
void removeLoop(Node* head)
    {
        Node* curr=head,*prev=NULL;
        set<Node*>s;
        while(curr){
            if(s.find(curr)!=s.end()){prev->next=NULL;break;}
            s.insert(curr);
            prev=curr;
            curr=curr->next;
        }
    }
```    
efficient: todo

----------------------------------------------------------------------
## 5. Find the starting point of the loop. 

----------------------------------------------------------------------
## 6. Remove Duplicates in a sorted Linked List.
```cpp
Node *removeDuplicates(Node *head)
{
 Node* curr=head;
 while(curr&&curr->next){
     if(curr->data==curr->next->data)
     {
            Node *temp=curr->next->next;
            free(curr->next);
            curr->next=temp;
     }
     else curr=curr->next;
 }
 return head;
}
```

----------------------------------------------------------------------
## 7. Remove Duplicates in a Un-sorted Linked List.
3 methods.
1) o(n^2) constant space:two loops - naive
```cpp
void removeDuplicates(struct Node *start)
{
    struct Node *ptr1, *ptr2, *dup;
    ptr1 = start;
 
    /* Pick elements one by one */
    while (ptr1 != NULL && ptr1->next != NULL)
    {
        ptr2 = ptr1;
 
        /* Compare the picked element with rest
           of the elements */
        while (ptr2->next != NULL)
        {
            /* If duplicate then delete it */
            if (ptr1->data == ptr2->next->data)
            {
                /* sequence of steps is important here */
                dup = ptr2->next;
                ptr2->next = ptr2->next->next;
                delete(dup);
            }
            else /* This is tricky */
                ptr2 = ptr2->next;
        }
        ptr1 = ptr1->next;
    }
}
```
2) o(nlogn) constant space (todo---merge sort on linked list)
    for this do merge sort in nlogn then apply the same method for removing duplicates from sorted list
    
3) o(n) time ,o(n) space
```cpp
void removeDuplicates(struct Node *start)
{
    // Hash to store seen values
    unordered_set<int> seen;
 
    /* Pick elements one by one */
    struct Node *curr = start;
    struct Node *prev = NULL;
    while (curr != NULL)
    {
        // If current value is seen before
        if (seen.find(curr->data) != seen.end())
        {
           prev->next = curr->next;
           delete (curr);
        }
        else
        {
           seen.insert(curr->data);
           prev = curr;
        }
        curr = prev->next;
    }
}
```

----------------------------------------------------------------------
## 8. Write a Program to Move the last element to Front in a Linked List.
```cpp
void moveToFront(Node **head_ref)  
{  
    if (*head_ref == NULL || (*head_ref)->next == NULL)  
        return;  
  
    
    Node *secLast = NULL;  
    Node *last = *head_ref;  
    while (last->next != NULL)  
    {  
        secLast = last;  
        last = last->next;  
    }
    secLast->next = NULL;
    last->next = *head_ref;
    *head_ref = last;  
} 
```

----------------------------------------------------------------------
## 9. Add “1” to a number represented as a Linked List.

1) Reversing the linked list then adding 1 to head and carry to next nodes and then reversing again.
```cpp
Node *reverse(Node *head)
{
        Node *prev=NULL,*curr=head,*next=NULL;
        while(curr)
        {
            next=curr->next;
            curr->next=prev;
            prev=curr;
            curr=next;
        }
        return prev;
}
Node *util(Node *head)
{
       int carry=1,sum=0;
       Node* curr=head,*temp;
       while(curr)
       {
           sum=carry+curr->data;
           carry=sum>9?1:0;
           sum=sum%10;
           curr->data=sum;
           temp=curr;
           curr=curr->next;
       }
       if(carry)
       {
           temp->next=new Node(carry);
       }
       return head;
}
Node* addOne(Node *head) 
{
        return(reverse(util(reverse(head))));
}
```
2) Backtracking approach : Reach the last node and increment by 1 and update carry. Then add carry to the nodes and update carry. 
```cpp
Node* util(Node* head,int &carry){
        if(!head) return head;
        head->next=util(head->next,carry);
        head->data+=(head->next==NULL)?1:carry;
        carry=head->data/10;
        head->data%=10;
        return head;
    }
Node* addOne(Node *head) 
{
    int carry=0;
    head=util(head,carry);
    if(carry==1) {
        Node* temp=head;
        head=new Node(1);
        head->next=temp;
    }
    return head;
}
```
see: https://www.geeksforgeeks.org/add-one-to-a-number-represented-as-linked-list-set-2/?ref=rp

----------------------------------------------------------------------
## 10. Add two numbers represented by linked lists.
```cpp
Node* reverseUtil(Node* pre,Node* head){
    if(!head) return pre;
    Node* next=head->next;
    head->next=pre;
    return reverseUtil(head,next);
}
Node *addTwoLists(Node *first, Node *second)
{
    int carry=0;
    Node* head=NULL;
    Node *ptr=head;
    while(first && second){
        int sum=first->data+second->data+carry;
        carry=sum/10;
        if(!head) head=new Node(sum%10),ptr=head;
        else ptr->next=new Node(sum%10),ptr=ptr->next;
        first=first->next;
        second=second->next;
    }
    while(first){
        int sum=first->data+carry;
        carry=sum/10;
        if(!head) head=new Node(sum%10),ptr=head;
        else ptr->next=new Node(sum%10),ptr=ptr->next;
        first=first->next;
    }
    while(second){
        int sum=second->data+carry;
        carry=sum/10;
        if(!head) head=new Node(sum%10),ptr=head;
        else ptr->next=new Node(sum%10),ptr=ptr->next;
        second=second->next;
    }
    if (carry) ptr->next = new Node(carry);
    return  reverseUtil(NULL,head);
}
```

----------------------------------------------------------------------
## 11. Intersection of two Sorted Linked List.
```cpp
Node* findIntersection(Node* head1, Node* head2)
{
    Node* head=NULL;
    Node* ptr=head;
    while(head1 && head2){
        if(head1->data==head2->data) {
            if(!head) head=new Node(head1->data),ptr=head;
            else ptr->next=new Node(head1->data),ptr=ptr->next;
            head1=head1->next;
            head2=head2->next;
        }
        else{
            if(head1->data<head2->data) head1=head1->next;
            else head2=head2->next;
        }
    }
    return head;
}
```

----------------------------------------------------------------------
## 12. Intersection Point of two Linked Lists.

1) Method 1: using two loops--naive time o(n*m) space o(1)
```cpp
ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        while (headA != nullptr) {
            ListNode *pB = headB;
            while (pB != nullptr) {
                if (headA == pB) return headA;
                pB = pB->next;
            }
            headA = headA->next;
        }
        return nullptr;
}

```

2) Method 2: using set--> time o(n+m) space o(n)
```cpp
ListNode *getIntersectionNode(ListNode *head1, ListNode *head2) {
        unordered_set<ListNode*>s;
        while(head1){
            s.insert(head1);
            head1=head1->next;
        }
        while(head2){
            if(s.find(head2)!=s.end())return head2;
            head2=head2->next;
        }
        return NULL;
}
```

3) Method 3: time o(m+n) space o(1)
one list is longer then other so first reach end of the shorter list next traverse move ahead the head of the longer list by that many nodes as there are extra nodes in that list. so now head1 and head2 both have same no of nodes...
```cpp
int intersectPoint(Node *head1, Node *head2) {
    Node *t1 = head1, *t2 = head2;
    while (t1 && t2)
        t1 = t1->next, t2 = t2->next;

    while (t1) t1 = t1->next, head1 = head1->next;
    while (t2) t2 = t2->next, head2 = head2->next;
    while (head1 && head2) {
        if (head1 == head2) return head1->data;
        head1 = head1->next, head2 = head2->next;
    }
    return -1;
}
```

----------------------------------------------------------------------
## 13. Merge Sort For Linked lists.[Very Important]
```cpp
Node* merge(Node* head1, Node* head2) {
    if (!head1) return head2;
    if (!head2) return head1;
    if (head1->data >= head2->data) {
        head2->next = merge(head2->next, head1);
        return head2;
    } else {
        head1->next = merge(head1->next, head2);
        return head1;
    }
    return NULL;
}

public:
Node* mergeSort(Node* head) {
    if (!head || !head->next) return head;
    Node* slow = head;
    Node* fast = head;
    while (fast && fast->next && fast->next->next) slow = slow->next, fast = fast->next->next;
    Node* head2  = slow->next;
    slow->next = NULL;
    head = mergeSort(head);
    head2 = mergeSort(head2);
    head = merge(head, head2);
    return head;
}
```

----------------------------------------------------------------------
## 14. Quicksort for Linked Lists.[Very Important]

----------------------------------------------------------------------
## 15. Find the middle Element of a linked list.

1) store all values in array and then return arr[n/2]--> space n , time n

2) find length of linked list and then run till half the length and return--> time n, space 1 but two pass

3) slow and fast pointer--> time n space 1 single pass
```cpp
ListNode* middleNode(ListNode* head) 
{
    ListNode* slow=head;
    ListNode* fast=head;
    while(fast!=nullptr && fast->next!=nullptr)
    {
        slow=slow->next;
        fast=(fast->next)->next;
    }
    return slow;
}
```
----------------------------------------------------------------------
## 16. Check if a linked list is a circular linked list.
```cpp
bool isCircular(Node *head)
{
   Node *curr=head->next;
   while(curr){
       if(curr==head)return 1;
       curr=curr->next;
   }
   return 0;
}
```

----------------------------------------------------------------------
## 17. Split a Circular linked list into two halves-->using slow and fast pointer
```cpp
void splitList(Node *head, Node **head1, Node **head2)
{
    Node * slow=head;
    Node * fast=head;
    while(fast->next!=head && fast->next->next!=head)///note this
    {
        slow=slow->next;
        fast=fast->next->next;
    }
    if(fast->next!=head)fast=fast->next;
    *head1=head;
    *head2=slow->next;
    slow->next=*head1;
    fast->next=*head2;
}
```

----------------------------------------------------------------------
## 18. Write a Program to check whether the Singly Linked list is a palindrome or not.

1) method 1: use stack--naive time o(n) space o(n)


2) method 2: time o(n) space o(1)--->instead of reversing the whole list can also reverse the second half of the list(slow and fast pointer approach) then check
```cpp
void reverse(Node **head){
    Node* curr=*head,*next=NULL,*prev=NULL;
      while(curr!=NULL)
      {
            next=curr->next;
            curr->next=prev;
            prev=curr;
            curr=next;
      }
      *head=prev;
}
bool isPalindrome(Node *head)
{
    //Your code here
    Node*curr=head;
    int n=0,t=0;
    while(curr){
        n=n*10+curr->data;
        curr=curr->next;
    }
    reverse(&head);
    curr=head;
    while(curr){
        t=t*10+curr->data;
        curr=curr->next;
    }
    reverse(&head);
    return t==n;
}
```

----------------------------------------------------------------------
## 19. Deletion from a Circular Linked List.
```cpp
void deleteNode(Node** head, int key)
{
    if (*head == NULL)
        return;
         
    // If the list contains only a single node
    if((*head)->data==key && (*head)->next==*head)
    {
        free(*head);
        *head=NULL;
        return;
    }
     
    Node *last=*head,*d;
     
    // If head is to be deleted
    if((*head)->data==key)
    {
         
        // Find the last node of the list
        while(last->next!=*head)
            last=last->next;
             
        // Point last node to the next of head i.e.
        // the second node of the list
        last->next=(*head)->next;
        free(*head);
        *head=last->next;
    }
     
    // Either the node to be deleted is not found
    // or the end of list is not reached
    while(last->next!=*head&&last->next->data!=key)
    {
        last=last->next;
    }
     
    // If node to be deleted was found
    if(last->next->data==key)
    {
        d=last->next;
        last->next=d->next;
        free(d);
    }
    else
        cout<<"no such keyfound";
}
```

----------------------------------------------------------------------
## 20. Reverse a Doubly Linked list.
1) naive: change the data of node rather than changing the nodes

2) efficient: given below
```cpp
struct Node* reverseDLL(struct Node * head)
{
    struct Node* temp=NULL,*curr=head;
    while(curr){
        temp=curr->prev;
        curr->prev=curr->next;
        curr->next=temp;
        curr=curr->prev;
    }
    if(temp)head=temp->prev;
    return head;
}
```

----------------------------------------------------------------------
## 21. Find pairs with a given sum in a sorted DLL.
 
method: same as done for array, if list is not sorted then sort it in n logn 
```cpp
void pairSum(struct Node *head, int x)
{
    // Set two pointers, first to the beginning of DLL
    // and second to the end of DLL.
    struct Node *first = head;
    struct Node *second = head;
    while (second->next != NULL)
        second = second->next;
 
    // To track if we find a pair or not
    bool found = false;
 
    // The loop terminates when two pointers
    // cross each other (second->next
    // == first), or they become same (first == second)
    while (first != second && second->next != first)
    {
        // pair found
        if ((first->data + second->data) == x)
        {
            found = true;
            cout << "(" << first->data<< ", "
                << second->data << ")" << endl;
 
            // move first in forward direction
            first = first->next;
 
            // move second in backward direction
            second = second->prev;
        }
        else
        {
            if ((first->data + second->data) < x)
                first = first->next;
            else
                second = second->prev;
        }
    }
 
    // if pair is not present
    if (found == false)
        cout << "No pair found";
}
```

----------------------------------------------------------------------
## 22. Count triplets in a sorted DLL whose sum is equal to given value “X”.

1) naive: time=o(n^3) space=o(1)

2) efficient: time=o(n^2) space=o(n)-->hashing
```cpp
int countTriplets(struct Node* head, int x)
{
    struct Node* ptr, *ptr1, *ptr2;
    int count = 0;
 
    // unordered_map 'um' implemented as hash table
    unordered_map<int, Node*> um;
 
    // insert the <node data, node pointer> tuple in 'um'
    for (ptr = head; ptr != NULL; ptr = ptr->next)
        um[ptr->data] = ptr;
 
    // generate all possible pairs
    for (ptr1 = head; ptr1 != NULL; ptr1 = ptr1->next)
        for (ptr2 = ptr1->next; ptr2 != NULL; ptr2 = ptr2->next) {
 
            // p_sum - sum of elements in the current pair
            int p_sum = ptr1->data + ptr2->data;
 
            // if 'x-p_sum' is present in 'um' and either of the two nodes
            // are not equal to the 'um[x-p_sum]' node
            if (um.find(x - p_sum) != um.end() && um[x - p_sum] != ptr1
                && um[x - p_sum] != ptr2)
 
                // increment count
                count++;
        }
 
    // required count of triplets
    // division by 3 as each triplet is counted 3 times
    return (count / 3);
}
```
3) more efficient: time=o(n^2) space=o(1)
```cpp
int countPairs(struct Node* first, struct Node* second, int value)
{
    int count = 0;
 
    // The loop terminates when either of two pointers
    // become NULL, or they cross each other (second->next
    // == first), or they become same (first == second)
    while (first != NULL && second != NULL &&
           first != second && second->next != first) {
 
        // pair found
        if ((first->data + second->data) == value) {
 
            // increment count
            count++;
 
            // move first in forward direction
            first = first->next;
 
            // move second in backward direction
            second = second->prev;
        }
 
        // if sum is greater than 'value'
        // move second in backward direction
        else if ((first->data + second->data) > value)
            second = second->prev;
 
        // else move first in forward direction
        else
            first = first->next;
    }
 
    // required count of pairs
    return count;
}
``` 
function to count triplets in a sorted doubly linked list whose sum is equal to a given value 'x'
```cpp
int countTriplets(struct Node* head, int x)
{
    // if list is empty
    if (head == NULL)
        return 0;
 
    struct Node* current, *first, *last;
    int count = 0;
 
    // get pointer to the last node of
    // the doubly linked list
    last = head;
    while (last->next != NULL)
        last = last->next;
 
    // traversing the doubly linked list
    for (current = head; current != NULL; current = current->next) {
 
        // for each current node
        first = current->next;
 
        // count pairs with sum(x - current->data) in the range
        // first to last and add it to the 'count' of triplets
        count += countPairs(first, last, x - current->data);
    }
 
    // required count of triplets
    return count;
}
```

----------------------------------------------------------------------
## 23. Sort a “k”sorted Doubly Linked list.[Very IMP]

----------------------------------------------------------------------
## 24. Rotate DoublyLinked list by N nodes.

----------------------------------------------------------------------
## 25. Rotate a Doubly Linked list in group of Given Size.[Very IMP]

----------------------------------------------------------------------
## 26. Can we reverse a linked list in less than O(n) ?

----------------------------------------------------------------------
## 27. Why Quicksort is preferred for. Arrays and Merge Sort for LinkedLists ?

----------------------------------------------------------------------
## 28. Flatten a Linked List

----------------------------------------------------------------------
## 29. Sort a LL of 0's, 1's and 2's

1) method 1: count 0s,1s,2s => then just change the data of the given linked list.

2) method 2: create three linked list and then just connect them.
```cpp
Node* segregate(Node *head) {
    
    // Add code here
    Node *zero = new Node(0);
    Node *one  = new Node(0);
    Node *two  = new Node(0);
    
    Node *zero_p = zero;
    Node *one_p  = one;
    Node *two_p  = two;
    
    Node* curr=head;
    while(curr){
        if(curr->data==0){
            zero->next=curr;
            zero=zero->next;
            curr=curr->next;
        }
        else if(curr->data==1){
            one->next=curr;
            one=one->next;
            curr=curr->next;
        }
        else{
            two->next=curr;
            two=two->next;
            curr=curr->next;
        }
    }
    head=zero_p->next;
    zero->next=one_p->next?one_p->next:two_p->next;
    one->next=two_p->next;
    two->next=nullptr;
    return head;
    
}
```

----------------------------------------------------------------------
## 30. Clone a linked list with next and random pointer

----------------------------------------------------------------------
## 31. Merge K sorted Linked list

----------------------------------------------------------------------
## 32. Multiply 2 no. represented by LL

----------------------------------------------------------------------
## 33. Delete nodes which have a greater value on right side

----------------------------------------------------------------------
## 34. Segregate even and odd nodes in a Linked List
```cpp
Node* divide(int n, Node *head){
    int e[n],o[n];
    int i=0,j=0;
    Node *curr=head;
    while(curr){
        if(curr->data%2==0){e[i++]=curr->data;}
        else o[j++]=curr->data; 
        curr=curr->next;
    }
    curr=head;
    for(int k=0;k<i;k++){curr->data=e[k];curr=curr->next;}
    for(int k=0;k<j;k++){curr->data=o[k];curr=curr->next;}
    return head;
}
```

----------------------------------------------------------------------
## 35. Program for n’th node from the end of a Linked List

1) method 1: using length of linked list--naive time n space 1 but two pass
```cpp
int getLen(Node *head){
    int count=0;
    while(head){
        count++;
        head=head->next;
    }
    //cout<<"count"<<count<<endl;
    return count;
}
int getNthFromLast(Node *head, int n)
{
       // Your code here
       int len=getLen(head);
       if(n>len)return -1;
       int l=len-n+1;
       Node *temp=head;
       while(--l){
           //cout<<l<<endl;
           temp=temp->next;
       }
       //cout<<"after"<<l<<endl;
       return temp->data;
}
```

2) method 2: using two pointers--time n space 1 single pass
```cpp
int getNthFromLast(Node *head, int n)
{
    Node *ref=head,*main=head;
    if(!head)return -1;
    int count=0;
    while(count<n){
        if(ref==NULL)return -1;
        ref=ref->next;
        count++;
    }
    while(ref){
        main=main->next;
        ref=ref->next;
    }
    return main->data;
}
```

----------------------------------------------------------------------
## 36. Find the first non-repeating character from a stream of characters

----------------------------------------------------------------------
