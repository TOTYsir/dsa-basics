# DSA Basics Note - 1

## Quick Sort

Code template

```cpp
void quick_sort(int q[], int l, int r) {
    if (l >= r) return; // 如果左边索引大于等于右边索引，说明数组已经有序或为空，直接返回

    int i = l - 1; // 左指针初始化为 l-1
    int j = r + 1; // 右指针初始化为 r+1
    int x = q[(l + r) >> 1]; // 选取中间元素作为分界点（基准值）

    while (i < j) {
        do i++; while (q[i] < x); // 从左向右找第一个大于等于 x 的元素
        do j--; while (q[j] > x); // 从右向左找第一个小于等于 x 的元素

        if (i < j) {
            // 如果左指针小于右指针，说明找到了一对需要交换的元素
            swap(q[i], q[j]); // 交换这两个元素的位置
        }
    }

    // 现在基准值 x 左边的元素都小于等于 x，右边的元素都大于等于 x

    quick_sort(q, l, j); // 递归排序左半部分
    quick_sort(q, j + 1, r); // 递归排序右半部分
}
```

Example - 第k个数

```cpp
#include <iostream>
using namespace std;
const int N = 1e5 + 10;
int num[N];

void quick_sort(int num[], int l, int r)
{
    if (l >= r) return;
    
    int i = l - 1, j = r + 1, x = num[l + r >> 1];
    while (i < j)
    {
        do i++; while (num[i] < x);
        do j--; while (num[j] > x);
        
        if (i < j) swap(num[i], num[j]);
    }
    
    quick_sort(num, l, j);
    quick_sort(num, j + 1, r);
}

int main()
{
    int n, k;
    scanf("%d %d\n", &n, &k);
    
    for(int i = 0; i < n; i++)
    {
        scanf("%d ", &num[i]);
    }
    
    quick_sort(num, 0, n - 1);
    
    printf("%d", num[k - 1]);
    
    return 0;
}
```

## Merge Sort

Code template

```cpp
#include <iostream>
using namespace std;

const int N = 1e5 + 10;
int num[N];  // 原始数组
int temp[N]; // 临时数组用于归并操作

// 归并排序函数
void merge_sort(int num[], int l, int r) {
    if (l >= r) return; // 递归终止条件：如果子数组大小为1或为空，直接返回
    
    int mid = l + r >> 1; // 计算中间索引
    merge_sort(num, l, mid);    // 递归排序左半部分
    merge_sort(num, mid + 1, r); // 递归排序右半部分
    
    int k = 0, i = l, j = mid + 1; // 初始化临时数组索引和左右子数组索引
    
    // 合并两个有序子数组
    while (i <= mid && j <= r) {
        if (num[i] <= num[j]) {
            temp[k++] = num[i++]; // 如果左半部分元素小于等于右半部分元素，将左半部分元素放入临时数组
        } else {
            temp[k++] = num[j++]; // 否则将右半部分元素放入临时数组
        }
    }

    // 处理剩余的元素（如果有）
    while (i <= mid) {
        temp[k++] = num[i++];
    }
    while (j <= r) {
        temp[k++] = num[j++];
    }
    
    // 将临时数组中的元素复制回原数组中
    for (i = l, j = 0; i <= r; i++, j++) {
        num[i] = temp[j];
    }
}
```

Example - 逆序对的数量

```cpp
#include <iostream>
using namespace std;
const int N = 1e5 + 10;
int num[N];
int temp[N];

long long merge_sort(int num[], int l, int r)
{
    if (l >= r)  return 0;
    
    long long output;
    int mid = (l + r) >> 1;
    
    output = merge_sort(num, l, mid) + merge_sort(num, mid + 1, r);

    
    int k = 0, i = l, j = mid + 1;
    
    while (i <= mid && j <= r)
    {
        if (num[i] <= num[j])
            temp[k++] = num[i++];
        else
        {
            temp[k++] = num[j++];
            output += mid - i + 1;
        }
    }
    
    while (i <= mid)  temp[k++] = num[i++];
    while (j <= r)  temp[k++] = num[j++];
    
    for (i = l, j = 0; i <= r; i++, j++)
    {
        num[i] = temp[j];
    }
    
    return output;
}

int main()
{
    int n;
    scanf("%d\n", &n);
    
    for(int i = 0; i < n; i++)
    {
        scanf("%d ", &num[i]);
    }
    
    long long result;
    result = merge_sort(num, 0, n - 1);
    
    printf("%llu", result);
    return 0;
}
```

## ****Binary Search****

Code template for integer

```cpp
bool check(int x) {
    // check() 函数用于检查数 x 是否满足某种性质
    // 返回 true 表示满足性质，返回 false 表示不满足
    /* ... */
}

// 区间 [l, r] 被划分成 [l, mid] 和 [mid + 1, r] 时使用：
int bsearch_1(int l, int r) {
    while (l < r) {
        int mid = l + r >> 1; // 计算中间位置
        if (check(mid)) {
            r = mid; // 如果 mid 满足性质，将右边界缩小到 mid
        } else {
            l = mid + 1; // 否则将左边界扩大到 mid + 1
        }
    }
    return l; // 返回满足性质的最小值（或第一个满足性质的位置）
}

// 区间 [l, r] 被划分成 [l, mid - 1] 和 [mid, r] 时使用：
int bsearch_2(int l, int r) {
    while (l < r) {
        int mid = l + r + 1 >> 1; // 计算中间位置（向右取整）
        if (check(mid)) {
            l = mid; // 如果 mid 满足性质，将左边界扩大到 mid
        } else {
            r = mid - 1; // 否则将右边界缩小到 mid - 1
        }
    }
    return l; // 返回满足性质的最大值（或最后一个满足性质的位置）
}
```

Code template for FP

```cpp
bool check(double x) {/* ... */} // 检查x是否满足某种性质

double bsearch_3(double l, double r)
{
    const double eps = 1e-6;   // eps 表示精度，取决于题目对精度的要求
    while (r - l > eps)
    {
        double mid = (l + r) / 2;
        if (check(mid)) r = mid;
        else l = mid;
    }
    return l;
}
```

Example - 数的范围

```cpp
#include <iostream>
using namespace std;

const int N = 1e5 + 10;
int num[N];
int check[N];

int bsearch_r(int num[], int l, int r, int x)
{
    while (l < r)
    {
        int mid = (l + r) >> 1;
        
        if (num[mid] >= x)   r = mid;
        else  l = mid + 1;
    }
    
    return l;
}

int bsearch_l(int num[], int l, int r, int x)
{
    while (l < r)
    {
        int mid = (l + r + 1) >> 1;
        
        if (num[mid] <= x)   l = mid;
        else  r = mid - 1;
    }
    
    return l;
}

int main()
{
    int n, q;
    scanf("%d %d", &n, &q);
    
    for (int i = 0; i < n; i++)
    {
        scanf("%d ", &num[i]);
    }
    
    for (int j = 0; j < q; j++)
    {
        scanf("%d\n", &check[j]);
    }
    
    for (int k = 0; k < q; k++)
    {
        int lower, upper, ans;
        lower = bsearch_r(num, 0, n - 1, check[k]);
        upper = bsearch_l(num, 0, n - 1, check[k]);
        
        if (num[lower] == check[k])
            printf("%d %d\n", lower, upper);
        else
            printf("-1 -1\n");
    }
    
    return 0;
}
```

Example - 数的三次方根

```cpp
#include <iostream>
using namespace std;

int main() {
    double x;
    cin >> x; // 输入要求平方根的数

    double l = -1e5, r = 1e5; // 初始化左边界和右边界，注意选择一个足够大的范围

    while (r - l > 1e-7) { // 当左右边界的差值小于1e-7时结束循环，达到精度要求
        double mid = (l + r) / 2; // 计算中间值

        if (mid * mid * mid >= x) {
            r = mid; // 如果 mid 的立方大于等于 x，将右边界缩小到 mid
        } else {
            l = mid; // 否则将左边界扩大到 mid
        }
    }

    printf("%.6f", l); // 输出平方根的近似值，保留六位小数

    return 0;
}
```

## 高精度

Code template for addition

```cpp
// 定义一个函数，用于实现两个非负大整数的加法
vector<int> add(vector<int> &A, vector<int> &B) {
    // 确保 A 的位数大于等于 B，如果不满足则调用交换参数顺序的函数
    if (A.size() < B.size()) return add(B, A);

    vector<int> C; // 存放结果的向量
    int t = 0; // 进位值初始化为0
    
    // 遍历 A 的每一位，同时考虑进位和 B 的对应位（如果有）
    for (int i = 0; i < A.size(); i++) {
        t += A[i]; // 加上 A 的当前位
        if (i < B.size()) t += B[i]; // 如果 B 还有位数，加上 B 的当前位
        
        C.push_back(t % 10); // 将当前位的结果加入到结果向量中
        t /= 10; // 更新进位值
    }

    if (t) C.push_back(t); // 如果最高位有进位，加入到结果中
    return C; // 返回结果向量
}
```

Example - 高精度加法

```cpp
#include <iostream>
#include <vector>
using namespace std;

// 定义一个函数，用于实现两个大整数的加法
vector<int> add(vector<int> &A, vector<int> &B) {
    vector<int> result; // 存放结果的向量
    int t = 0; // 进位值初始化为0
    
    // 遍历A和B的每一位，同时考虑进位
    for (int i = 0; i < A.size() || i < B.size(); i++) {
        if (i < A.size())  t += A[i]; // 如果A还有位数，加上A的当前位
        if (i < B.size())  t += B[i]; // 如果B还有位数，加上B的当前位
        
        result.push_back(t % 10); // 将当前位的结果加入到结果向量中
        t /= 10; // 更新进位值
    }
    
    if (t)  result.push_back(1); // 如果最高位有进位，加入到结果中
    return result; // 返回结果向量
}

int main() {
    string a, b; // 输入的两个大整数的字符串表示
    vector<int> A, B; // 存放大整数的向量
    
    cin >> a >> b; // 输入两个大整数
    
    // 将大整数的每一位倒序存入向量A和B中
    for (int i = a.size() - 1; i >= 0; i--)  A.push_back(a[i] - '0');
    for (int i = b.size() - 1; i >= 0; i--)  B.push_back(b[i] - '0');
    
    auto C = add(A, B); // 调用add函数进行加法运算，结果存入向量C
    
    // 输出结果向量C，从高位到低位
    for (int i = C.size() - 1; i >= 0; i--)  printf("%d", C[i]);
    
    return 0;
}
```

Code template for subtraction

```cpp
// C = A - B, 满足A >= B, A >= 0, B >= 0
vector<int> sub(vector<int> &A, vector<int> &B)
{
    vector<int> C;
    for (int i = 0, t = 0; i < A.size(); i ++ )
    {
        t = A[i] - t;
        if (i < B.size()) t -= B[i];
        C.push_back((t + 10) % 10);
        if (t < 0) t = 1;
        else t = 0;
    }

    while (C.size() > 1 && C.back() == 0) C.pop_back();
    return C;
}
```

Example - 高精度减法

```cpp
#include <iostream>
#include <vector>
using namespace std;

// 比较函数，用于比较两个大整数的大小
bool cmp(vector<int> &A, vector<int> &B) 
{
    if (A.size() != B.size()) return A.size() > B.size(); // 首先比较位数

    for (int i = A.size() - 1; i >= 0; i--) 
    {
        if (A[i] != B[i]) return A[i] > B[i]; // 逐位比较，从高位到低位
    }

    return true; // 如果两个数相等，返回 true
}

// 大整数减法函数
vector<int> sub(vector<int> &A, vector<int> &B) 
{
    vector<int> result; // 存放结果的向量
    int t = 0; // 借位值初始化为0

    for (int i = 0; i < A.size(); i++) 
    {
        t = A[i] - t; // 减去借位
        if (i < B.size())  t -= B[i]; // 如果B还有位数，再减去B的当前位
        result.push_back((t + 10) % 10); // 将当前位的结果加入到结果向量中

        if (t < 0) t = 1; // 根据借位情况更新借位值
        else t = 0;
    }
    
    while (result.size() > 1 && result.back() == 0)  result.pop_back(); // 去掉结果中的前导零
    
    return result; // 返回结果向量
}

int main() 
{
    string a, b; // 输入的两个大整数的字符串表示
    vector<int> A, B; // 存放大整数的向量

    cin >> a >> b; // 输入两个大整数

    // 将大整数的每一位倒序存入向量A和B中
    for (int i = a.size() - 1; i >= 0; i--) A.push_back(a[i] - '0');
    for (int i = b.size() - 1; i >= 0; i--) B.push_back(b[i] - '0');

    vector<int> C; // 存放减法结果的向量

    // 根据比较结果选择大整数进行减法运算
    if (cmp(A, B)) 
    {
        C = sub(A, B); // 如果A >= B，执行 A - B
        for (int i = C.size() - 1; i >= 0; i--) printf("%d", C[i]); // 输出结果向量C，从高位到低位
    }
    else 
    {
        C = sub(B, A); // 否则执行 B - A
        printf("-"); // 输出负号
        for (int i = C.size() - 1; i >= 0; i--)  printf("%d", C[i]); // 输出结果向量C，从高位到低位
    }

    return 0;
}
```

Code template for multiplication

```cpp
// 大整数乘法函数，计算 C = A * b，其中 A 为非负整数，b 为非负整数
vector<int> mul(vector<int> &A, int b)
{
    vector<int> C; // 存放结果的向量

    int t = 0; // 进位值初始化为0
    for (int i = 0; i < A.size() || t; i++)
    {
        if (i < A.size()) t += A[i] * b; // 如果 A 还有位数，乘以 b
        C.push_back(t % 10); // 将当前位的结果加入到结果向量中
        t /= 10; // 更新进位值
    }

    while (C.size() > 1 && C.back() == 0) C.pop_back(); // 去掉结果中的前导零

    return C; // 返回结果向量
}
```

Example - 高精度乘法

```cpp
#include <iostream>
#include <vector>

using namespace std;

// 大整数乘法函数
vector<int> mul(vector<int> &A, int b)
{
    vector<int> result; // 存放结果的向量
    int t = 0; // 进位值初始化为0
    
    // 遍历A的每一位，同时考虑进位
    for (int i = 0; i < A.size() || t; i++)
    {
        if (i < A.size())  t += A[i] * b; // 如果A还有位数，乘以b
        
        result.push_back(t % 10); // 将当前位的结果加入到结果向量中
        t /= 10; // 更新进位值
    }
    
    return result; // 返回结果向量
}

int main()
{
    string a; // 输入的大整数的字符串表示
    int b; // 输入的整数
    cin >> a >> b;

    vector<int> A;
    
    // 将大整数的每一位倒序存入向量A中
    for (int i = a.size() - 1; i >= 0; i--) A.push_back(a[i] - '0');
    
    if (b == 0)  printf("0"); // 如果b为0，直接输出0，避免前导零的出现
    else
    {
        auto C = mul(A, b); // 调用mul函数进行乘法运算，结果存入向量C
        for (int i = C.size() - 1; i >= 0; i--)  printf("%d", C[i]); // 输出结果向量C，从高位到低位
    }
    
    return 0;
}
```

Code template for division

```cpp
// 大整数除法函数，计算 A / b，并返回商 C 和余数 r
// 前提条件是 A 大于等于 0，b 大于 0
vector<int> div(vector<int> &A, int b, int &r)
{
    vector<int> C; // 存放商的向量
    r = 0; // 初始化余数为0
    
    for (int i = A.size() - 1; i >= 0; i--) // 从最高位开始遍历大整数 A
    {
        r = r * 10 + A[i]; // 将余数扩大10倍并加上当前位的值
        
        C.push_back(r / b); // 计算商并存入结果向量
        r %= b; // 计算新的余数
    }
    
    reverse(C.begin(), C.end()); // 反转结果向量，得到正确的顺序
    
    while (C.size() > 1 && C.back() == 0) C.pop_back(); // 去掉结果中的前导零
    
    return C; // 返回商的向量
}
```

Example - 高精度除法

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

// 大整数除法函数，计算 A / b，同时返回余数 r
vector<int> div(vector<int> &A, int b, int &r)
{
    vector<int> result; // 存放商的向量
    r = 0; // 初始化余数为0
    
    for (int i = A.size() - 1; i >= 0; i--)
    {
        r = A[i] + r * 10; // 将余数扩大10倍并加上当前位的值
        
        result.push_back(r / b); // 计算商并存入结果向量
        r %= b; // 计算新的余数
    }
    
    reverse(result.begin(), result.end()); // 反转结果向量，得到正确的顺序
    
    while (result.back() == 0 && result.size() > 1)  result.pop_back(); // 去掉结果中的前导零
    
    return result; // 返回商的向量
}

int main()
{
    string a; // 输入的大整数的字符串表示
    int b; // 输入的整数
    cin >> a >> b;

    vector<int> A;
    for (int i = a.size() - 1; i >= 0; i--) A.push_back(a[i] - '0'); // 将大整数的每一位倒序存入向量A中
    
    int r; // 存放余数的变量
    auto C = div(A, b, r); // 调用div函数进行除法运算，结果存入向量C，余数存入变量r
    
    for (int i = C.size() - 1; i >= 0; i--)  printf("%d", C[i]); // 输出结果向量C，从高位到低位
    printf("\n%d", r); // 输出余数
    
    return 0;
}
```

## Prefix Sum

```cpp
S[i] = a[1] + a[2] + ... a[i]
a[l] + ... + a[r] = S[r] - S[l - 1]
```

Example - 前缀和

```cpp
#include <iostream>
using namespace std;

const int N = 1e5 + 10;
int a[N], s[N];

int main()
{
    int n, m;
    scanf("%d %d", &n, &m);
    
    for (int i = 1; i <= n; i++)  
    {
        scanf("%d ", &a[i]); // 读取数组 a 的元素
        s[i] = s[i - 1] + a[i]; // 计算前缀和数组 s
    }
    
    for (int i = 0; i < m; i++)  
    {
        int l, r;
        scanf("%d %d", &l, &r); // 读取查询的区间 [l, r]
        
        printf("%d\n", s[r] - s[l - 1]); // 输出区间 [l, r] 的和
    }
    
    return 0;
}
```

Example - 子矩阵的和(前缀和矩阵)

```cpp
#include <iostream>
using namespace std;

const int N = 1010;

int a[N][N], s[N][N];

int main()
{
    int n, m, q;
    scanf("%d %d %d", &n, &m, &q); // 读取矩阵的行数、列数和查询数量

    // 读取矩阵 a 的元素
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++)
        {
            scanf("%d", &a[i][j]);
        }

    // 计算前缀和矩阵 s
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++)
        {
            // 计算 s[i][j]，表示从 (1,1) 到 (i,j) 区域的元素和
            s[i][j] = s[i - 1][j] + s[i][j - 1] - s[i - 1][j - 1] + a[i][j];
        }

    // 处理查询
    while (q--)
    {
        int x1, y1, x2, y2;
        scanf("%d %d %d %d", &x1, &y1, &x2, &y2); // 读取查询的坐标
        
        // 利用前缀和矩阵计算查询区域的和
        // 思路是总和减去两个不包含区域的部分，再加上左上角的部分
        int sum = s[x2][y2] - s[x1 - 1][y2] - s[x2][y1 - 1] + s[x1 - 1][y1 - 1];
        
        // 输出查询结果
        printf("%d\n", sum);
    }

    return 0;
}
```

## 差分

Code template

```cpp
给区间[l, r]中的每个数加上c：B[l] += c, B[r + 1] -= c
```

Example - 差分 797

```cpp
#include <iostream>
using namespace std;
const int N = 100010;

int n, m;
int a[N], b[N];

int main () {
    cin >> n >> m; // 输入 n（元素个数）和 m（操作次数）

    for (int i = 1; i <= n; i++) {
        cin >> a[i]; // 读取数组 a 的元素
        b[i] = a[i] - a[i - 1]; // 计算差分数组 b
    }

    while (m--) {
        int l, r, c;
        cin >> l >> r >> c; // 读取操作的区间 [l, r] 和增量 c
        b[l] += c; // 更新差分数组的左端点
        b[r + 1] -= c; // 更新差分数组的右端点
    }

    for (int i = 1; i <= n; i++)  
        a[i] = a[i - 1] + b[i]; // 根据差分数组更新原数组 a

    for (int i = 1; i <= n; i++)  
        cout << a[i] << ' '; // 输出更新后的数组 a

    cout << endl;

    return 0;
}
```

Code template for 2-D matrix

```cpp
给以(x1, y1)为左上角，(x2, y2)为右下角的子矩阵中的所有元素加上c：
S[x1, y1] += c, S[x2 + 1, y1] -= c, S[x1, y2 + 1] -= c, S[x2 + 1, y2 + 1] += c
```

Example - 差分矩阵 798

```cpp
#include <iostream>
using namespace std;

const int N = 1010;
int n, m, q;
int a[N][N], b[N][N];

// 定义插入函数，用于更新矩阵的某个区域
void insert(int x1, int y1, int x2, int y2, int d) {
    b[x1][y1] += d;         // 更新左上角
    b[x1][y2 + 1] -= d;     // 更新右上角
    b[x2 + 1][y1] -= d;     // 更新左下角
    b[x2 + 1][y2 + 1] += d; // 更新右下角
}

int main () {
    cin >> n >> m >> q; // 输入矩阵的大小 n 和 m，以及查询的数量 q
    
    // 读取初始矩阵元素并执行插入操作
    for (int i = 1; i <= n; i++) 
    {
        for (int j = 1; j <= m; j++) 
        {
            int x;
            cin >> x; // 读取初始矩阵元素
            insert(i, j, i, j, x); // 执行插入操作，即更新单个元素
        }
    }
    
    // 处理查询操作
    while (q--) 
    {
        int x1, y1, x2, y2, d;
        cin >> x1 >> y1 >> x2 >> y2 >> d; // 读取查询坐标和增量
        
        // 执行插入操作，即更新查询区域的元素
        insert(x1, y1, x2, y2, d);
    }
    
    // 根据插入操作后的矩阵元素计算前缀和矩阵，并输出结果
    for (int i = 1; i <= n; i++) 
    {
        for (int j = 1; j <= m; j++) 
        {
            a[i][j] = a[i - 1][j] + a[i][j - 1] - a[i - 1][j - 1] + b[i][j]; // 计算前缀和
            cout << a[i][j] << ' '; // 输出结果
        }
        cout << endl;
    }
    
    return 0;
}
```

## 双指针

Example - 最长连续不重复子序列 799

```cpp
#include <iostream>
using namespace std;

const int N = 1e5 + 10;
int a[N], s[N];

int main() {
    int n;
    int result = 1; // 初始化结果为1，至少有一个元素
    
    cin >> n; // 输入数组的长度
    for (int i = 0; i < n; i++)  cin >> a[i]; // 输入数组的元素
    
    for (int i = 0, j = 0; i < n; i++) {
        s[a[i]]++; // 统计每个元素出现的次数
        
        while (s[a[i]] > 1) {
            s[a[j]]--; // 如果当前元素重复出现，移动左指针并减少对应元素的计数
            j++;
        }
        
        result = max(result, i - j + 1); // 更新最长不重复子数组的长度
    }
    
    cout << result << endl; // 输出最长不重复子数组的长度
    
    return 0;
}
```

## Bit operation

Code template

```cpp
求n的第k位数字: n >> k & 1
返回n的最后一位1：lowbit(n) = n & -n
```

Example - 二进制中1的个数 801

```cpp
#include <iostream>
using namespace std;

const int N = 1e5 + 10;
int a[N];

// 计算一个整数 x 的二进制表示中最低位的 1 所代表的值
int lowbit(int x)
{
    return x & (-x);
}

int main()
{
    int n;
    cin >> n;
    
    for (int i = 0; i < n; i++) 
    {
        cin >> a[i];
    
        int result = 0;
        while (a[i])
        {
            a[i] -= lowbit(a[i]); // 减去最低位的 1 所代表的值
            result++; // 计数加一
        }
        
        cout << result << " "; // 输出结果
    }
    
    return 0;
}
```

## Discretization 离散化