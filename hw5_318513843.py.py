# QUESTION 1
############
class Permutation:
    def __init__(self, perm):
        def legal(perm):
            list_1 = []
            for i in range(len(perm)):
                list_1.append(0)
            for i in range(len(perm)):
                if perm[i] > len(perm) - 1:
                    return False
                if list_1[perm[i]] != 0:
                    return False
                else:
                    list_1[perm[i]] = "b"
            for i in range(len(list_1)):
                if list_1[i] == 0:
                    return False
            return True

        self.perm = None
        if legal(perm):
            self.perm = perm

    def __getitem__(self, i):
        return self.perm[i]

    def compose(self, other):
        res = []
        for i in range(len(self.perm)):
            res += [self[other[i]]]
        return Permutation(res)
    def inv(self):
       res = ["m" for i in range(len(self.perm))]
       for i in range( len(self.perm)):
            res[self[i]] = i
       return Permutation(res)


    def __eq__(self, other):
        return self.perm == other.perm

    def __ne__(self, other):
        return self.perm != other.perm

    def order(self):
        cnt = 1
        temp = self.compose(self)
        while (temp != self):
            temp = temp.compose(self)
            cnt += 1
        return cnt


p = Permutation([2, 3, 1, 0])
print(p.perm)
q = Permutation([2, 2, 1, 0])
print(q.perm)
print(p[0])
k = Permutation([1, 0, 2])
l = Permutation([0, 2, 1])
r = k.compose(l)
print(r.perm)
m = p.inv()
print(m.perm)
print(p.order())
# This function is not part of the class Permutation
def compose_list(lst):
    def compose_list_rec(mean_time_perm, lst):
        if lst == []:
            return mean_time_perm
        mean_time_perm = mean_time_perm.compose(lst[0])
        return compose_list_rec(mean_time_perm, lst[1:])
    res = compose_list_rec(lst[0], lst[1:])
    return res
p1 = Permutation([1,0,2,3])
p2 = Permutation([2,3,1,0])
p3 = Permutation([3,2,1,0])
d = compose_list([p1,p2,p3])
print(d.perm)
############
# QUESTION 2
############

def printree(t, bykey=True):
    """Print a textual representation of t
    bykey=True: show keys instead of values"""
    # for row in trepr(t, bykey):
    #        print(row)
    return trepr(t, bykey)


def trepr(t, bykey=False):
    """Return a list of textual representations of the levels in t
    bykey=True: show keys instead of values"""
    if t == None:
        return ["#"]

    thistr = str(t.key) if bykey else str(t.val)

    return conc(trepr(t.left, bykey), thistr, trepr(t.right, bykey))


def conc(left, root, right):
    """Return a concatenation of textual represantations of
    a root node, its left node, and its right node
    root is a string, and left and right are lists of strings"""

    lwid = len(left[-1])
    rwid = len(right[-1])
    rootwid = len(root)

    result = [(lwid + 1) * " " + root + (rwid + 1) * " "]

    ls = leftspace(left[0])
    rs = rightspace(right[0])
    result.append(ls * " " + (lwid - ls) * "_" + "/" + rootwid * " " + "|" + rs * "_" + (rwid - rs) * " ")

    for i in range(max(len(left), len(right))):
        row = ""
        if i < len(left):
            row += left[i]
        else:
            row += lwid * " "

        row += (rootwid + 2) * " "

        if i < len(right):
            row += right[i]
        else:
            row += rwid * " "

        result.append(row)

    return result


def leftspace(row):
    """helper for conc"""
    # row is the first row of a left node
    # returns the index of where the second whitespace starts
    i = len(row) - 1
    while row[i] == " ":
        i -= 1
    return i + 1


def rightspace(row):
    """helper for conc"""
    # row is the first row of a right node
    # returns the index of where the first whitespace ends
    i = 0
    while row[i] == " ":
        i += 1
    return i


class Tree_node():
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.left = None
        self.right = None

    def __repr__(self):
        return "(" + str(self.key) + ":" + str(self.val) + ")"


class Binary_search_tree():

    def __init__(self):
        self.root = None

    def __repr__(self):  # no need to understand the implementation of this one
        out = ""
        for row in printree(self.root):  # need printree.py file
            out = out + row + "\n"
        return out

    def lookup(self, key):
        ''' return node with key, uses recursion '''

        def lookup_rec(node, key):
            if node == None:
                return None
            elif key == node.key:
                return node
            elif key < node.key:
                return lookup_rec(node.left, key)
            else:
                return lookup_rec(node.right, key)

        return lookup_rec(self.root, key)

    def insert(self, key, val):
        ''' insert node with key,val into tree, uses recursion '''

        def insert_rec(node, key, val):
            if key == node.key:
                node.val = val  # update the val for this key
            elif key < node.key:
                if node.left == None:
                    node.left = Tree_node(key, val)
                else:
                    insert_rec(node.left, key, val)
            else:  # key > node.key:
                if node.right == None:
                    node.right = Tree_node(key, val)
                else:
                    insert_rec(node.right, key, val)
            return

        if self.root == None:  # empty tree
            self.root = Tree_node(key, val)
        else:
            insert_rec(self.root, key, val)

    def minimum(self):
        ''' return node with minimal key '''
        if self.root == None:
            return None
        node = self.root
        left = node.left
        while left != None:
            node = left
            left = node.left
        return node

    def depth(self):
        ''' return depth of tree, uses recursion'''

        def depth_rec(node):
            if node == None:
                return -1
            else:
                return 1 + max(depth_rec(node.left), depth_rec(node.right))

        return depth_rec(self.root)

    def size(self):
        ''' return number of nodes in tree, uses recursion '''

        def size_rec(node):
            if node == None:
                return 0
            else:
                return 1 + size_rec(node.left) + size_rec(node.right)

        return size_rec(self.root)

    def max_sum(self):
            def max_sum_rec(node):
                sum = 0
                if node.left == None and node.right == None:
                    return node.val
                elif node.left != None and node.right == None:
                    return node.val + max_sum_rec(node.left)
                elif node.left == None and node.right != None:
                    return node.val + max_sum_rec(node.right)
                elif node.left != None and node.right != None:
                    right_sum = max_sum_rec(node.right)
                    left_sum = max_sum_rec(node.left)
                    if right_sum <= left_sum:
                        return node.val + left_sum
                    elif right_sum > left_sum:
                        return node.val + right_sum

            if self.root == None:  # empty tree
                sum = 0
            else:
                sum = max_sum_rec(self.root)
            return sum
    def is_balanced(self):
        if self.root is None:
            return True

        def is_balanced(node):
            if node is None:
                return (0, True)
            else:
                (hightLeft, balancedLeft) = is_balanced(node.left)
                (hightRight, balancedRight) = is_balanced(node.right)
                balancedNow = balancedLeft and balancedRight
                if not balancedNow or abs(hightRight - hightLeft) > 1:
                    return (-1, False)
                hightNow = max(hightLeft, hightRight) + 1
                return (hightNow, balancedNow)

        return is_balanced(self.root)[1]

    def diam(self):
        if self.root is None:
            return 0

        def len_max(node):
            if node is None:
                return 0
            else:
                left = len_max(node.left)
                right = len_max(node.right)
                return max(left,right) + 1

        def diam_rec(node):
            if node is None:
                return 0
            if node.right is None and node.left is None:
                return 1
            return max(len_max(node.left) + len_max(node.right) +1 ,diam_rec(node.right), diam_rec(node.left))

        return diam_rec(self.root)

t = Binary_search_tree()
print(t.max_sum())
t.insert('e', 1)
t.insert('b', 2)
print(t.max_sum())

t.insert('a', 8)
t.insert('d', 4)
t.insert('c', 10)
t.insert('i', 3)
t.insert('g', 5)
t.insert('f', 7)
t.insert('h', 15)
t.insert('j', 6)
t.insert('k', 5)
print(t.max_sum())
print(t.is_balanced())
p = Binary_search_tree()
print(p.is_balanced())
p.insert("b",10)
p.insert("d",10)
p.insert("a",10)
p.insert("c",10)
print(p.is_balanced())
p.insert("h",10)
p.insert("f",10)
print(p.is_balanced())
print(p.diam())

############
# QUESTION 3
############
def same_tree(lst1, lst2):
    if len(lst1) == 1 and lst1[0]==lst2[0]:
        return True
    Lower_lst1 = []
    higher_lst1 = []
    Lower_lst2 = []
    higher_lst2 = []
    for i in range(1,len(lst1)):
        if lst1[i] < lst1[0]:
            Lower_lst1.append(lst1[i])
        else:
            higher_lst1.append(lst1[i])
    for j in range(1,len(lst2)):
         if lst2[j] < lst2[0]:
             Lower_lst2.append(lst2[j])
         else:
             higher_lst2.append(lst2[j])

    if higher_lst2 == [] and higher_lst1 == []:
        if Lower_lst1[0] != Lower_lst2[0]:
          return False
        else:
           return same_tree(Lower_lst1, Lower_lst2)
    elif Lower_lst1 == [] and Lower_lst2 == []:
        if higher_lst1[0] != higher_lst2[0]:
          return False
        else:
            return same_tree(higher_lst1,higher_lst2)
    elif Lower_lst1[0] != Lower_lst2[0] or higher_lst2[0] != higher_lst1[0]:
        return False
    else:
        return same_tree(higher_lst1, higher_lst2) and same_tree(Lower_lst1, Lower_lst2)

aba = [3,2,4,5,1,6]
dda = [3,2,4,5,6,1]
print(same_tree(aba, dda))
############
# QUESTION 4
############

class Node():

    def __init__(self, val):
        self.value = val
        self.next = None
        self.prev = None

    def __repr__(self):
        return str(self.value) + "(" + str(id(self)) + ")"
    # This shows pointers as well for educational purposes


class DLList():

    def __init__(self, seq=None):
        self.head = None
        self.tail = None
        self.len = 0
        if seq != None:
            for item in seq:
                self.insert(item)

    def __len__(self):
        return self.len

    def __repr__(self):
        out = ""
        p = self.head
        while p != None:
            out += str(p) + ", "  # str(p) envokes __repr__ of class Node
            p = p.next
        return "[" + out[:-2] + "]"

    def insert(self, val, first=False):
       node = Node(val)
       if self.head == None:
           self.tail = node
           self.head = node
       else:
         if first == True:
           tmp = self.head
           self.head = node
           node.next = tmp
           tmp.perv = node

         else:
           tmp = self. tail
           self.tali = node
           tmp.next = node
           node.perv = tmp
       self.len += 1

    def reverse(self):
       tmp = self.head
       self.head= self.tail
       meantim_node = self.head
       while meantim_node != None:
           next_node = meantim_node.perv
           prev_node = meantim_node.next
           meantim_node.next = next_node
           meantim_node.perv = prev_node
           meantim_node = next_node
       return self




    def rotate(self, k):
       if k <= self.len-k:
           for j in range(0,k):
             meantim_last = self.tail
             prev_to_meantim_last = meantim_last.prev
             self.insert(meantim_last.value,True)
             prev_to_meantim_last.next = None
             self.tail = prev_to_meantim_last
       else:
           for j in range(0,self.len - k):
               meantim_First = self.head
               next_to_meantim_First = meantim_First.next
               self.insert(meantim_First.value, False)
               next_to_meantim_First.prev = None
               self.haed = next_to_meantim_First
       return self


    def delete_node(self, node):
        if node.next != None and node.perv != None:
            next_node = node.next
            prev_node = node.prev
            next_node.prev = prev_node
            prev_node.next = next_node
        elif node.next != None and node.perv == None:
            next_node = node.next
            self.head = next_node
            next_node.prev = None
        elif node. next == None and node.prv != None:
            prev_node = node.prev
            self.tail = prev_node
            prev_node.next = None
        else:
            self.tail = None
            self.head = None
        self.len = self.len -1



p = DLList
############
# QUESTION 6
############
# a
def prefix_suffix_overlap(lst, k):
    return [(i, j)
            for i, lst[i] in enumerate(lst) for j, lst[j] in enumerate(lst)
            if lst[i]!= lst[j] and lst[i][:k] == lst[j][-k:]]
print(prefix_suffix_overlap(["a"*10,"b"*4 + "a"*6,"c"*5 + "b"*4 + "a"],5))


# c
#########################################
### Dict class ###
#########################################

class Dict:
    def __init__(self, m, hash_func=hash):
        """ initial hash table, m empty entries """
        self.table = [[] for i in range(m)]
        self.hash_mod = lambda x: hash_func(x) % m

    def __repr__(self):
        L = [self.table[i] for i in range(len(self.table))]
        return "".join([str(i) + " " + str(L[i]) + "\n" for i in range(len(self.table))])

    def insert(self, key, value):
        """ insert key,value into table
            Allow repetitions of keys """
        i = self.hash_mod(key)  # hash on key only
        item = [key, value]  # pack into one item
        self.table[i].append(item)

    def find(self, key):
        result = []
        i = self.hash_mod(key)
        for m in self.table[i]:
            if m[0] == key:  # Avoiding collisions
                result.append(m[1])
        return result
d = Dict(3)
d.insert(56, "a")
d.insert(56, "b")
print( d.find(56))

#########################################
### End Dict class ###
#########################################    

# d
def prefix_suffix_overlap_hash1(lst, k):
    res = []
    beginning = Dict(len(lst))
    for index in range(len(lst)):
        current_begin = lst[index][:k]
        beginning.insert(current_begin, index)
    for i in range(len(lst)):
        current_ending = lst[i][-k:]
        matching = beginning.find(current_ending)
        for j in matching:
            if i != j:
                if lst[j][:k] == lst[i][-k:]:
                    res.append((j, i))
    return res
print(prefix_suffix_overlap_hash1(["a"*10,"b"*4 + "a"*6,"c"*5 + "b"*4 + "a"],5))

def prefix_suffix_overlap_hash2(lst, k):
    dict = {}
    res = []
    for index in range(len(lst)):
        if lst[index][:k] in dict.keys():
            dict[lst[index][:k]].append(index)
        else:
            dict[lst[index][:k]] = [index]
    for i in range(len(lst)):
        if lst[i][-k:] in dict:
            for list_number in dict[lst[i][-k:]]:
                if list_number != i:
                    res.append((list_number, i))
    return res
print(prefix_suffix_overlap_hash2(["a"*10,"b"*4 + "a"*6,"c"*5 + "b"*4 + "a"],5))

########
# Tester
########

def test():
    # Testing Q1
    # Question 1
    p = Permutation([2, 3, 1, 0])
    if p.perm != [2, 3, 1, 0]:
        print("error in Permutation.__init__")
    q = Permutation([1, 0, 2, 4])
    if q.perm != None:
        print("error in Permutation.__init__")
    if p[0] != 2 or p[3] != 0:
        print("error in Permutation.__getitem__")

    p = Permutation([1, 0, 2])
    q = Permutation([0, 2, 1])
    r = p.compose(q)
    if r.perm != [1, 2, 0]:
        print("error in Permutation.compose")

    p = Permutation([1, 2, 0])
    invp = p.inv()
    if invp.perm != [2, 0, 1]:
        print("error in Permutation.inv")

    p1 = Permutation([1, 0, 2, 3])
    p2 = Permutation([2, 3, 1, 0])
    p3 = Permutation([3, 2, 1, 0])
    lst = [p1, p2, p3]
    q = compose_list(lst)
    if q.perm != [1, 0, 3, 2]:
        print("error in compose_list")

    identity = Permutation([0, 1, 2, 3])
    if identity.order() != 1:
        print("error in Permutation.order")
    p = Permutation([0, 2, 1])
    if p.order() != 2:
        print("error in Permutation.order")

    # Testing Q2
    # Question 2
    t = Binary_search_tree()
    if t.max_sum() != 0:
        print("error in Binary_search_tree.max_sum")
    t.insert('e', 1)
    t.insert('b', 2)
    if t.max_sum() != 3:
        print("error in Binary_search_tree.max_sum")
    t.insert('a', 8)
    t.insert('d', 4)
    t.insert('c', 10)
    t.insert('i', 3)
    t.insert('g', 5)
    t.insert('f', 7)
    t.insert('h', 9)
    t.insert('j', 6)
    t.insert('k', 5)
    if (t.max_sum() != 18):
        print("error in Binary_search_tree.max_sum")

    t = Binary_search_tree()
    if t.is_balanced() != True:
        print("error in Binary_search_tree.is_balanced")
    t.insert("b", 10)
    t.insert("d", 10)
    t.insert("a", 10)
    t.insert("c", 10)
    if t.is_balanced() != True:
        print("error in Binary_search_tree.is_balanced")
    t.insert("e", 10)
    t.insert("f", 10)
    if t.is_balanced() != False:
        print("error in Binary_search_tree.is_balanced")

    t2 = Binary_search_tree()
    t2.insert('c', 10)
    t2.insert('a', 10)
    t2.insert('b', 10)
    t2.insert('g', 10)
    t2.insert('e', 10)
    t2.insert('d', 10)
    t2.insert('f', 10)
    t2.insert('h', 10)
    if t2.diam() != 6:
        print("error in Binary_search_tree.diam")

    t3 = Binary_search_tree()
    t3.insert('c', 1)
    t3.insert('g', 3)
    t3.insert('e', 5)
    t3.insert('d', 7)
    t3.insert('f', 8)
    t3.insert('h', 6)
    t3.insert('z', 6)
    if t3.diam() != 5:
        print("error in Binary_search_tree.diam")

    # Testing Q3
    lst = DLList("abc")
    a = lst.head
    if a == None or a.next == None or a.next.next == None:
        print("error in DLList.insert")
    else:
        b = lst.head.next
        c = lst.tail
        if lst.tail.prev != b or b.prev != a or a.prev != None:
            print("error in DLList.insert")

    lst.insert("d", True)
    if len(lst) != 4 or lst.head.value != "d":
        print("error in DLList.insert")

    prev_head_id = id(lst.head)
    lst.reverse()
    if id(lst.tail) != prev_head_id or lst.head.value != "c" or lst.head.next.value != "b" or lst.tail.value != "d":
        print("error in DLList.reverse")

    lst.rotate(1)
    if lst.head.value != "d" or lst.head.next.value != "c" or lst.tail.value != "a":
        print("error in DLList.rotate")
    lst.rotate(3)
    if lst.head.value != "c" or lst.head.next.value != "b" or lst.tail.prev.value != "a":
        print("error in DLList.rotate")

    lst.delete_node(lst.head.next)
    if lst.head.next != lst.tail.prev or len(lst) != 3:
        print("error in DLList.delete_node")
    lst.delete_node(lst.tail)
    if lst.head.next != lst.tail or len(lst) != 2:
        print("error in DLList.delete_node")

    # Question 5
    s0 = "a" * 100
    s1 = "b" * 40 + "a" * 60
    s2 = "c" * 50 + "b" * 40 + "a" * 10
    lst = [s0, s1, s2]
    k = 50
    if prefix_suffix_overlap(lst, k) != [(0, 1), (1, 2)] and \
            prefix_suffix_overlap(lst, k) != [(1, 2), (0, 1)]:
        print("error in prefix_suffix_overlap")
    if prefix_suffix_overlap_hash1(lst, k) != [(0, 1), (1, 2)] and \
            prefix_suffix_overlap_hash1(lst, k) != [(1, 2), (0, 1)]:
        print("error in prefix_suffix_overlap_hash1")
    if prefix_suffix_overlap_hash2(lst, k) != [(0, 1), (1, 2)] and \
            prefix_suffix_overlap_hash2(lst, k) != [(1, 2), (0, 1)]:
        print("error in prefix_suffix_overlap_hash2")
