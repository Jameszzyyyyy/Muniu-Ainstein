import random
import math
import  numpy as np

from result_set import  KNNResultSet,RadiusNNResultSet

class Node:
    def __init__(self,key,value=-1):
        self.left = None
        self.right = None
        self.key =key
        self.value = value      #value可以用作储存其他数值，譬如点原来的序号

    def __str__(self):
        return "key: %s, value: %s" % (str(self.key), str(self.value))

# 为二叉树加入新的节点 adding a new node to the tree
def insert(root,key,value=-1):
    if root is None:
        root = Node(key,value)      #赋初值
    else:
        if key < root.key:
            root.left = insert(root.left,key,value)   #小数放左边
        elif key > root.key:
            root.right = insert(root.right,key,value)  #大数放右边
        else:   # don't insert if key already exist in the tree
            pass
    return  root

#二叉树的三种应用 three orders for searching the bst
def inorder(root):
    # Inorder (Left, Root, Right)： from smallest to largest
    if root is not None:
        inorder(root.left)
        print(root)
        inorder(root.right)


def preorder(root):
    # Preorder (Root, Left, Right)： search all non-leaf nodes before leaves
    if root is not None:
        print(root)
        preorder(root.left)
        preorder(root.right)


def postorder(root):
    # Postorder (Left, Right, Root)： search all leaves before non-leaf nodes
    if root is not None:
        postorder(root.left)
        postorder(root.right)
        print(root)

def knn_search(root:Node,result_set:KNNResultSet,key):
    if root is None:
        return False

    # compare the root itself
    result_set.add_point(math.fabs(root.key - key),root.value)       #计算worst_dist ,并把当前root.value(index二叉树)里的值加入到result_set 中
    if result_set.worstDist() == 0:
        return True

    if root.key >= key:
        # iterate left branch first
        if knn_search(root.left, result_set, key):
            return True
        elif math.fabs(root.key-key) < result_set.worstDist():
            return knn_search(root.right, result_set, key)
        return False
    else:
        # iterate right branch first
        if knn_search(root.right, result_set, key):
            return True
        elif math.fabs(root.key-key) < result_set.worstDist():
            return knn_search(root.left, result_set, key)
        return False

def radius_search(root: Node, result_set: RadiusNNResultSet, key):
    if root is None:
        return False

    # compare the root itself
    result_set.add_point(math.fabs(root.key - key), root.value)

    if root.key >= key:
        # iterate left branch first
        if radius_search(root.left, result_set, key):
            return True
        elif math.fabs(root.key-key) < result_set.worstDist():
            return radius_search(root.right, result_set, key)
        return False
    else:
        # iterate right branch first
        if radius_search(root.right, result_set, key):
            return True
        elif math.fabs(root.key-key) < result_set.worstDist():
            return radius_search(root.left, result_set, key)
        return False





def search_recursively(root,key):               #1NN 搜索 ，递归法
    if root is None or root.key == key:
        return root
    if key < root.key:
        return search_recursively(root.left,key)
    elif key > root.key:
        return search_recursively(root.right,key)

def search_iterative(root, key):                #1NN 搜索 ，循环判断
    current_node = root
    while current_node is not None:
        if current_node.key == key:
            return current_node
        elif key < current_node.key:
            current_node = current_node.left
        elif key > current_node.key:
            current_node = current_node.right
    return current_node



def main():
    # Data generation
    db_size = 100
    k = 5    #搜寻5个点
    radius = 2.0
    data = np.random.permutation(db_size).tolist()   #random.permutation 随机排列一个数组

    root =None
    for i,point in enumerate(data):
        root = insert(root,point,i)

    query_key = 6
    result_set = KNNResultSet(capacity=k)
    knn_search(root, result_set, query_key)
    print('kNN Search:')
    print('index - distance')
    print(result_set)

    result_set = RadiusNNResultSet(radius=radius)
    radius_search(root, result_set, query_key)
    print('Radius NN Search:')
    print('index - distance')
    print(result_set)

if __name__ == '__main__':
    main()

