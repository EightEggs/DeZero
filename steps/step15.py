# Complex compute diagram (theory)
"""
在反向传播时，由于我们总是取出 funcs 列表的最后一个元素来计算，没有考虑函数的优先级，因此导致了在复杂的计算图中某些节点被跳过计算。

为了解决这个问题，我们需要引入一个新的概念——节点的 generation。

每个节点都有一个 generation 属性，它表示该节点被创建的顺序。

当一个节点的输入节点的 generation 小于等于该节点的 generation 时，该节点才会被计算。

因此，我们可以按照 generation 的大小来遍历节点，从而确保先计算优先级较高的节点。

我们需要从 funcs 列表中找到 generation 最大的节点，并不一定要进行排序，而是使用 heapq 库中的小顶堆来实现。

优先级队列中的元素是 (-generation, func) 元组，其中 generation 表示节点的生成顺序，func 表示节点对应的函数。

我们可以按照 -generation 大小来比较元组，从而确保优先级最高的节点总是处于堆顶。
"""
