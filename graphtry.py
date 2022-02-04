from toposort import toposort

graph = {'A': ['B', 'C'],
             'B': ['C', 'D'],
             'C': ['D'],
             'E': ['F'],
             'F': ['C']}

list_toposort = list(toposort(graph))
list_toposort.reverse()
print(list_toposort)