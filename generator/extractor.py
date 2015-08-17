__author__ = 'Blaise'
import numpy as  np

def get_rules(tree, feature_names):
     left      = tree.children_left
     right     = tree.children_right
     threshold = tree.threshold
     features  = [feature_names[i] for i in tree.feature]

     # get ids of child nodes
     #print(left)
     #print(right)
     #print(threshold)
     #print(features)
     idx = np.argwhere(left == -1)[:,0]
     #idx2 = np.argwhere(right == -1)[:,0]
     #print(idx)
     #print(idx2)
     def recurse(left, right, child, lineage=None):
          if lineage is None:
               lineage = [child]
          if child in left:
               parent = np.where(left == child)[0].item()
               split = '<='
          else:
               parent = np.where(right == child)[0].item()
               split = '>'

          lineage.append('('+str(features[parent]) + ' ' + split + ' ' + str(threshold[parent]) +')')

          if parent == 0:
               lineage.reverse()
               return lineage
          else:
               return recurse(left, right, parent, lineage)

     rules = []
     for child in idx:
          rule = '('
          value = [0,0]

          for node in recurse(left, right, child):
               if isinstance(node,int):
                    value = tree.value[node]
               else:
                   if rule != '(':
                       rule += ' and '
                   rule += node
          rule+=')'
          value = value[0]

          if value[1] >= value[0]:
            rules.append(rule)
     return rules
