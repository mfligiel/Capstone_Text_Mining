def graphviz(graph, edgecolor='grey', nodecolor='purple'):
  fig, ax = plt.subplots(figsize=(10, 8))
  pos = nx.spring_layout(graph, k=2)
  # Plot networks
  nx.draw_networkx(graph, pos,
                 font_size=16,
                 width=3,
                 edge_color=edgecolor,
                 node_color=nodecolor,
                 with_labels = False,
                 ax=ax)
  # Create offset labels
  for key, value in pos.items():
    x, y = value[0]+.135, value[1]+.045
    ax.text(x, y,
            s=key,
            bbox=dict(facecolor='red', alpha=0.25),
            horizontalalignment='center', fontsize=13)
    
  plt.show()
  