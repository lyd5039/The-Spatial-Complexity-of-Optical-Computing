import numpy as np

import networkx as nx
import re



def get_ports_pos(M, nx_in,ny_in,nx_out,ny_out, label_format=None):
  if M.ndim != 2:
    raise ValueError("Input matrix must be 2D.")

  dim_out, dim_in = M.shape
  if dim_in != nx_in*ny_in:
    raise ValueError("Input dimension incompatible.")
  if dim_out != nx_out*ny_out:
    raise ValueError("Output dimension incompatible.")

  xcor_in = np.linspace(1/2, nx_in-1/2, num=nx_in) - nx_in/2
  ycor_in = np.linspace(1/2, ny_in-1/2, num=ny_in) - ny_in/2
  x_grid_in, y_grid_in = np.meshgrid(xcor_in, ycor_in)

  list_names_in = []
  list_xcor_in = []
  list_ycor_in = []
  list_coordinates_in = []
  for xx in range(nx_in):
    for yy in range(ny_in):
      if label_format is None:
        list_names_in.append(f'i{yy+1},{xx+1}')
      elif label_format == 'Plotly':
        list_names_in.append(f'i<sub>{yy+1},{xx+1}</sub>')   # for matplotlib, use f'$i_{{{yy+1},{xx+1}}}$'
      elif label_format == 'matplotlib':
        list_names_in.append(f'$\mathrm{{i}}_{{{yy+1},{xx+1}}}$')
      list_xcor_in.append(x_grid_in[yy][xx])
      list_ycor_in.append(y_grid_in[yy][xx])
      list_coordinates_in.append(np.array((x_grid_in[yy][xx], y_grid_in[yy][xx]))) # DON'T flip x,y to follow the matrix layout convention


  xcor_out = np.linspace(1/2, nx_out-1/2, num=nx_out) - nx_out/2
  ycor_out = np.linspace(1/2, ny_out-1/2, num=ny_out) - ny_out/2
  x_grid_out, y_grid_out = np.meshgrid(xcor_out, ycor_out)
  list_names_out = []
  list_xcor_out = []
  list_ycor_out = []
  list_coordinates_out = []
  for xx in range(nx_out):
    for yy in range(ny_out):
      if label_format is None:
        list_names_out.append(f'o{yy+1},{xx+1}')
      elif label_format == 'Plotly':
        list_names_out.append(f'o<sub>{yy+1},{xx+1}</sub>')
      elif label_format == 'matplotlib':
        list_names_out.append(f'$\mathrm{{o}}_{{{yy+1},{xx+1}}}$')
      list_xcor_out.append(x_grid_out[yy][xx])
      list_ycor_out.append(y_grid_out[yy][xx])
      list_coordinates_out.append(np.array((x_grid_out[yy][xx], y_grid_out[yy][xx]))) # DON'T flip x,y to follow the matrix layout convention

  return list_names_in,list_xcor_in,list_ycor_in,list_coordinates_in, list_names_out,list_xcor_out,list_ycor_out,list_coordinates_out


def get_graph(M, nx_in,ny_in,nx_out,ny_out, threshold=0.01):
  """
  map an 2D matrix to a graph

  Parameters:
  M (numpy.ndarray): Input 2D NumPy array.

  Returns:
  G: graph corresponding to the linear layer
  pos: position of all nodes in the graph
  """
  dim_out, dim_in = M.shape

  list_names_in,_,_,list_coordinates_in, list_names_out,_,_,list_coordinates_out = get_ports_pos(M, nx_in,ny_in,nx_out,ny_out, label_format=None)

  G = nx.Graph()
  G.add_nodes_from(list_names_in, bipartite=0)
  G.add_nodes_from(list_names_out, bipartite=1)

  for ii in range(dim_out):
    for jj in range(dim_in):
      if np.abs(M[ii,jj]) >= threshold:
        G.add_edge(list_names_out[ii], list_names_in[jj], weight=M[ii,jj])

  pos = dict()
  pos.update(dict(zip(list_names_in, list_coordinates_in)))
  pos.update(dict(zip(list_names_out, list_coordinates_out)))

  return G,pos


def get_cross_edges_for_given_plane(G,pos, plane_point1,plane_point2):
  vec1 = (plane_point2[0] - plane_point1[0], plane_point2[1] - plane_point1[1])

  list_output_nodes = [node for node in list(G.nodes()) if re.match(r'^o\d+,\d+$', node)]
  list_input_nodes = [node for node in list(G.nodes()) if re.match(r'^i\d+,\d+$', node)]

  list_subg_nodes = []
  # check every output port
  for output_node in list_output_nodes:
    node_position = pos[output_node]
    vec2 = (node_position[0] - plane_point1[0], node_position[1] - plane_point1[1])
    cross_product = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    if cross_product > 0:
      # the port is above the line passing point1 and point2
      list_subg_nodes.append(output_node)

  # check every input port
  for input_node in list_input_nodes:
    node_position = pos[input_node]
    vec2 = (node_position[0] - plane_point1[0], node_position[1] - plane_point1[1])
    cross_product = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    if cross_product > 0:
      # the port is above the line passing point1 and point2
      list_subg_nodes.append(input_node)


  # build and analyze subgraphs
  edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
  H1 = nx.subgraph(G, list_subg_nodes)
  H2 = nx.subgraph(G, G.nodes-H1.nodes)

  # cross_edges: edges from H1 to H2
  list_cross_edges = [e for e in G.edges
            if e[0] in H1 and e[1] in H2
            or e[0] in H2 and e[1] in H1]


  # check if the cut is valid
  # if one of the subgraphs is empty, the cut is invalid
  if H1.number_of_nodes() == 0 or H2.number_of_nodes() == 0:
    cut_is_valid = False
  else:
    cut_is_valid = True

  return list_cross_edges, cut_is_valid


def mesh_input_plane_periphery(M, nx_in,ny_in,nx_out,ny_out, steps_1_port):
  """
  The periphery is meshed into 2*(nx_in+ny_in)*steps_1_port points
  """
  list_names_in,list_xcor_in,list_ycor_in,list_coordinates_in, list_names_out,list_xcor_out,list_ycor_out,list_coordinates_out = get_ports_pos(M, nx_in,ny_in,nx_out,ny_out)

  xcor_side1 = np.linspace(min(list_xcor_in)-1/2, max(list_xcor_in)+1/2, num=nx_in*steps_1_port+1)

  list_periphery_points = []
  for ii in range(len(xcor_side1)):
    list_periphery_points.append(np.array((xcor_side1[ii], min(list_ycor_in)-1/2)))

  ycor_side2 = np.linspace(min(list_ycor_in)-1/2, max(list_ycor_in)+1/2, num=ny_in*steps_1_port+1)
  for ii in range(len(ycor_side2)):
    if ii != 0:
      list_periphery_points.append(np.array((max(list_xcor_in)+1/2, ycor_side2[ii])))

  xcor_side3 = np.linspace(max(list_xcor_in)+1/2, min(list_xcor_in)-1/2, num=nx_in*steps_1_port+1)
  for ii in range(len(xcor_side3)):
    if ii != 0:
      list_periphery_points.append(np.array((xcor_side3[ii], max(list_ycor_in)+1/2)))

  ycor_side4 = np.linspace(max(list_ycor_in)+1/2, min(list_ycor_in)-1/2, num=ny_in*steps_1_port+1)
  for ii in range(len(ycor_side4)):
    if ii != 0 and ii != len(ycor_side4)-1:
      list_periphery_points.append(np.array((min(list_xcor_in)-1/2, ycor_side4[ii])))

  return list_periphery_points


def get_list_N_cross_edges(M, nx_in,ny_in,nx_out,ny_out, threshold, steps_1_port, min_cut_length):
  """
  list_N_cross_edges: a list containing
  the numbers of cross edges for ALL valid cuts
  """
  G,pos = get_graph(M, nx_in,ny_in,nx_out,ny_out, threshold)
  list_periphery_points = mesh_input_plane_periphery(M, nx_in,ny_in,nx_out,ny_out, steps_1_port)

  list_N_cross_edges = []
  # loop over all possible cuts
  for ii in range(len(list_periphery_points)):
    for jj in range(ii + 1, len(list_periphery_points)):
      point1 = list_periphery_points[ii]
      point2 = list_periphery_points[jj]

      if np.linalg.norm(point1 - point2) > min_cut_length:
      # the distance between the two points > min_cut_length

        list_cross_edges, cut_is_valid = get_cross_edges_for_given_plane(G,pos, plane_point1=point1,plane_point2=point2)

        if cut_is_valid:
          list_N_cross_edges.append(len(list_cross_edges))
  
  return list_N_cross_edges



####### functions for analyzing cones #######
def get_cross_cones_for_given_plane(G,pos, plane_point1,plane_point2):
  vec1 = (plane_point2[0] - plane_point1[0], plane_point2[1] - plane_point1[1])

  list_output_nodes = [node for node in list(G.nodes()) if re.match(r'^o\d+,\d+$', node)]
  list_input_nodes = [node for node in list(G.nodes()) if re.match(r'^i\d+,\d+$', node)]

  list_subg_nodes = []
  # check every output port
  for output_node in list_output_nodes:
    node_position = pos[output_node]
    vec2 = (node_position[0] - plane_point1[0], node_position[1] - plane_point1[1])
    cross_product = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    if cross_product > 0:
      # the port is above the line passing point1 and point2
      list_subg_nodes.append(output_node)

  # check every input port
  for input_node in list_input_nodes:
    node_position = pos[input_node]
    vec2 = (node_position[0] - plane_point1[0], node_position[1] - plane_point1[1])
    cross_product = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    if cross_product > 0:
      # the port is above the line passing point1 and point2
      list_subg_nodes.append(input_node)


  # build and analyze subgraphs
  edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
  H1 = nx.subgraph(G, list_subg_nodes)
  H2 = nx.subgraph(G, G.nodes-H1.nodes)


  list_output_nodes_with_cones_that_cross_cut = []
  # loop over all output nodes
  # check if the cone associated with each of them crosses the cut
  for output_node in list_output_nodes:
    if output_node in H1.nodes and output_node not in H2.nodes:
      for neighbor in G.neighbors(output_node):
        if neighbor in H2.nodes and neighbor in list_input_nodes:
          list_output_nodes_with_cones_that_cross_cut.append(output_node)
          break  # stop once you find one crossing edge
    elif output_node in H2.nodes and output_node not in H1.nodes:
      for neighbor in G.neighbors(output_node):
        if neighbor in H1.nodes and neighbor in list_input_nodes:
          list_output_nodes_with_cones_that_cross_cut.append(output_node)
          break  # stop once you find one crossing edge

  # check if the cut is valid
  # if one of the subgraphs is empty, the cut is invalid
  if H1.number_of_nodes() == 0 or H2.number_of_nodes() == 0:
    cut_is_valid = False
  else:
    cut_is_valid = True

  return list_output_nodes_with_cones_that_cross_cut, cut_is_valid


def mesh_half_of_input_plane_periphery(M, nx_in,ny_in,nx_out,ny_out, steps_1_port):
  """
  Half of the input plane's periphery is meshed into (nx_in+ny_in)*steps_1_port points
  """
  list_names_in,list_xcor_in,list_ycor_in,list_coordinates_in, list_names_out,list_xcor_out,list_ycor_out,list_coordinates_out = get_ports_pos(M, nx_in,ny_in,nx_out,ny_out)

  xcor_side1 = np.linspace(min(list_xcor_in)-1/2, max(list_xcor_in)+1/2, num=nx_in*steps_1_port+1)

  list_periphery_points = []
  for ii in range(len(xcor_side1)):
    list_periphery_points.append(np.array((xcor_side1[ii], min(list_ycor_in)-1/2)))

  ycor_side2 = np.linspace(min(list_ycor_in)-1/2, max(list_ycor_in)+1/2, num=ny_in*steps_1_port+1)
  for ii in range(len(ycor_side2)):
    if ii != 0 and ii != len(ycor_side2)-1:
      list_periphery_points.append(np.array((max(list_xcor_in)+1/2, ycor_side2[ii])))

  return list_periphery_points


def get_list_N_cross_cones__cut_passes_center(M, nx_in,ny_in,nx_out,ny_out, threshold, steps_1_port):
  """
  list_N_cross_cones: a list containing
  the numbers of output nodes that has at least one connecting edge (to a input) that crosses the cut

  list_cut_length: a list containing
  the lengths of the cuts' projections on the input (output) plane

  The two lists have the SAME order

  ONLY sweep over cuts that pass through the center point of the input plane
  """
  G,pos = get_graph(M, nx_in,ny_in,nx_out,ny_out, threshold)
  list_half_periphery_points = mesh_half_of_input_plane_periphery(M, nx_in,ny_in,nx_out,ny_out, steps_1_port)

  list_N_cross_cones = []
  list_cut_length = []
  # loop over all possible cuts
  #for ii in range(len(list_periphery_points)):
  #  for jj in range(ii + 1, len(list_periphery_points)):
  #    point1 = list_periphery_points[ii]
  #    point2 = list_periphery_points[jj]

  # loop over all possible cuts that pass through the center point of the input plane
  for ii in range(len(list_half_periphery_points)):
    point1 = list_half_periphery_points[ii]
    point2 = -1* point1 # this is the central symmetric point of point1 w.r.t center_point = np.array([0.0, 0.0])
    #if np.linalg.norm(point1 - point2) > min_cut_length and np.linalg.norm(point1 - point2) < max_cut_length:

    list_output_nodes_with_cones_that_cross_cut, cut_is_valid = get_cross_cones_for_given_plane(G,pos, plane_point1=point1,plane_point2=point2)

    if cut_is_valid:
      list_N_cross_cones.append(len(list_output_nodes_with_cones_that_cross_cut))
      list_cut_length.append(np.linalg.norm(point1 - point2))
  
  return list_N_cross_cones, list_cut_length