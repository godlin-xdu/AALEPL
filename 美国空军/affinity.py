import torch as t
import torch.nn.functional as F



def eightcorner_activation(x, size):
  """Retrieves neighboring pixels one the eight corners from a
  (2*size+1)x(2*size+1) patch.

  Args:
    x: A tensor of size [batch_size, height_in, width_in, channels]
    size: A number indicating the half size of a patch.

  Returns:
    A tensor of size [batch_size, channels,height_in, width_in, 8]
  """
  # Get the number of channels in the input.
  shape_x = x.size()
  if len(shape_x) != 4:
    raise ValueError('Only support for 4-D tensors!')
  n,c, h, w = shape_x
  x = x.permute(0,2,3,1)
  # Pad at the margin.
  p = size
  x_pad = F.pad(x,
                 pad=(0,0,size,size,size,size),
                 mode='constant',
                 value=0)

  # Get eight corner pixels/features in the patch.
  x_groups = []
  for st_y in range(0,2*size+1,size):
    for st_x in range(0,2*size+1,size):
      if st_y == size and st_x == size:
        # Ignore the center pixel/feature.
        continue

      x_neighbor = x_pad[:, st_y:st_y+h, st_x:st_x+w, :]
      x_groups.append(x_neighbor)

  output = [t.unsqueeze(c, dim=4) for c in x_groups]
  output = t.cat(output, dim=4)
  
  return output


def ignores_from_label(labels, num_classes, size):
  """Retrieves ignorable pixels from the ground-truth labels.

  This function returns a binary map in which 1 denotes ignored pixels
  and 0 means not ignored ones. For those ignored pixels, they are not
  only the pixels with label value >= num_classes, but also the
  corresponding neighboring pixels, which are on the the eight cornerls
  from a (2*size+1)x(2*size+1) patch.
  
  Args:
    labels: A tensor of size [batch_size, height_in, width_in], indicating 
      semantic segmentation ground-truth labels.
    num_classes: A number indicating the total number of valid classes. The 
      labels ranges from 0 to (num_classes-1), and any value >= num_classes
      would be ignored.
    size: A number indicating the half size of a patch.

  Return:
    A tensor of size [batch_size, height_in, width_in, 8]
  """
  # Get the number of channels in the input.
  shape_lab = labels.size()
  if len(shape_lab) != 3:
    raise ValueError('Only support for 3-D label tensors!')
  n, h, w = shape_lab

  # Retrieve ignored pixels with label value >= num_classes.
  ignore = t.gt(labels, num_classes-1) # NxHxW

  # Pad at the margin.
  p = size
  #print(ignore.size())
  ignore_pad = F.pad(ignore,
                      pad=(size,size,size,size),
                      mode='constant',
                      value=1)
  #print(ignore_pad.size())
  # Retrieve eight corner pixels from the center, where the center
  # is ignored. Note that it should be bi-directional. For example,
  # when computing AAF loss with top-left pixels, the ignored pixels
  # might be the center or the top-left ones.
  ignore_groups= []


  
  for st_y in range(0,2*size+1,size):
    for st_x in range(0,2*size+1,size):
      if st_y == size and st_x == size:
        continue
      ignore_neighbor = ignore_pad[:,st_y:st_y+h,st_x:st_x+w]
      mask = ignore_neighbor|ignore
      ignore_groups.append(mask)
      

  ignore_groups = [
    t.unsqueeze(c, dim=3) for c in ignore_groups
  ] # NxHxWx1
  ignore = t.cat(ignore_groups, dim=3) #NxHxWx8

  return ignore

def edges_from_label(labels, size, ignore_class=255):
  """Retrieves edge positions from the ground-truth labels.

  This function computes the edge map by considering if the pixel values
  are equal between the center and the neighboring pixels on the eight
  corners from a (2*size+1)*(2*size+1) patch. Ignore edges where the any
  of the paired pixels with label value >= num_classes.

  Args:
    labels: A tensor of size [batch_size, height_in, width_in], indicating 
      semantic segmentation ground-truth labels.
    size: A number indicating the half size of a patch.
    ignore_class: A number indicating the label value to ignore.

  Return:
    A tensor of size [batch_size, height_in, width_in, 1, 8]
  """
  # Get the number of channels in the input.
  shape_lab = labels.size()
  if len(shape_lab) != 4:
    raise ValueError('Only support for 4-D label tensors!')
  n, h, w, c = shape_lab

  # Pad at the margin.
  p = size
  labels_pad = F.pad(
    labels, pad=(0,0,size,size,size,size),
    mode='constant',
    value=ignore_class)

  # Get the edge by comparing label value of the center and it paired pixels.
  edge_groups= []
  for st_y in range(0,2*size+1,size):
    for st_x in range(0,2*size+1,size):
      if st_y == size and st_x == size:
        continue
      labels_neighbor = labels_pad[:,st_y:st_y+h,st_x:st_x+w]
      #print(labels_neighbor.size(),labels.size())
      edge = t.ne(labels_neighbor, labels)
      edge_groups.append(edge)

  edge_groups = [
    t.unsqueeze(c, dim=4) for c in edge_groups
  ] # NxHxWx1x1
  edge = t.cat(edge_groups, dim=4) #NxHxWx1x8

  return edge
