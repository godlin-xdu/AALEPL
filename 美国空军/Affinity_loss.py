import torch as t
import torch
import ipdb
from Affinity.affinity import *
def affinity_loss_confidence(labels,
                  probs,
                  confidence_mx,
                  gate,
                  sigma,
                  conv_size = 4,                 
                  num_classes=21,
                  kld_margin=3):
  """Affinity Field (AFF) loss.

  This function computes AFF loss. There are several components in the


  function:
  1) extracts edges from the ground-truth labels.
  2) extracts ignored pixels and their paired pixels (the neighboring
     pixels on the eight corners).

  3) extracts neighboring pixels on the eight corners from a 3x3 patch.

  4) computes KL-Divergence between center pixels and their neighboring
     pixels from the eight corners.

  Args:


    labels: A tensor of size [batch_size, height_in, width_in], indicating 
      semantic segmentation ground-truth labels.
    probs: A tensor of size [batch_size, height_in, width_in, num_classes],
      indicating segmentation predictions.


    num_classes: A number indicating the total number of valid classes.
    kld_margin: A number indicating the margin for KL-Divergence at edge.
  Returns:
    Two 1-D tensors value indicating the loss at edge and non-edge.
  """
  # Compute ignore map (e.g, label of 255 and their paired pixels).
  #ipdb.set_trace()
  
  labels = t.squeeze(labels, dim=-1) # NxHxW
  labels_bg = labels.clone()
  labels_bg[labels_bg > 0] = 255
  labels_oj = labels.clone()
  labels_oj[labels_oj == 0] = 255
  
  ignore = ignores_from_label(labels, num_classes, conv_size) # NxHxWx8
  not_ignore = ~ignore
  not_ignore = t.unsqueeze(not_ignore, dim=3) # NxHxWx1x8

  ignore_oj = ignores_from_label(labels_oj, num_classes, conv_size) # NxHxWx8
  not_ignore_oj = ~ignore_oj
  not_ignore_oj = t.unsqueeze(not_ignore_oj, dim=3)

  ignore_bg = ignores_from_label(labels_bg, num_classes, conv_size) # NxHxWx8
  not_ignore_bg = ~ignore_bg
  not_ignore_bg = t.unsqueeze(not_ignore_bg, dim=3)
  labels = t.unsqueeze(labels,dim = 3)
  labels_oj = t.unsqueeze(labels_oj,dim = 3)

  labels_bg = t.unsqueeze(labels_bg,dim = 3)



  edge = edges_from_label(labels, conv_size, 255) # NxHxWxCx8

  edge = edge&not_ignore#连通边且不为255

  edge_oj = edges_from_label(labels_oj,conv_size,255)
  
  edge_oj = edge_oj&not_ignore

  edge_bg = edge - edge_oj
  

  not_edge = (~edge)&not_ignore

  not_edge_bg = (~edge)&not_ignore_bg
  not_edge_oj = (~edge)&not_ignore_oj
  not_edge_bg_indices = t.nonzero(not_edge_bg.view(-1))
  not_edge_oj_indices = t.nonzero(not_edge_oj.view(-1))

  edge_indices = t.nonzero(edge.view(-1))

  edge_oj_indices = t.nonzero(edge_oj.view(-1))

  edge_bg_indices = t.nonzero(edge_bg.view(-1))

  not_edge_indices = t.nonzero(not_edge.view(-1))

  # Extract eight corner from the center in a patch as paired pixels.
  probs_paired = eightcorner_activation(probs, conv_size)  # NxHxWxCx8 
  probs = probs.permute(0,2,3,1)
  probs = t.unsqueeze(probs, dim=4) # NxHxWxCx1

  neg_probs = t.clamp(1-probs, 1e-4, 1.0)
  probs = t.clamp(probs, 1e-4, 1.0)
  neg_probs_paired= t.clamp(1-probs_paired, 1e-4, 1.0)
  probs_paired = t.clamp(probs_paired, 1e-4, 1.0)

  # Compute KL-Divergence.
  kldiv = probs_paired*t.log(probs_paired/probs)

  kldiv += neg_probs_paired*t.log(neg_probs_paired/neg_probs)
  kldiv = kldiv.sum(dim = 3, keepdim = True)

  #min-max
  confidence_mx_paired = eightcorner_activation(confidence_mx.unsqueeze(1), conv_size)#NxHxWxCx1
  N,H,W,C1,C2 = confidence_mx_paired.size()
  confidence_mx = confidence_mx.unsqueeze(3).unsqueeze(4).float()
  confidence_mx_expand = confidence_mx.expand(N,H,W,C1,C2)
  confidence_mx_stack = t.stack([confidence_mx_paired.float(),confidence_mx_expand],dim = 0)
  confidence_min,_ = t.min(confidence_mx_stack,dim=0)
  confidence_max,_ = t.max(confidence_mx_stack,dim=0)

  kldiv_confidence_max = kldiv*confidence_max

  if len(not_edge_indices) != 0 and len(edge_indices) != 0:
      
      not_edge_loss = kldiv_confidence_max
      not_edge_loss = not_edge_loss.view(-1)

      if len(not_edge_bg_indices) !=0 and len(not_edge_oj_indices) !=0 :
          not_edge_bg_loss = not_edge_loss[not_edge_bg_indices]
          not_edge_oj_loss = not_edge_loss[not_edge_oj_indices]
          not_edge_loss = t.mean(not_edge_bg_loss) + t.mean(not_edge_oj_loss)
      elif len(not_edge_bg_indices) !=0:
          not_edge_bg_loss = not_edge_loss[not_edge_bg_indices]
          not_edge_loss = t.mean(not_edge_bg_loss)

      else:
          not_edge_oj_loss = not_edge_loss[not_edge_oj_indices]
          not_edge_loss = t.mean(not_edge_oj_loss)

      z = t.zeros(1).cuda()

      if len(edge_bg_indices) !=0 and len(edge_oj_indices) !=0 :
          edge_bg_loss = t.max(z, kld_margin*confidence_max-kldiv)
          edge_bg_loss = edge_bg_loss.view(-1)
          edge_bg_loss = edge_bg_loss[edge_bg_indices]

          edge_oj_loss = t.max(z, kld_margin*confidence_max-kldiv)
          edge_oj_loss = edge_oj_loss.view(-1)
          edge_oj_loss = edge_oj_loss[edge_oj_indices]

          edge_loss = t.mean(edge_oj_loss) + t.mean(edge_bg_loss)
         
      elif len(edge_bg_indices) !=0:
          edge_bg_loss = t.max(z, kld_margin*confidence_max-kldiv)
          edge_bg_loss = edge_bg_loss.view(-1)
          edge_bg_loss = edge_bg_loss[edge_bg_indices]
          edge_loss = t.mean(edge_bg_loss)
      else:
          edge_oj_loss = t.max(z, kld_margin*confidence_max-kldiv)
          edge_oj_loss = edge_oj_loss.view(-1)
          edge_oj_loss = edge_oj_loss[edge_oj_indices]

          edge_loss = t.mean(edge_oj_loss)

      loss = not_edge_loss + 2*edge_loss

      return loss
  elif len(edge_indices) != 0:

      z = t.zeros(1).cuda()


      if len(edge_bg_indices) !=0 and len(edge_oj_indices) !=0 :
          edge_bg_loss = t.max(z, kld_margin*confidence_max-kldiv)
          edge_bg_loss = edge_bg_loss.view(-1)

          edge_bg_loss = edge_bg_loss[edge_bg_indices]

          edge_oj_loss = t.max(z, kld_margin*confidence_max-kldiv)
          edge_oj_loss = edge_oj_loss.view(-1)

          edge_oj_loss = edge_oj_loss[edge_oj_indices]

          edge_loss = t.mean(edge_oj_loss) + t.mean(edge_bg_loss)
      elif len(edge_bg_indices) !=0:
          edge_bg_loss = t.max(z, kld_margin*confidence_max-kldiv)
          edge_bg_loss = edge_bg_loss.view(-1)
          edge_bg_loss = edge_bg_loss[edge_bg_indices]

          edge_loss = t.mean(edge_bg_loss)
      else:

          edge_oj_loss = t.max(z, kld_margin*confidence_max-kldiv)
          edge_oj_loss = edge_oj_loss.view(-1)
          edge_oj_loss = edge_oj_loss[edge_oj_indices]

          edge_loss = t.mean(edge_oj_loss)

      loss = 2*edge_loss

      return loss
  elif len(not_edge_indices) != 0:
      not_edge_loss = kldiv_confidence_max
      not_edge_loss = not_edge_loss.view(-1)

      if len(not_edge_bg_indices) !=0 and len(not_edge_oj_indices) !=0 :
          not_edge_bg_loss = not_edge_loss[not_edge_bg_indices]
          not_edge_oj_loss = not_edge_loss[not_edge_oj_indices]
          not_edge_loss = t.mean(not_edge_bg_loss) + t.mean(not_edge_oj_loss)

          loss = not_edge_loss

      else:
          loss = t.zeros(1).cuda().detach()
      
      return loss
  else:
      loss = t.zeros(1).cuda().detach()
      return loss
