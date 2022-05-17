from turtle import screensize
import torch
import torch.nn.functional as F

def lifted_loss(score, target, margin=1):
    """
      Lifted loss, per "Deep Metric Learning via Lifted Structured Feature Embedding" by Song et al
      Implemented in `pytorch`
    """

    score = F.normalize(score, dim=1)
	
    loss = 0
    counter = 0
    
    bsz = score.size(0)
    mag = (score ** 2).sum(1).expand(bsz, bsz)
    sim = score.mm(score.transpose(0, 1))
    
    dist = (mag + mag.transpose(0, 1) - 2 * sim)
    dist = torch.nn.functional.relu(dist).sqrt()
    
    for i in range(bsz):
        t_i = target[i]
        

        for j in range(i + 1, bsz):
            
            t_j = target[j]
            
            if t_i == t_j:

                # Negative component
                # !! Could do other things (like softmax that weights closer negatives)
                l_ni = (margin - dist[i][target != t_i]).exp().sum()
                l_nj = (margin - dist[j][target != t_j]).exp().sum()
                l_n  = (l_ni + l_nj).log()
                
                # Positive component
                l_p  = dist[i,j]
                
                loss += torch.nn.functional.relu(l_n + l_p) ** 2
                counter += 1
    return loss / (2 * counter)