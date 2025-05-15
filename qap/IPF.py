import torch


def scale_row(M,  row_sum):
    tot = M.sum(0)
    D= torch.diag(row_sum/tot )
    return( M@D)

def scale_col(M, col_sum):
    tot = M.sum(1)
    D= torch.diag(col_sum/tot)
    return(D@M)

def ipf(M, num_it, row_sum = 1, col_sum = 1, eps = 1e-3  ):
    for z in range(num_it):
        row_tot = M.sum(0)
        D= torch.diag(row_sum/row_tot)
        M = M@D
        col_tot = M.sum(1)
        D= torch.diag(col_sum/col_tot)
        M = D@M
        
        if abs(row_sum - sum(row_tot)) <= eps and abs(col_sum - sum(col_tot)) <= eps:
            break
    return M
