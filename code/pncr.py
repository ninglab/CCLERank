import sys
import numpy as np
import argparse
import time
import pdb


def calculate_uv(U, V):
    P = np.zeros((U.shape[1], V.shape[1]))
    for i in range(U.shape[1]):
        for h in range(V.shape[1]):
            P[i, h] = U[:, i].dot(V[:, h])
    return P

def svd_ini(D, d):
    DD = np.copy(D)
    total = 0
    count = 0
    for i in range(DD.shape[0]):
        for j in range(DD.shape[1]):
            if DD[i, j] != 0 and ~np.isnan(DD[i, j]):
                count += 1
                total += DD[i, j]
        for j in range(DD.shape[1]):
            if DD[i, j] == 0 or np.isnan(DD[i, j]):
                DD[i, j] = 0
            else:
                DD[i, j] -= total / count
    U, s, V = np.linalg.svd(DD, full_matrices=True)
    V = V.T
    return (U[:, 0:d] * np.power(s[0:d], 0.5)).T,  (V[:, 0:d] * np.power(s[0:d], 0.5)).T

def split_items(D, rel, sample=False):
    rel_set = []
    nonrel_set = []
    length_set = []
    for i in range(D.shape[0]):
        rel_items = []
        nonrel_items = []
        for j in range(D.shape[1]):
            if D[i, j] != 0 and ~np.isnan(D[i, j]):
                if D[i, j] >= rel[i]:
                    rel_items.append(j)
                else:
                    nonrel_items.append(j)
        if sample == True:
            k = min(len(rel_items), len(nonrel_items))
            nonrel_items = np.random.choice(nonrel_items, size=k, replace=False)
        rel_set.append(rel_items)
        nonrel_set.append(nonrel_items)
        length_set.append([len(rel_items), len(nonrel_items)])
    return [rel_set, nonrel_set, length_set]

def find_pairs(D, rel_set, nonrel_set):
    rel_pairs = []
    rel_pairs_count = []
    for i in range(D.shape[0]):
        rel_pairs_i = []
        count = 0
        for j in rel_set[i]:
            for k in rel_set[i]:
                if D[i, j] > D[i, k]:
                    count += 1
                    rel_pairs_i.append( [ j, k ] )
        rel_pairs_count.append(count)  
        rel_pairs.append(rel_pairs_i)     
    return [rel_pairs, rel_pairs_count]

def V_accelerator(D, U, V, P, sets_acc, rel, r_uv, r_rel):
    rel_set, nonrel_set, length_set = sets_acc
    V_ppush = []
    V_order = []
    for i in range(D.shape[0]):
        useri_ppush = np.zeros(V.shape[0])
        for j in nonrel_set[i]:
            for k in rel_set[i]:
                useri_ppush += U[:, i] / (1 + np.exp(P[i, k] - P[i, j]))
        if length_set[i][0] > 0:
            useri_ppush /= length_set[i][0]
        if length_set[i][1] > 0:
            useri_ppush /= length_set[i][1]
        V_ppush.append(useri_ppush)
        useri_order = []
        count = 0
        for h in range(V.shape[1]):
            total = np.zeros(V.shape[0])
            if D[i, h] != 0 and ~np.isnan(D[i, h]):
                if D[i, h] >= rel[i]:
                    item_set = rel_set[i]
                    for j in item_set:
                        if D[i, h] > D[i, j]:
                            count += 1
                            total -= U[:, i] / (1 + np.exp(P[i, h] - P[i, j]))
                        elif D[i, h] < D[i, j]:
                            count += 1
                            total += U[:, i] / (1 + np.exp(P[i, j] - P[i, h]))
            useri_order.append(total)                          
        if count > 0:
            if count > 1:
                count /= 2
            for hh in range(V.shape[1]):
                useri_order[hh] /= count
        V_order.append(useri_order)
    return V_ppush, V_order

def get_error(D, U, V, P, sets_acc, pairs_acc, usim, rel, r_uv, r_usim, r_rel):
    start = time.time()
    rel_set, nonrel_set, length_set = sets_acc
    rel_pairs, rel_pairs_count = pairs_acc
    ppush = 0
    order = 0
    user_sim = 0
    for i in range(D.shape[0]):
        if length_set[i][0] != 0 and length_set[i][1] != 0: 
            height = 0
            rel_p = []
            for j in rel_set[i]:
                rel_p.append( P[i, j] )
            for j in nonrel_set[i]:
                height += np.sum( np.log(1 + np.exp(- (np.array(rel_p) - P[i, j]) ) ) )
            if length_set[i][0] > 0:
                height /= length_set[i][0]
            if length_set[i][1] > 0:
                height /= length_set[i][1]
            ppush += height
        total = 0
        for rp in rel_pairs[i]:
            total += np.log(1 + np.exp(- (P[i, rp[0]] - P[i, rp[1]]) ) )
        if rel_pairs_count[i] > 0:
            total /= rel_pairs_count[i]
        order += total
        for j in range(D.shape[0]):
            user_sim += usim[i, j] * np.sum(np.power(U[:, i] - U[:, j], 2))
    p1 = ppush
    p2 = order
    p3 = 0.5 * (np.sum(np.power(U, 2)) / U.shape[1] + np.sum(np.power(V, 2)) / V.shape[1])
    p4 = 0.5 * user_sim / (U.shape[1] * U.shape[1])
    error = (1.0 - r_rel) * p1 + r_rel * p2 + r_uv * p3 + r_usim * p4
    return [(1.0 - r_rel) * p1, r_rel * p2, r_uv * p3, r_usim * p4, error]
    
def gradient_U(D, U, V, P, sets_acc, pairs_acc, usim, rel, r_uv, r_usim, r_rel):
    start = time.time()
    rel_set, nonrel_set, length_set = sets_acc
    rel_pairs, rel_pairs_count = pairs_acc
    gradient_U = np.zeros((U.shape[0], U.shape[1]))
    for i in range(D.shape[0]):
        ppush = np.zeros(U.shape[0])
        for j in nonrel_set[i]:
            for k in rel_set[i]:
                ppush += (V[:, j] - V[:, k]) / (1 + np.exp(P[i, k] - P[i, j]))
        if length_set[i][0] > 0:
            ppush /= length_set[i][0]
        if length_set[i][1] > 0:
            ppush /= length_set[i][1]
        rel_order = np.zeros(U.shape[0])
        for rp in rel_pairs[i]:
            rel_order += (V[:, rp[1]] - V[:, rp[0]]) / (1 + np.exp(P[i, rp[0]] - P[i, rp[1]]))
        if rel_pairs_count[i] > 0:
            rel_order /= rel_pairs_count[i]
        order = rel_order
        user_sim = np.zeros(U.shape[0])
        for j in range(D.shape[0]):
            user_sim += usim[i, j] * (U[:, i] - U[:, j])
        gradient_U[:, i] = (1.0 - r_rel) * ppush + r_rel * order + r_uv * U[:, i] / U.shape[1] + r_usim * user_sim / (U.shape[1] * U.shape[1])
    return gradient_U

def gradient_V(D, U, V, P, sets_acc, rel, r_uv, r_rel):
    start = time.time()
    V_ppush, V_order = V_accelerator(D, U, V, P, sets_acc, rel, r_uv, r_rel)
    gradient_V = np.zeros((V.shape[0], V.shape[1]))
    for h in range(V.shape[1]):
        ppush = np.zeros(V.shape[0])
        order = np.zeros(V.shape[0])
        for i in range(D.shape[0]):
            if D[i, h] != 0 and ~np.isnan(D[i, h]):
                if D[i, h] >= rel[i]:
                    ppush -= V_ppush[i]
                else:
                    ppush += V_ppush[i]
                order += V_order[i][h]
        gradient_V[:, h] = (1.0 - r_rel) * ppush + r_rel * order + r_uv * V[:, h] / V.shape[1] 
    return gradient_V              

def pncr(D, usim, max_iter, d, rel, learning_rate, tol, r_uv, r_usim, r_rel):
    U, V = svd_ini(D, d)
    error_change = []
    P = calculate_uv(U, V)
    for i in range(max_iter):
        start = time.time()
        if i % 50 == 1:
            learning_rate *= 0.95
        sample_times = 10       
        if r_rel == 1.0:
            sample_times = 1
        sample_sets = []
        p1_before = 0
        p2_before = 0
        p3_before = 0
        p4_before = 0
        error_before = 0
        gu = np.zeros((U.shape[0], U.shape[1]))
        for j in range(sample_times):
            sets_acc = split_items(D, rel, sample=True)
            rel_set, nonrel_set, length_set = sets_acc
            pairs_acc = find_pairs(D, rel_set, nonrel_set)
            sample_sets.append([sets_acc, pairs_acc])
            sets_acc = sample_sets[j][0]
            pairs_acc = sample_sets[j][1]
            p1, p2, p3, p4, err = get_error(D, U, V, P, sets_acc, pairs_acc, usim, rel, r_uv, r_usim, r_rel)
            p1_before += p1
            p2_before += p2
            p3_before += p3
            p4_before += p4
            error_before += err
            gu += gradient_U(D, U, V, P, sets_acc, pairs_acc, usim, rel, r_uv, r_usim, r_rel)
        p1_before /= sample_times
        p2_before /= sample_times
        p3_before /= sample_times
        p4_before /= sample_times
        error_before /= sample_times
        gu /= sample_times
        U -= learning_rate * gu
        P = calculate_uv(U, V)
        gv = np.zeros((V.shape[0], V.shape[1]))
        for j in range(sample_times):
            sets_acc = sample_sets[j][0]
            pairs_acc = sample_sets[j][1]
            gv += gradient_V(D, U, V, P, sets_acc, rel, r_uv, r_rel)
        gv /= sample_times
        V -= learning_rate * gv
        P = calculate_uv(U, V)
        p1_after = 0
        p2_after = 0
        p3_after = 0
        p4_after = 0
        error_after = 0
        for j in range(sample_times):
            sets_acc = sample_sets[j][0]
            pairs_acc = sample_sets[j][1]
            p1, p2, p3, p4, err = get_error(D, U, V, P, sets_acc, pairs_acc, usim, rel, r_uv, r_usim, r_rel)
            p1_after += p1
            p2_after += p2
            p3_after += p3
            p4_after += p4
            error_after += err
        p1_after /= sample_times
        p2_after /= sample_times
        p3_after /= sample_times
        p4_after /= sample_times
        error_after /= sample_times
        error_change.append((error_before - error_after) / error_before)
        if i > 10 and abs(error_change[i]) < tol:
            break
        if i > 10 and error_change[i] < 0 and error_change[i-1] < 0 and error_change[i-2] < 0:
            break
        print 'iteration time: ', time.time() - start
    return U, V

def main():
    start = time.time()

    #######################################
    # add parameters
    #######################################
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_file')
    parser.add_argument('-sim_file')
    parser.add_argument('-rel_file')
    parser.add_argument('-max_iter')
    parser.add_argument('-d')
    parser.add_argument('-learning_rate')
    parser.add_argument('-tol')
    parser.add_argument('-alpha')
    parser.add_argument('-beta')
    parser.add_argument('-gamma')
    parser.add_argument('-train_U_file')
    parser.add_argument('-train_V_file')
    parser.add_argument('-pred_file')

    args = parser.parse_args()
    train_file         = str(args.train_file)
    sim_file           = str(args.sim_file)
    rel_file           = str(args.rel_file)
    max_iter           = int(args.max_iter)
    d                  = int(args.d)
    learning_rate      = float(args.learning_rate)
    tol                = float(args.tol)
    r_rel              = float(args.alpha)
    r_uv               = float(args.beta)
    r_usim             = float(args.gamma)
    train_U_file       = str(args.train_U_file)
    train_V_file       = str(args.train_V_file)
    pred_file          = str(args.pred_file)


    np.random.seed(0)

    # read file
    usim     = np.genfromtxt(sim_file, delimiter=',', missing_values='nan')
    D_train  = np.genfromtxt(train_file, delimiter=',', missing_values='nan')
    rel      = np.genfromtxt(rel_file, delimiter=',', missing_values='nan')
    usim_train = np.copy(usim)

    D   = -D_train
    rel = -rel
 
    # train
    U, V = pncr(D, usim_train, max_iter, d, rel, learning_rate, tol, r_uv, r_usim, r_rel)
    U = U.transpose()

    np.savetxt(train_U_file  , U        , fmt='%.6f', delimiter=',')
    np.savetxt(train_V_file  , V        , fmt='%.6f', delimiter=',')

    # predict
    pred = np.dot(U, V)

    # write output
    np.savetxt(pred_file, pred, fmt='%.6f', delimiter=',')




  
    
if __name__ == '__main__':
    main()
