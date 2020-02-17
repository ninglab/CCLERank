import sys
import numpy as np
import argparse

def remove_nonrated(D):
    R = []
    for i in range(D.shape[0]):
        r = []
        for j in range(D.shape[1]):
            if ~np.isnan(D[i, j]) and D[i, j] != 0:
                r.append(D[i, j])
        R.append(r)
    return np.array(R)

def sort_by_s_desc(D, S):
    ind = np.argsort(-S, axis=1)
    X = np.zeros((D.shape[0], D.shape[1]))
    for i in range(D.shape[0]):
        X[i, :] = D[i, ind[i, :]]
    return remove_nonrated(X)

def average_precision(D, S, k, rel):
    acc = []
    X = sort_by_s_desc(D, S)
    for i in range(D.shape[0]):
        kk = min(k, len(X[i]))
        if kk == 0:
            continue
        elif np.sum(np.count_nonzero(X[i] >= rel[i])) == 0:
            continue
        else:
            apk = []
            for r in range(kk):
                if (X[i][r] >= rel[i]):
                    pr = np.sum(np.count_nonzero(X[i][0:r+1] >= rel[i])) / (r+1.0)
                    apk.append(pr)
            if len(apk) > 0:
                acc.append(np.mean(apk))
            else:
                acc.append(0)
    return np.mean(acc)

def average_hit(D, S, k, rel):
    acc = []
    X = sort_by_s_desc(D, S)
    for i in range(D.shape[0]):
        kk = min(k, len(X[i]))
        if kk == 0:
            continue
        #elif np.sum(np.count_nonzero(X[i] >= rel[i])) == 0:
        #    continue
        else:
            row = np.asarray(X[i])
            n_hit = np.sum(row[0:kk] >= rel[i])
            acc.append(n_hit)
            
    return np.mean(acc)



def best_average_precision(D, rel, k):
    apk = []
    D = sort_by_s_desc(D, D)
    for i in range(D.shape[0]):
        kk = min(k, len(D[i]))
        if kk == 0:
            continue
        elif np.sum(np.count_nonzero(D[i] >= rel[i])) == 0:
            continue
        else:
            num_rel = np.sum(np.count_nonzero(D[i][0:kk] >= rel[i]))
            apk.append(1 if num_rel > 0 else 0)
    return np.mean(apk)

def c_index(D, S):
    acc = []
    X = sort_by_s_desc(D, S)
    for u in range(D.shape[0]):
        if len(X[u]) == 0:
            continue
        elif len(X[u]) == 1:
            acc.append(1)
        else:
            total = 0
            correct = 0
            for i in range( len(X[u]) ):
                for j in range(0, i):
                    total += 1
                    if X[u][i] <= X[u][j]:
                        correct += 1
            acc.append(correct / float(total))
    return np.mean(acc)

def c_index_rel(D, S, rel):
    acc = []
    X = sort_by_s_desc(D, S)
    for u in range(D.shape[0]):
        if len(X[u]) == 0:
            continue
        else:
            rel_items = []
            for i in range( len(X[u])):
                if X[u][i] >= rel[u]:
                    rel_items.append(i)
            if len(rel_items) == 0:
                continue
            elif len(rel_items) == 1:
                acc.append(1)
                continue
            else:
                total = 0
                correct = 0
                for i in range(len(rel_items)):
                    for j in range(0, i):
                        total += 1
                        if X[u][rel_items[i]] <= X[u][rel_items[j]]:
                            correct += 1
                acc.append(correct / float(total))
    
    return np.mean(acc)


def rmse(D, S, rel):
    n_rel = 0.0
    n_all = 0.0
    sqrt_error_rel = []
    sqrt_error_all = []

    for u in range(len(D)):
        n_rel = 0.0
        n_all = 0.0
        current_sq_error_all = 0.0
        current_sq_error_rel = 0.0
        for i in range(len(D[u])):
            if D[u][i] != 0.0 and ~np.isnan(D[u][i]):
                current_sq_error_all += (D[u][i] - S[u][i])**2
                n_all += 1
                if D[u][i] >= rel[u]:
                    current_sq_error_rel += (D[u][i] - S[u][i])**2
                    n_rel += 1
        if n_rel > 0:
            sqrt_error_rel.append((current_sq_error_rel/n_rel)**0.5)
        if n_all > 0:
            sqrt_error_all.append((current_sq_error_all/n_all)**0.5)

    
    #np.savetxt('best_rmse/rp2_rmse_rel_4.txt', np.asarray(sqrt_error_rel))
    #np.savetxt('best_rmse/rp2_rmse_all_4.txt', np.asarray(sqrt_error_all))


    rmse_rel = sum(sqrt_error_rel)/len(sqrt_error_rel) if len(sqrt_error_rel) > 0 else 0
    rmse_all = sum(sqrt_error_all)/len(sqrt_error_all) if len(sqrt_error_all) > 0 else 0

    return rmse_rel, rmse_all

def top_k_non_effective_rmse(D, S, rel, k):

    n_ineffective_list = []
    n_train_list       = []
    n_nan_list         = []       
    sqrt_error_list    = []

    # sort descendingly and get the index by each cell line
    ind = np.argsort(-S, axis=1)


    for u in range(len(D)):
        current_ineffective = 0.0
        current_train       = 0.0
        current_nan         = 0.0
        current_sq_error    = 0.0
        for position in range(k):
            i = ind[u, position]
            
            # find nan
            if np.isnan(D[u][i]):
                current_nan   += 1
            # find training example
            elif D[u][i] == 0.0: 
                current_train += 1
            # find non-effective and calculate RMSE
            elif D[u][i] < rel[u]:
                current_sq_error += (D[u][i] - S[u][i])**2
                current_ineffective += 1

        n_ineffective_list.append(current_ineffective)
        n_train_list.append(current_train)
        n_nan_list.append(current_nan)
        if current_ineffective > 0:
            sqrt_error_list.append((current_sq_error/current_ineffective)**0.5)


    n_ineffective = sum(n_ineffective_list) / len(n_ineffective_list)
    n_train       = sum(n_train_list) / len(n_train_list)
    n_nan         = sum(n_nan_list) / len(n_nan_list)
    rmse_nonrel   = sum(sqrt_error_list) / len(sqrt_error_list)


    return rmse_nonrel, n_ineffective, n_train, n_nan






#def eval(D, U, V, ks, rel):
def eval(D, S, ks, rel):
    #S = U.T.dot(V)
    cind = c_index(D, S)
    cind_rel = c_index_rel(D, S, rel)
    rmse_rel, rmse_all = rmse(D, S, rel)
    ap_at_k = []
    ah_at_k = []
    best_ap_at_k = []
    for k in ks:
        ap_at_k.append(average_precision(D, S, k, rel))
        ah_at_k.append(average_hit(D, S, k, rel))
        best_ap_at_k.append(best_average_precision(D, rel, k))

    rmse_nonrel, n_ineffective, n_train, n_nan = top_k_non_effective_rmse(D, S, rel, 5)

    return cind_rel, cind, rmse_rel, rmse_all, ap_at_k, ah_at_k, best_ap_at_k, rmse_nonrel, n_ineffective, n_train, n_nan

def main(): 
    
    #######################################
    # add parameters
    #######################################
    parser = argparse.ArgumentParser()
    parser.add_argument('-test_file')
    parser.add_argument('-rel_file')
    parser.add_argument('-pred_file')
    parser.add_argument('-eval_file')

    args       = parser.parse_args()
    test_file  = str(args.test_file)
    rel_file   = str(args.rel_file)
    pred_file  = str(args.pred_file)
    eval_file  = str(args.eval_file)


    D_test = np.genfromtxt(test_file , delimiter=',', missing_values='nan')
    D_pred = np.genfromtxt(pred_file, delimiter=',', missing_values='nan')
    rel    = np.genfromtxt(rel_file, delimiter=',', missing_values='nan')


    S      =   D_pred
    D_test = - D_test   # lower test scores rank higher
    rel    = - rel

    cind_rel, cind, rmse_rel, rmse_all, ap_at_k, ah_at_k, best_ap_at_k, rmse_nonrel, n_ineffective, n_train, n_nan = eval(D_test, S, [5, 10], rel)

    writer = open(eval_file, 'w')
    #writer.write("%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (cind_rel, cind, rmse_rel, rmse_all, ap_at_k[0], ah_at_k[0], ap_at_k[1], ah_at_k[1], rmse_nonrel, n_ineffective, n_train, n_nan))

    writer.write("CI among sensitive drugs: %.4f\n" % cind_rel)
    writer.write("CI among all drugs: %.4f\n" % cind)
    writer.write("RMSE among sensitive drugs: %.4f\n" % rmse_rel)
    writer.write("RMSE among all drugs: %.4f\n" % rmse_all)
    writer.write("AP@5: %.4f\n" % ap_at_k[0])
    writer.write("AH@5: %.4f\n" % ah_at_k[0])
    writer.write("AP@10: %.4f\n" % ap_at_k[1])
    writer.write("AH@10: %.4f\n" % ah_at_k[1])
    writer.write("RMSE among non-sensitive testing drugs @top5: %.4f\n" % rmse_nonrel)
    writer.write("Average number of non-sensitive testing drugs @top5 pred: %.4f\n" % n_ineffective)
    writer.write("Average number of training drugs @top5 pred: %.4f\n" % n_train)
    writer.write("Average number of missing values @top5 pred: %.4f\n" % n_nan)

    writer.close()


if __name__ == '__main__':
    main()
