import numpy as np
from tree import *
from main import *

if __name__ == "__main__":

    # initialization
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", help="training filename", default="../data/test_train.txt")
    parser.add_argument("--test_file", help="training filename", default="../data/test_test.txt")
    parser.add_argument("--path", help="result path", default="../results/")

    parser.add_argument("--start", help="start epoch", default="")
    parser.add_argument("--num_run", help="number of iterations for alternatively creating the trees", default=5)
    parser.add_argument("--save_rate", help="frequency to save model", default=5)
    
    parser.add_argument("--num_dim", help="the number of latent dimension", default=20)
    parser.add_argument("--learning_rate", help="learning rate", default=0.05)
    parser.add_argument("--lambda_u", help="regularization parameter for user vectors", default=1)
    parser.add_argument("--lambda_v", help="regularization parameter for item vectors", default=1)
    parser.add_argument("--lambda_bpr", help="regularization parameter for BPR term", default=100)
    parser.add_argument("--num_BPRpairs", help="number of BPR pairs for each udpate", default=20)
    parser.add_argument("--batch_size", help="batch size for stochastic gradient descent", default=100)
    parser.add_argument("--num_iter_user", help="number of iterations for user vector update", default=20)
    parser.add_argument("--num_iter_item", help="number of iterations for item vector udpate", default=1000)

    args = parser.parse_args()  
    
    lambda_u = float(args.lambda_u)
    lambda_v = float(args.lambda_v)
    lambda_BPR = float(args.lambda_bpr)
    num_BPRpairs = int(args.num_BPRpairs)
    num_iter_user = int(args.num_iter_user)
    num_iter_item = int(args.num_iter_item)
    lr = float(args.learning_rate)
    num_dim = int(args.num_dim)

    max_depth = 6
    batch_size = 100
    random_seed = 0

    num_run = int(args.num_run)
    train_file = args.train_file
    test_file = args.test_file
    path = args.path + '/'
    start = args.start
    save_rate = int(args.save_rate)
    
    print "*******arguments********"
    print "NUM_DIM", num_dim
    print "MAX_DEPTH", max_depth
    print "LAMBDA_V", lambda_v
    print "LAMBDA_U", lambda_u
    print "LAMBDA_BPR", lambda_BPR
    print "NUM_BPRPAIRS", num_BPRpairs
    print "BATCH_SIZE", batch_size
    print "NUM_ITER_U", num_iter_user
    print "NUM_ITER_V", num_iter_item
    print "learning rate", lr
    print "random_seed", random_seed
    print "*******---------********"

    rating_matrix, user_opinion, item_opinion = getRatingMatrix(train_file)
    test_rating, user_opinion_test, item_opinion_test = getRatingMatrix(test_file)


    num_users, num_items = rating_matrix.shape
    num_features = user_opinion.shape[1]
    print "Number of users", num_users 
    print "Number of items", num_items
    print "Number of features", num_features

    if len(start) == 0:
        start = '0'
        user_vector, item_vector = MatrixFactorization(num_dim, lr, lambda_u, lambda_v, 50, rating_matrix)
        np.savetxt(path + "item_vector0.txt", item_vector, fmt="%0.8f")
        np.savetxt(path + "user_vector0.txt", user_vector, fmt="%0.8f")
        
        pred = np.dot(user_vector, item_vector.T)
        print "********** MF NDCG **********"
        ndcg_10 = get_ndcg(pred, test_rating, 10)
        print "NDCG@10: ", ndcg_10
        ndcg_20 = get_ndcg(pred, test_rating, 20)
        print "NDCG@20: ", ndcg_20
        ndcg_50 = get_ndcg(pred, test_rating, 50)
        print "NDCG@50: ", ndcg_50
    else:
        item_vector = np.loadtxt(path + 'item_vector' + start + '.txt')
        user_vector = np.loadtxt(path + 'user_vector' + start + '.txt')
    
    pred = np.dot(user_vector, item_vector.T)

    i = int(start)
    while i < num_run + int(start):
        user_vector_old = user_vector
        iterm_vector_old = item_vector
        pred_old = pred

        print "********** Round", i+1, "create user tree **********"
        user_tree = Tree(Node(None, 1), rating_matrix=rating_matrix, opinion_matrix=user_opinion, anchor_vectors=item_vector, lr=lr,
                         num_dim=num_dim, max_depth=max_depth, num_BPRpairs=num_BPRpairs, lambda_anchor=lambda_v, lambda_target=lambda_u, 
                         lambda_BPR=lambda_BPR, num_iter=num_iter_user, batch_size=batch_size, random_seed=random_seed)
        # create the user tree with the known item latent factors
        user_tree.create_tree(user_tree.root, user_tree.opinion_matrix, user_tree.rating_matrix)
        print "get user vectors"
        user_vector = user_tree.get_vectors()
        # add the refinement to the leave nodes of user tree as personalized representation
        print "add personalized term"
        user_vector = user_tree.personalization(user_vector)

        print "********** Round", i+1, "create item tree **********"
        item_tree = Tree(Node(None, 1), rating_matrix=rating_matrix.T, opinion_matrix=item_opinion, anchor_vectors=user_vector, lr=lr,
                        num_dim=num_dim, max_depth=max_depth, num_BPRpairs=num_BPRpairs, lambda_anchor=lambda_u, lambda_target=lambda_v,
                        lambda_BPR=lambda_BPR, num_iter=num_iter_item, batch_size=batch_size, random_seed=random_seed)
        # create the item tree with the learned user latent factors
        item_tree.create_tree(item_tree.root, item_tree.opinion_matrix, item_tree.rating_matrix)
        item_vector = item_tree.get_vectors()
        # add the refinement to the leave nodes of item tree as personalized representation
        item_vector = item_tree.personalization(item_vector)

        
        pred = np.dot(user_vector, item_vector.T)


        i = i + 1
        if i%save_rate == 0:
            np.save(path + "item_tree", [item_tree])
            np.save(path + "user_tree", [user_tree])
            np.savetxt(path + "item_vector" + str(i) + ".txt", item_vector, fmt="%0.8f")
            np.savetxt(path + "user_vector" + str(i) + ".txt", user_vector, fmt="%0.8f")
            
        error = LA.norm(pred_old - pred) ** 2

        print "Error: ", error
        ndcg_10 = get_ndcg(pred, test_rating, 10)
        print "NDCG@10: ", ndcg_10
        ndcg_20 = get_ndcg(pred, test_rating, 20)
        print "NDCG@20: ", ndcg_20
        ndcg_50 = get_ndcg(pred, test_rating, 50)
        print "NDCG@50: ", ndcg_50

        if error < 0.1:
            print "!!!!!!CHANGE learning rate now!!!!!!"
            break
