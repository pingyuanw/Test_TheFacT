import numpy as np
from tree import *
from main import *

if __name__ == "__main__":

    # initialization
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", help="training filename", default="../data/test_train.txt")
    parser.add_argument("--path", help="result path", default="../results/")
    parser.add_argument("--start", help="start epoch", default="")
    parser.add_argument("--num_run", help="number of iterations for alternatively creating the trees", default=5)
    parser.add_argument("--save_rate", help="frequency to save model", default=5)
    
    args = parser.parse_args()  
    num_dim = 20
    max_depth = 6
    num_BPRpairs = 20
    lr = 0.05
    lambda_u = 1
    lambda_v = 1
    lambda_BPR = 100
    num_run = int(args.num_run)
    num_iter_user = 20
    num_iter_item = 1000
    batch_size = 100
    random_seed = 0
    path = args.path + '/'
    train_file = args.train_file
    start = args.start
    save_rate = int(args.save_rate)
    
    rating_matrix, user_opinion, item_opinion = getRatingMatrix(train_file)

    num_users, num_items = rating_matrix.shape
    num_features = user_opinion.shape[1]
    print "Number of users", num_users 
    print "Number of items", num_items
    print "Number of features", num_features
    print "Number of latent dimensions: ", num_dim
    print "Maximum depth of the regression tree: ", max_depth    

    item_vector = np.load(path + 'item_vector' + start + '.npy')
    user_vector = np.load(path + 'user_vector' + start + '.npy')
    pred = np.dot(user_vector, item_vector.T)

    if len(start) == 0:
        start = '0'
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
        error = LA.norm(pred_old - pred) ** 2
        print ">>>>>Error: ", error
        if error < 0.1:
            print "!!!!!!CHANGE learning rate now!!!!!!"
            break
        i = i + 1
        if i%save_rate == 0:
            np.save(path + "item_tree", [item_tree])
            np.save(path + "user_tree", [user_tree])
            np.save(path + "item_vector" + str(i), item_vector)
            np.save(path + "user_vector" + str(i), user_vector)
