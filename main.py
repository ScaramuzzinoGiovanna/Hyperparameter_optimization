import rbfopt
from bayes_opt import BayesianOptimization
import random
from utility import *
from net import *
import argparse
from keras.models import load_model
from keras import backend as K



def random_opt(learning_rate, decay):
    val_loss, losses = f_opt(learning_rate, decay)
    losses_random_vect.append([learning_rate, decay, val_loss, losses])

    return val_loss


def bayes_opt(learning_rate, decay):
    val_loss, losses = f_opt(learning_rate, decay)
    losses_bayes_vect.append([learning_rate, decay, val_loss, losses])

    return -val_loss


def rbf_opt(x):
    learning_rate = x[0]
    decay = x[1]
    val_loss, losses = f_opt(learning_rate, decay)
    losses_rbf_vect.append([learning_rate, decay, val_loss, losses])

    return val_loss


def f_opt(learning_rate, decay):
    print('[INFO] ', optimization_type)
    print('[INFO] LR: {:.6f}, WD: {:.6f}'.format(learning_rate, decay))
    train_acc, train_loss, val_acc, val_loss, losses, epoch = train_model(train_generator, validation_generator,
                                                                          learning_rate, decay, n_epochs, batch_size,
                                                                          optimization_type,folder_weights)

    K.clear_session()

    #test_acc, test_loss = test_model(best_model, test_generator)

    saveHyp(path_hyp, learning_rate, decay, val_acc, val_loss, epoch, train_acc, train_loss)
    saveLoss(path_loss, learning_rate, decay, val_loss, losses)
    return val_loss, losses


batch_size = 32
n_eval = 25
n_epochs = 15
val_split = 0.2
hyp = {"learning_rate": (1e-5, 1e-1), "decay": (1e-5, 1e-1)}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--type optimizer", default='rbf', help="name of the optimizer: rbf, bayesian, random")
    args = vars(ap.parse_args())

    train_dir = "db/db_60000/train"
    test_dir = "db/db_60000/test"

    folder_out = create_folder('out')

    train_generator, validation_generator, test_generator = preprocessing(train_dir, test_dir, val_split, batch_size)

    if 'rbf' == args['type optimizer']:
        min_hyp = [hyp['learning_rate'][0], hyp['decay'][0]]
        max_hyp = [hyp['learning_rate'][1], hyp['decay'][1]]

        losses_rbf_vect = []
        optimization_type = 'rbf'

        path_loss, path_hyp, folder_csv, folder_plots, folder_weights = create_folder_and_csv(folder_out,
                                                                                              optimization_type)
        bb = rbfopt.RbfoptUserBlackBox(2, np.asarray(min_hyp), np.asarray(max_hyp), np.array(['R', 'R']), rbf_opt)
        settings = rbfopt.RbfoptSettings(max_evaluations=n_eval, target_objval=0.0, num_global_searches=3,
                                         minlp_solver_path='MINLP_simple.cpython-36.pyc')
        alg = rbfopt.RbfoptAlgorithm(settings, bb)
        val, x, itercount, evalcount, fast_evalcount = alg.optimize()
        print('best lr:{} and wd:{}'.format(x[0], x[1]))
        plot_loss(losses_rbf_vect, optimization_type, n_epochs, folder_plots)
        best_model = load_model(folder_weights + '/{}_weights_{}_{}.hdf5'.format(optimization_type, x[0], x[1]))
        test_acc, test_loss = test_model(best_model, test_generator)

    elif 'bayesian' == args['type optimizer']:
        losses_bayes_vect = []
        optimization_type = 'bayesian'

        path_loss, path_hyp, folder_csv, folder_plots, folder_weights = create_folder_and_csv(folder_out,
                                                                                              optimization_type)

        optimize = BayesianOptimization(f=bayes_opt, pbounds=hyp)
        optimize.maximize(
            init_points=3,
            n_iter=n_eval - 3,
        )
        print(optimize.max)
        lr_opt=optimize.max['params']['learning_rate']
        decay_opt=optimize.max['params']['decay']
        print(lr_opt,decay_opt)
        plot_loss(losses_bayes_vect, optimization_type, n_epochs, folder_plots)
        best_model = load_model(folder_weights + '/{}_weights_{}_{}.hdf5'.format(optimization_type,lr_opt,decay_opt))
        test_acc, test_loss = test_model(best_model, test_generator)

    elif 'random' == args['type optimizer']:
        optimization_type = 'random'

        path_loss, path_hyp, folder_csv, folder_plots, folder_weights = create_folder_and_csv(folder_out,
                                                                                              optimization_type)
        losses_random_vect = []

        for i in range(n_eval):
            best_loss = 9999999999
            lr = random.uniform(hyp['learning_rate'][0], hyp['learning_rate'][1])
            wd = random.uniform(hyp['decay'][0], hyp['decay'][1])

            loss = random_opt(lr, wd)
            if best_loss > loss:
                best_loss = loss
                learning_rate_opt = lr
                decay_opt = wd

        print(learning_rate_opt, decay_opt, best_loss)
        plot_loss(losses_random_vect, optimization_type, n_epochs, folder_plots)
        best_model = load_model(folder_weights + '/{}_weights_{}_{}.hdf5'.format(optimization_type,learning_rate_opt,decay_opt))
        test_acc, test_loss = test_model(best_model, test_generator)

    else:
        print('Error! insert correct name of optimizer')
