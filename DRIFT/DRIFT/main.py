import os
import time

import numpy as np

from DRIFT.COS import COS
from DRIFT.Data_Manager import setup_model
from DRIFT.Preprocess import preprocess
from utils.utils import write, computes_metrics
from utils.utils import DRIFT_Cipher

def _update_block(block, model, do, second=False):
    model.receiver = do
    start_do = do.time_in_here
    start = time.time()
    _, embedding_loss = model.session.run([model.train, model.target], feed_dict=block)
    end = time.time()
    end_do = do.time_in_here
    model.time_in_here += end - start - (end_do - start_do)
    return embedding_loss



def update_model(dt, dos, model, X, computation_time, MIN_BLOCKS=0, MIN_TIMES=0, MIN_EPOCH=1 ):
    """
    Args:
        dt: The testing datas for the evaluation
        dos: All the DO present in the system
        model: The Tensorflow models who contains all the gradients
        X: All the training and testing data use for evaluation of the loss
        computation_time: The times added for the federated part
        MIN_BLOCKS: The maximal number of blocks of interaction before stopping the training
        MIN_TIMES: The maximal number of seconds before stopping the training

    Returns:

    """

    test_losses = []
    losses = []
    times = [0]
    blocks = [0]
    x_train, x_test = X
    model.session.run([model.target], feed_dict=x_train)
    t = 0
    e = 0
    while blocks[-1] < MIN_BLOCKS or times[-1] < MIN_TIMES or e < MIN_EPOCH:
        start = time.time()
        ############################# T R A I N I N G ##############################################################
        nb_blocks = 0
        user_losses = []
        for genre in dos:
            do = dos[genre]
            for user_triplets in do.all_users_triplets:
                if len(user_triplets) == 0:
                    raise Exception("User_triplets vide")
                user_losses.append(np.mean([_update_block(block, model, do) for block in user_triplets]))
            nb_blocks += len(do.all_users_triplets)
        losses.append(np.mean(user_losses))
        ############################# T E S T I N G ##############################################################
        test_loss = model.session.run([model.target], feed_dict=x_test)
        if test_loss is None:
            raise Exception(f"Error with the test, {test_loss}, {model.target}, {x_test} ")

        test_losses += test_loss
        ############################# E V A L U A T I O N ##############################################################

        blocks += [nb_blocks + blocks[-1]]
        t += model.time_in_here + computation_time
        model.time_in_here = 0
        times.append(t)
        e += 1

    return test_losses, losses, blocks, times





def create_central_server(nb_users, nb_items):
    # Send the data, from the DO to the COS
    c = COS(4, nb_users + 1, nb_items + 1)
    return c


def main(args):
    target_genre = args["target_genre"]
    do_max = None
    print(target_genre)
    if target_genre == "SAROS" :
        do_max = 1
    nonce = os.urandom(12)
    t = time.time()
    all_data, dos, nb_items, key = preprocess(nonce, args["dataset"], do_max)
    times = {"Preprocess": time.time() - t}

    data_train, data_test, dataset = all_data

    nb_u = max(set(data_train[0]))
    c = create_central_server(nb_items=nb_items, nb_users=nb_u)

    # Sending the data of the new DO to the COS
    is_main = (target_genre == "Main" or target_genre == "SAROS")
    print("Sending the data to the COS")
    if is_main:
        items = []
        for genre in dos:
            do = dos[genre]
            c.add_DO(do)
            c.model.receiver = do
            items += do.get_item()
        a_cos = [data_train[data_train[1].isin(items)], target_genre]
    else:
        do = dos[target_genre]
        c.add_DO(do)
        c.model.receiver = do
        d = data_train[data_train[1].isin(do.get_item())]
        a_cos = [d, target_genre]

    # Training part
    sub_data, genre = a_cos
    model, X, computation_time = setup_model(c, (sub_data, data_test), is_main, DRIFT_Cipher(key, nonce))
    computation_time *= int(target_genre != "SAROS")

    test_losses, losses, blocks, tme = update_model(data_test, dos, model, X, computation_time, MIN_BLOCKS=args["MIN_b"],
                                                      MIN_TIMES=args["MIN_t"], MIN_EPOCH=args["MIN_e"])
    times["COS"] = c.time_in_here
    times["Update"] = tme
    for genre in dos:
        do = dos[genre]
        times[f'DO_{genre}'] = do.time_in_here
    print(computation_time)
    if args["predict"]:
        computes_metrics(data_test, f"{target_genre if is_main else 'DO'}", model)
    if args["write"] :
        write(target_genre + "_secure", losses, test_losses, blocks, times)

    print(losses, test_losses, blocks, times)
