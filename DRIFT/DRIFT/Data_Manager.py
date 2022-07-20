import time

from utils.utils import build_all_triplets_for_loss, positive_items_for_u


def setup_model(cos, dfs, is_main, cipher):

    interactions, data_test = dfs
    model = cos.model
    print("Build graph ...")
    model.build_graph()
    print("Initialize ...")
    model.initialize_session()
    print("Done ...")
    train, test, time = get_data_train_test(interactions, data_test, cos, cipher,is_main=is_main)
    u_all, n_all, p_all = train
    us_list_test, neg_all_test, pos_all_test = test

    x_test = {
        cos.model.user_ids: us_list_test,
        cos.model.left_ids: pos_all_test,
        cos.model.right_ids: neg_all_test,
        cos.model.target_y: [1] * len(us_list_test)
    }

    x_train = {
        cos.model.user_ids: u_all,
        cos.model.left_ids: p_all,
        cos.model.right_ids: n_all,
        cos.model.target_y: [1] * len(u_all)
    }

    return cos.model, (x_train, x_test), time

def send_item(dos, secure, is_positive, model):

    for do in dos:
        start = time.time()
        all_X = do.update_block(secure, is_positive)
        t = time.time() - start
        if all_X is not None :
            for do_X in all_X :
                X = {
                    model.user_ids: do_X["user_ids"],
                    model.left_ids: do_X["left_ids"],
                    model.right_ids: do_X["right_ids"],
                    model.target_y: do_X["target_y"],
                }
                do.all_users_triplets.append([X])
    return all_X, t


def get_data_train_test(sub_data, data_test, cos, cipher, is_main=False):
    """

    Args:
        cos: The server who manage the interaction

    This function will get all the interactions and return the block of interaction,
    depending on the data in the data owners.

    theses one are independent of the data owners.

    Returns: all the users, all the positive and negatives interaction.


    Returns: all the users, all the positive and negatives interaction.

    """
    users = list(set(data_test[0]))
    sub_users = list(set(sub_data[0]))
    u_all = []
    n_all = []
    p_all = []
    ##################################
    neg_all_test = []
    pos_all_test = []
    us_list_test = []
    t = 0
    for u in users:
        print(f"Creating the triplets for user {u}")
        ################################## T E S T I N G ##################################
        test = positive_items_for_u(data_test, u)
        if test is not None:
            pos_us_test, neg_us_test = test
            pos_triplets, neg_triplets = build_all_triplets_for_loss(pos_us_test, neg_us_test)
            pos_all_test += pos_triplets
            neg_all_test += neg_triplets
            us_list_test += [u] * len(pos_triplets)

        if not is_main:
            print("Not main, verifying users")
            if u not in sub_users:
                print("Not main, continue users")
                continue

        ################################## T R A I N I N G ##################################

        df_u = sub_data[sub_data[0] == u].sort_values(by=[3])
        items_for_user = list(df_u[1])
        clicks = list(df_u[2])
        # skip user if there are no positive or negative items for him
        if not (1 < sum(clicks) < len(clicks)):
            continue
        i = 0
        while clicks[i]:
            i += 1

        # Need to start with a negative item
        item = items_for_user[i]
        plaintext = cipher.encrypt(u, item)
        dos = cos.get_data_owner_from_item(item)
        send_item(dos, plaintext, 0, cos.model)
        for item, is_positive in zip(items_for_user, clicks):
            secure = cipher.encrypt(u, item)
            dos = cos.get_data_owner_from_item(item)
            all_X, ts = send_item(dos, secure, is_positive, model=cos.model)
            t += ts
            if all_X is None:
                continue
            for X in all_X :
                u_all += X["user_ids"]
                p_all += X["left_ids"]
                n_all += X["right_ids"]

    return (u_all, n_all, p_all), (us_list_test, neg_all_test, pos_all_test), t
