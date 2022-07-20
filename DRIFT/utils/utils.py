import time
from ast import literal_eval

import tensorflow as tf

from DRIFT.DRIFT_Function import DRIFT_Function
from utils.write import write_prediction, write
from operator import itemgetter
from Crypto.Random import get_random_bytes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

latent_dim = 4  # embedding size
train_part = 0.8
threshold = 5e4

PATH = "output/"

GENRES = ['Main', 'Animation', 'Sci-Fi', 'Thriller', 'Comedy', 'Crime', 'Drama', 'Adventure', 'Children', 'Action',
          'Romance',
          'War', 'Western', 'Horror', 'Musical', 'IMAX', 'Mystery', 'Fantasy', 'Documentary',
          '(no genres listed)', 'Film-Noir']

#######SAROS#########@
# time problem in here
def tf_cross_entropy(raw_margins, target_values,trunc_max=100, model=None):

    elementwise_entropy_loss = -tf.math.log(raw_margins)
    msg = f'NaN or Inf in loss vector '
    checked_elwise_loss = tf.compat.v1.verify_tensor_all_finite(elementwise_entropy_loss,
                                                      msg=msg, name='checked_elwise_ce')

    mean_loss = tf.reduce_mean(tf.minimum(checked_elwise_loss, trunc_max))

    return mean_loss



def tf_cross_entropy_gradient(embedding_margins, target_values, other):
    """
        Since the loss function is equal to target * log(sigmoid(margin)), with margin = U(Vi, Vi')
        The gradient will be target * sigmoid'(margin) / sigmoid(margin)
        with sigmoid'(margin) = sigmoid(margin) * (1 - sigmoid(margin)) * margin'
        This result to a gradient equals to target * (1 - sigmoid(margin)) * margin'

        Parameters:
            embedding_margins: The margin a scalar value
            target_values: The target value of the model (1)
            other : The d-dimension vector used for convergence :
                                * For the user U = (Vi - Vi')
                                * For the positive items Vi = U
                                * For the negative items Vi' = Vi

    """
    sig = tf.sigmoid(embedding_margins) # Scalar
    left = tf.multiply(target_values, 1. - sig) # Scalar
    return tf.linalg.matvec(tf.transpose(other), left) # d-dimension Vector


def tf_mean_l2(w):
    elementwise_sq_norm = tf.reduce_sum(tf.pow(w, 2), axis=1)
    checked_elwise_l2 = tf.compat.v1.verify_tensor_all_finite(elementwise_sq_norm, msg='NaN or Inf in norm', name='checked_elwise_l2')
    mean_l2 = tf.reduce_mean(checked_elwise_l2)
    return mean_l2


class SAROS(DRIFT_Function):

    def __init__(self, n_embeddings, u=0, i=0, alpha_reg=0.01, seed=None):
        self.N_USERS = u
        self.N_ITEMS = i
        self.N_EMBEDDINGS = n_embeddings
        self.alpha_reg = alpha_reg
        self.seed = seed
        self.graph = tf.Graph()
        self.receiver = None
        self.time_in_here = 0
        if seed:
            self.graph.seed = seed
        self.start = 0

    def chrono(self):
        if self.start == 0 :
            self.start = time.time()
        else :
            self.time_in_here += time.time() - self.start
            self.start = 0

    def build_graph(self):

        with self.graph.as_default():
            self.chrono()
            self.user_ids = tf.compat.v1.placeholder(tf.int32, (None,), name='user_ids')
            self.left_ids = tf.compat.v1.placeholder(tf.int32, (None,), name='left_ids')
            self.right_ids = tf.compat.v1.placeholder(tf.int32, (None,), name='right_ids')
            self.target_y = tf.compat.v1.placeholder(tf.float32, (None,), name='target_y')

            # main parameters
            self.user_latents = tf.Variable(tf.random.uniform(shape=(self.N_USERS, self.N_EMBEDDINGS), seed=123),
                                            trainable=True, name='user_latents')
            self.item_latents = tf.Variable(tf.random.uniform(shape=(self.N_ITEMS, self.N_EMBEDDINGS), seed=124),
                                            trainable=True, name='item_latents')


            # get embeddings
            self.embedding_user = tf.nn.embedding_lookup(self.user_latents, self.user_ids, name='embedding_user')
            self.embedding_left = tf.nn.embedding_lookup(self.item_latents, self.left_ids, name='embedding_left')
            self.embedding_right = tf.nn.embedding_lookup(self.item_latents, self.right_ids, name='embedding_right')
            self.chrono()
            #### MODIFIED ####
            _, loss = self.receiver.get_loss(self.embedding_left, self.embedding_right, self.embedding_user)
            self.chrono()
            self.embedding_loss = loss
            self.regularization = tf_mean_l2(self.embedding_user) + tf_mean_l2(self.embedding_left) + tf_mean_l2(
                self.embedding_right)
            self.target = self.embedding_loss + self.alpha_reg * self.regularization
            self.opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
            self.train = self.opt.minimize(self.target)
            self.init_all_vars = tf.compat.v1.global_variables_initializer()
            self.chrono()
            return


    def initialize_session(self):
        config = tf.compat.v1.ConfigProto()
        # for reduce memory allocation
        config.gpu_options.allow_growth = True
        print("\t Creating the session ...")
        self.session = tf.compat.v1.Session(graph=self.graph, config=config)
        print("\t Init all vars ...")
        self.session.run(self.init_all_vars)

    @property
    def weights_u(self):
        return self.user_latents.eval(session=self.session)

    @property
    def weights_i(self):
        return self.item_latents.eval(session=self.session)

    #Works
    def score_items(self, user):
        return tf.reduce_sum(tf.multiply( self.item_latents, self.weights_u[user]), axis=1).eval(session=self.session)


    def get_recommendation(self, user: int, item):
        items = self.score_items(user)
        items_index = enumerate(items)
        items_index = list(filter(lambda x : x[0] in item, items_index))
        items_index.sort(reverse=True, key=itemgetter(1))
        return [item[0] for item in items_index]

    def destroy(self):
        self.session.close()
        self.graph = None


# saving all negatives and positives for each user
def get_negatives_and_positives_from_user(it_for_u, clicks):
    neg_all_us = []
    pos_all_us = []

    for n in range(len(clicks)):
        if clicks[n] > 1 :
            raise Exception(f"Need implicit interactions, get interaction {clicks[n]}")

        if clicks[n] == 0:
            neg_all_us = neg_all_us + [it_for_u[n]]

        if clicks[n] == 1:
            pos_all_us = pos_all_us + [it_for_u[n]]

    return pos_all_us, neg_all_us


# build triplets for test predictions
def build_all_triplets_for_loss(positives, negatives):
    neg_triplets = negatives * len(positives)
    pos_triplets = []
    for it in positives:
        pos_triplets += [it] * len(negatives)
    return pos_triplets, neg_triplets



######### W R I T I N G ##############
def _write_files(losses, times, test_losses, PATH, PATH_END):
    with open(f'{PATH}train/train_loss_online_ml{PATH_END}.txt', 'w') as f:
        f.write(str(losses))

    with open(f'{PATH}time/time_online_ml{PATH_END}.txt', 'w') as f:
        f.write(str(times))

    with open(f'{PATH}test/test_loss_online_ml{PATH_END}.txt', 'w') as f:
        f.write(str(test_losses))


def positive_items_for_u(df, user):
    df_user = df[df[0] == user].sort_values(by=[3])
    it_for_u = list(df_user[1])
    click = list(df_user[2])
    if len(it_for_u) != len(click) :
        print(f"Problem in the datasets with user {user}")
        return None

    if sum(click) == 0 or sum(click) == len(click):
        return None

    pos, neg = get_negatives_and_positives_from_user(it_for_u, click)
    return pos, neg

def computes_metrics(df, genre: str, model: SAROS):
    users = list(set(df[0]))
    res_gt = ""
    res_pr = ""
    for user in users:
        df_user = df[df[0] == user].sort_values(by=[3])
        it_for_u = list(df_user[1])
        click = list(df_user[2])

        if sum(click) == 0 :
            continue
        pertinent_items, _ = get_negatives_and_positives_from_user(it_for_u, click)
        recommend = model.get_recommendation(user, it_for_u)
        res_gt += f"{user} {pertinent_items}\n"
        res_pr += f"{user} {recommend}\n"

    write_prediction(res_gt, genre=genre, type="gt")
    write_prediction(res_pr, genre=genre, type="pr")



def generate_symetric_key():
    """Returns a new symmetric key."""
    return get_random_bytes(32)



class DRIFT_Cipher:
    """
    It uses the AES-256-GCM implementation available in the Ionic cryptographic SDK.
    https://dev.ionic.com/sdk/tasks/crypto-aes-gcm
    """
    def __init__(self, key, nonce):
        # AES-256-GCM requires a key length of 256 bits (32 bytes)
        if len(key) != 32:
            raise Exception("Invalid AES-256-GCM key length: requires 256 bits length key")
        self.key = key
        self.cipher = AESGCM(self.key)
        self.nonce = nonce


    def encrypt(self, user, item):
        """Returns the plaintext encrypted with the symmetric key."""
        plaintext = str((user, item)).encode('utf-8')
        return self.cipher.encrypt(self.nonce, plaintext, None)

    def decrypt(self, plaintext):
        """Returns the ciphertext decrypted with the symmetric key."""
        return literal_eval(self.cipher.decrypt(self.nonce, plaintext, None).decode('utf-8'))



    def __encrypt_aes_gcm(self, message : bytes):
        """Returns the given plaintext encrypted with AES-GCM."""
        nonce = get_random_bytes(12)
        return nonce, self.cipher.encrypt(
            nonce=nonce,
            data=message,
            associated_data=b'',
        )

    def __decrypt_aes_gcm(self, encrypted_data : (bytes,bytes)) -> bytes:
        """Returns the decrypted ciphertext with AES-GCM."""
        nonce, encrypted_data = encrypted_data
        return self.cipher.decrypt(
            nonce=nonce,
            data=encrypted_data,
            associated_data=b''
        )