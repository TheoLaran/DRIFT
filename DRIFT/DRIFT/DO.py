import time

import numpy as np

from DRIFT.DRIFT_Function import DRIFT_Function
from utils.utils import DRIFT_Cipher, tf_cross_entropy, tf_cross_entropy_gradient
import tensorflow as tf


class Block:
    def __init__(self):
        self._positive = []
        self._negative = []

    def reset(self):
        self._positive = []
        self._negative = []

    def add_negative(self, i):
        self._negative.append(i)

    def add_positive(self, i):
        if len(self._negative) == 0:
            return
        self._positive = [i]

    def get_negative(self):
        return self._negative

    def get_positive(self):
        return self._positive

    def is_complete(self):
        return len(self._positive) != 0

    def clone(self):
        b = Block()
        b._negative = list(self._negative)
        b._positive = list(self._positive)
        return b

    def __repr__(self):
        return f"positive : {self._positive}\n negative : {self._negative}\n\n"


class DataOwner(DRIFT_Function):
    def __init__(self, genre, key, nonce):
        self.cipher = DRIFT_Cipher(key, nonce)
        self.all_users_triplets = []
        self._items = []
        self._blocks = {}
        self._genre = genre
        self.assigments = []
        self.time_in_here = 0
        self.save = None


    def get_nb_users_nb_items(self):
        return len(self._blocks), len(self._items)

    def add_item(self, i):
        self._items.append(i)

    def get_item(self):
        return self._items

    def create_block_from_interaction(self, b: Block, u: int):
        neg = b.get_negative()
        pos = b.get_positive()
        if len(pos) != 1:
            raise Exception("Should have only one positive item")
        return {
            "user_ids": [u] * len(neg),
            "left_ids": pos * len(neg),
            "right_ids": neg,
            "target_y": [1] * len(neg)
        }


    def _receive_block(self, b: Block, u: int):
        if self.save is None :
            self.save = {u: b}
            return
        all_X = []
        all_X.append(self.create_block_from_interaction(b, u))
        for user in self.save.keys():
            all_X.append(self.create_block_from_interaction(self.save[user], user))

        self.save = None
        return all_X

    def get_loss(self, embedding_left, embedding_right, embedding_user, second=False):
        def _get_loss():
            """
                Faire un seul arbre qui calcule tout
                Get the miss ranking loss of a block
                1) Get the embeddings user, left, right from the COS
                2) compute dif, margin, target_y
                3) computes the Loss$
                4)


            """
            # raw margins for primal ranking loss
            embedding_diff = embedding_left - embedding_right
            embedding_margins = tf.sigmoid(tf.reduce_sum(tf.multiply(embedding_user, embedding_diff), axis=1,
                                                   name='embedding_margins'))

            target_value = 1.

            grad_left = tf_cross_entropy_gradient(embedding_margins, target_value, -embedding_user)

            grad_right = tf_cross_entropy_gradient(embedding_margins, target_value,
                                                                         embedding_user)

            grad_user = tf_cross_entropy_gradient(embedding_margins, target_value,
                                                                        embedding_diff)

            embedding_loss = tf_cross_entropy(embedding_margins, 1.)

            return (grad_left, grad_right, grad_user), embedding_loss
        return self.computes_function(_get_loss)

    def update_block(self, secure, is_positive):
        def _update_block():
            # decrypt
            u, i = self.cipher.decrypt(secure)

            if u not in self._blocks:
                self._blocks[u] = Block()

            if not is_positive and self._blocks[u].is_complete():
                self._blocks[u].reset()

            if is_positive:
                self._blocks[u].add_positive(i)

            else:
                self._blocks[u].add_negative(i)

            if self._blocks[u].is_complete():
                send = self._blocks[u].clone()

                return self._receive_block(send, u)

            return
        return self.computes_function(_update_block)

    def __repr__(self):
        return f"DO {self._genre} \n"
