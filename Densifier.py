import os
import itertools
import sys
import random
import pickle

random.seed(42)
os.environ["MKL_NUM_THREADS"] = "40"
os.environ["NUMEXPR_NUM_THREADS"] = "40"
os.environ["OMP_NUM_THREADS"] = "40"

from helpers import *
from sys import exit


def parse_words(add_bib):
    pos, neg = [], []
    with open("sentiment-lx/mypos.txt", "r") as f:
        for line in f.readlines():
            if not add_bib:
                pos.append(line.strip())
            else:
                pos.append(line.strip()+"@bib")
    with open("sentiment-lx/myneg.txt", "r") as f:
        for line in f.readlines():
            if not add_bib:
                neg.append(line.strip())
            else:
                neg.append(line.strip()+"@bib")
    return pos, neg


def batches(it, size):
    batch = []
    for item in it:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if len(batch) > 0: yield batch # yield the last several items


class Densifier(object):
    def __init__(self, alpha, d, ds, lr, batch_size, seed=42):
        self.d = d
        self.ds = ds
        self.Q = np.matrix(scipy.stats.ortho_group.rvs(d, random_state=seed))
        self.P = np.matrix(np.eye(ds, d))
        self.D = np.transpose(self.P) * self.P
        self.zeros_d = np.matrix(np.zeros((self.d, self.d)))
        self.lr = lr
        self.batch_size = batch_size
        self.alpha = alpha

    def _gradient(self, loss, vec_diff):
        if loss == 0.:
            print ("WARNING: check if there are replicated seed words!")
            return self.zeros_d[0, :]
        return self.Q[0, :] * vec_diff * np.transpose(vec_diff) / loss

    def train(self, num_epoch, pos_vecs, neg_vecs, save_to, save_every):
        bs = self.batch_size
        save_step = 0
        diff_ps = list(itertools.product(pos_vecs, neg_vecs))
        same_ps = list(itertools.combinations(pos_vecs, 2)) + \
                  list(itertools.combinations(neg_vecs, 2))
        for e in xrange(num_epoch):
            random.shuffle(diff_ps)
            random.shuffle(same_ps)
            steps_orth = 0
            steps_print = 0
            steps_same_loss, steps_diff_loss = [], []
            for (mini_diff, mini_same) in zip(batches(diff_ps, bs), batches(same_ps, bs)):
                steps_orth += 1
                steps_print += 1
                save_step += 1
                diff_grad, same_grad = [], []

                EW, EV = [], []
                for ew, ev in mini_diff:
                    EW.append(np.asarray(ew))
                    EV.append(np.asarray(ev))
                VEC_DIFF = np.asarray(EW) - np.asarray(EV)
                DIFF_LOSS = np.absolute(VEC_DIFF * self.Q[0, :].reshape(self.d,1))
                for idx in range(len(EW)):
                    diff_grad_step = self._gradient(DIFF_LOSS[idx][0,0], VEC_DIFF[idx].reshape(self.d, 1))
                    diff_grad.append(diff_grad_step)

                EW, EV = [], []
                for ew, ev in mini_same:
                    EW.append(np.asarray(ew))
                    EV.append(np.asarray(ev))
                VEC_SAME = np.asarray(EW) - np.asarray(EV)
                SAME_LOSS = np.absolute(VEC_SAME * self.Q[0, :].reshape(self.d,1))
                for idx in range(len(EW)):
                    same_grad_step = self._gradient(SAME_LOSS[idx][0,0], VEC_SAME[idx].reshape(self.d, 1))
                    same_grad.append(same_grad_step)

                diff_grad = np.mean(diff_grad, axis=0)
                same_grad = np.mean(same_grad, axis=0)

                self.Q[0, :] -= self.lr * (-1. * self.alpha * diff_grad * 2. + (1.-self.alpha) * same_grad * 2.)
                steps_same_loss.append(np.mean(SAME_LOSS))
                steps_diff_loss.append(np.mean(DIFF_LOSS))
                if steps_print % 10 == 0:
                    print ("=" * 25)
                    try:
                        print ("Diff-loss: {:4f}, Same-loss: {:4f}, LR: {:4f}".format(
                        np.mean(steps_diff_loss), np.mean(steps_same_loss), self.lr))
                        print (np.sum(self.Q))
                    except:
                        print (np.mean(steps_diff_loss))
                        print (np.mean(steps_same_loss))
                        print (self.lr)
                    steps_same_loss, steps_diff_loss = [], []
                if steps_orth % sys.maxsize == 0:
                    self.Q = Densifier.make_orth(self.Q)
                if save_step % save_every == 0:
                    self.save(save_to)
                    print ("Model saved! Step: {}".format(save_step))
            print ("="*25 + " one epoch finished! ({}) ".format(e) + "="*25)
            self.lr *= 0.99
        print ("Training finished ...")
        self.save(save_to)

    def save(self, save_to):
        with open(save_to, "wb") as f:
            pickle.dump(self.__dict__, f)
        print ("Trained model saved ...")

    @staticmethod
    def make_orth(Q):
        U, _, V = np.linalg.svd(Q)
        return U * V

    def transform(self, E):
        transformed = np.dot(self.P, np.dot(self.Q, E.T))
        return transformed

    def save_transformed(self, path, transformed, emb_vocab):
        transformed = np.squeeze(np.array(transformed))
        pd.Series(index=emb_vocab, data=transformed).to_csv(path, sep='\t', header=None)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5.)
    parser.add_argument("--alpha", type=float, default=.5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--out_dim", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--emb-space", type=str)
    parser.add_argument("--lexicon", type=str)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--save-to", type=str, default="trained_densifier.pck")
    parser.add_argument("--save-lexicon", type=str, default="transformed_lexicon.tsv")
    args = parser.parse_args()

    # pos_words, neg_words = parse_words(add_bib=False)
    pos_words, neg_words = load_lexicon(args.lexicon)

    ## load word embedding
    # myword2vec = word2vec(args.EMB_SPACE)
    emb, emb_dim, emb_vocab, emb_vectors = load_embeddings(args.emb_space)
    print ("Finish loading embedding ...")

    map(lambda x: random.shuffle(x), [pos_words, neg_words])
    pos_vecs, neg_vecs = map(lambda x: emblookup(x, emb), [pos_words, neg_words])

    assert len(pos_vecs) > 0
    assert len(neg_vecs) > 0
    mydensifier = Densifier(args.alpha, emb_dim, args.out_dim, args.lr, args.batch_size)
    mydensifier.train(args.epochs,
                      pos_vecs,
                      neg_vecs,
                      args.save_to,
                      args.save_every)

    transformed = mydensifier.transform(emb_vectors)
    mydensifier.save_transformed(args.save_lexicon, transformed, emb_vocab)

