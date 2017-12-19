"""
A simple version of OpenAI's Proximal Policy Optimization (PPO). [https://arxiv.org/abs/1707.06347]

Distributing workers in parallel to collect data, then stop worker's roll-out and train PPO on collected data.
Restart workers once PPO is updated.

The global PPO updating rule is adopted from DeepMind's paper (DPPO):
Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]

View more on my tutorial website: https://morvanzhou.github.io/tutorials

Dependencies:
tensorflow r1.3
gym 0.9.2
"""

import cv2
import tensorflow as tf
import numpy as np
import multiprocessing as mp
from env import Maze
import util as U
from baseline import baseline as base
import time

Map=\
['#o#*#o#*#',
 'o1o  *o^o',
 '# # # #o#',
 'o #####^o',
 '#*#####o#',
 'oo^oo^  o',
 '#oo^#oo0#'
]

EP_MAX = 100
EP_LEN = 200
N_WORKER = 4                # parallel workers
GAMMA = 0.9                 # reward discount factor
A_LR = 0.0001               # learning rate for actor
C_LR = 0.0002               # learning rate for critic
MIN_BATCH_SIZE = 256         # minimum batch size for updating PPO
UPDATE_STEP = 20            # loop update operation n-steps
EPSILON = 0.2               # for clipping surrogate objective
GAME = 'Pendulum-v0'
S_DIM, S_shape, A_DIM, n_actions = 161 * 207 * 3, [None, 161, 207, 3], 1, 6         # state and action dimension

tf.set_random_seed(1)
np.random.seed(1)

class PPO(object):
    def __init__(self, Load = False):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, S_shape, 'state')

        # critic
        with tf.variable_scope('critic'):
            conv1 = tf.layers.conv2d(self.tfs, 16, 8, 4, 'same', activation=tf.nn.relu)
            conv2 = tf.layers.conv2d(conv1, 32, 4, 2, 'same', activation=tf.nn.relu)
            flat = U.flattenallbut0(conv2)
            l1 = tf.layers.dense(flat, 256, activation=tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # operation of choosing action
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.int32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
        ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
        surr = ratio * self.tfadv                       # surrogate loss

        self.aloss = -tf.reduce_mean(tf.minimum(        # clipped surrogate objective
            surr,
            tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv))

        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        self.saver = tf.train.Saver()
        self.last_ep = 0

        if (Load):
            print('Loading!')
            self.saver.restore(self.sess, '/home/icenter/tmp/Crazy/params')
        else:
            self.sess.run(tf.global_variables_initializer())

    def update(self, QUEUE):
        global GLOBAL_UPDATE_COUNTER, GLOBAL_EP
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()                     # wait until get batch of data
                print('Start_update')
                self.sess.run(self.update_oldpi_op)     # copy pi to old pi
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]      # collect data from all workers
                data = np.vstack(data)
                s, a, r = data[:, :S_DIM], data[:, S_DIM: S_DIM + A_DIM], data[:, -1:]
                s = np.reshape(s, (-1, 161, 207, 3))
                adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
                # update actor and critic in a update loop
                [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(UPDATE_STEP)]
                [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(UPDATE_STEP)]

                if (GLOBAL_EP >= EP_MAX):
                    print('Saving!')
                    self.saver.save(self.sess, '/home/icenter/tmp/Crazy/params', write_meta_graph=False)
                    self.last_ep = GLOBAL_EP

                print('End_update')
                UPDATE_EVENT.clear()        # updating finished
                GLOBAL_UPDATE_COUNTER = 0   # reset counter
                ROLLING_EVENT.set()         # set roll-out available

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            conv1 = tf.nn.relu(U.conv2d(self.tfs, 16, "l1", [8, 8], [4, 4], pad="VALID", trainable=trainable))
            conv2 = tf.nn.relu(U.conv2d(conv1, 32, "l2", [4, 4], [2, 2], pad="VALID", trainable=trainable))
            flat = U.flattenallbut0(conv2)
            den1 = tf.nn.relu(U.dense(flat, 256, 'lin', U.normc_initializer(1.0), trainable=trainable))
            self.probs = tf.nn.softmax(U.dense(den1, n_actions, "logits", U.normc_initializer(0.01), trainable=trainable))

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        u = tf.distributions.Categorical(probs = self.probs)
        return u, params

        '''
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 200, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params
        '''

    def choose_action(self, s):
        #print(s.shape)
        #time.sleep(1)
        s = s[np.newaxis, :]
        #print(self.sess.run(self.probs, {self.tfs: s}))
        #time.sleep(1)
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return a

    def get_v(self, s):
        s = s[np.newaxis,:]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]


class Worker(object):
    def __init__(self, wid):
        self.wid = wid
        self.env = Maze(Map)
        self.ppo = GLOBAL_PPO

    def work(self, QUEUE):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            s = self.env.reset()
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            t = 0
            print('start : %d' % self.wid)
            while True:
                if not ROLLING_EVENT.is_set():                  # while global PPO is updating
                    ROLLING_EVENT.wait()                        # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []   # clear history buffer, use new policy to collect data
                a = self.ppo.choose_action(s)
                baseline_a = base.choose_action(self.env, 1)
                s_, r, done = self.env.step({(0, a), (1, U.ch(baseline_a))})
                r = r[0]
                buffer_s.append(s.flatten())
                buffer_a.append(a)
                buffer_r.append(r)                    # normalize reward, find to be useful
                s = s_
                ep_r += r

                t += 1
                #print('step : %d, reward : %d, done : %d' % (t, r, done))

                GLOBAL_UPDATE_COUNTER += 1               # count to minimum batch size, no need to wait other workers
                if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE or done:
                    #print(GLOBAL_EP)
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = self.ppo.get_v(s_)
                    discounted_r = []                           # compute discounted reward
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    QUEUE.put(np.hstack((bs, ba, br)))          # put data in the queue
                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        #print('update')
                        ROLLING_EVENT.clear()       # stop collecting data
                        UPDATE_EVENT.set()          # globalPPO update
                    if GLOBAL_EP >= EP_MAX:         # stop training
                        print('Train over')
                        COORD.request_stop()
                        break

                if done:
                    # record reward changes, plot later
                    if len(GLOBAL_RUNNING_R) == 0: GLOBAL_RUNNING_R.append(ep_r)
                    else: GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1]*0.9+ep_r*0.1)
                    GLOBAL_EP += 1
                    print('{0:.1f}%'.format(GLOBAL_EP/EP_MAX*100), '|W%i' % self.wid,  '|Ep_r: %.2f' % ep_r,)
                    break


if __name__ == '__main__':
    GLOBAL_PPO = PPO()

    UPDATE_EVENT, ROLLING_EVENT = mp.Event(), mp.Event()
    UPDATE_EVENT.clear()            # not update now
    ROLLING_EVENT.set()             # start to roll out
    workers = [Worker(wid=i) for i in range(N_WORKER)]
    QUEUE = mp.Queue()  # workers putting data in this queue
    
    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    GLOBAL_RUNNING_R = []
    COORD = tf.train.Coordinator()

    threads = []
    for worker in workers:          # worker threads
        t = mp.Process(target=worker.work, args=(QUEUE, ))
        t.start()                   # training
        threads.append(t)
    # add a PPO updating thread
    threads.append(mp.Process(target=GLOBAL_PPO.update, args=(QUEUE, )))
    threads[-1].start()

    COORD.join(threads)
    print('aasdas')
