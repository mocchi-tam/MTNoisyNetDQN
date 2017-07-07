import chainer
import chainer.links as L
import chainer.functions as F
from chainer import initializers
from chainer import variable

import chainerrl
import gym

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=1000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--train', '-t', type=int, default=1,
                        help='If negative, skip training')
    parser.add_argument('--resume', '-r', type=int, default=-1,
                        help='If positive, resume the training from snapshot')
    parser.add_argument('--noise', '-n', type=int, default=-1,
                        help='If positive, adding noise to network')
    args = parser.parse_args()
    
    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    
    flag_train = True if args.train > 0 else False
    flag_resum = True if args.resume > 0 else False
    n_epoch = args.epoch if flag_train == True else 1
    noise = True if args.noise > 0 else False
    
    tsm = MTModel(args.gpu, flag_train, flag_resum, n_epoch, args.batchsize, noise)
    tsm.run()
    
class NoisyLinear(L.Linear):
    def _initialize_params(self, in_size):
        super(NoisyLinear, self)._initialize_params(in_size)
        self.in_size = in_size
        factor = 1.0 / np.sqrt(in_size)
        
        ini_w = np.ones_like(self.W) * factor
        ws_initializer = initializers._get_initializer(ini_w)
        self.w_sigma = variable.Parameter(ws_initializer)
        self.w_sigma.initialize((self.out_size, in_size))
        
        ini_b = np.zeros(self.out_size)
        bs_initializer = initializers._get_initializer(ini_b)
        self.b_sigma = variable.Parameter(bs_initializer)
        self.b_sigma.initialize((self.out_size))
        
    def __call__(self, x, noise=True, test=False):
        if self.W.data is None:
            self._initialize_params(x.size // x.shape[0])
        
        if not test and noise:
            xp = self.xp
            e_i = xp.random.normal(size=(self.in_size)).astype(xp.float32)
            e_j = xp.random.normal(size=(self.out_size)).astype(xp.float32)
            e_w = xp.outer(e_j,e_i)
            e_w = variable.Variable(xp.sign(e_w)*xp.sqrt(xp.abs(e_w)))
            e_b = variable.Variable(xp.sign(e_j)*xp.sqrt(xp.abs(e_j)))
            
            W = self.W + self.w_sigma * e_w
            b = self.b + self.b_sigma * e_b
        else:
            W = self.W
            b = self.b
        
        return F.connection.linear.linear(x, W, b)
    
class MTNNet(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_units, noise=True):
        super(MTNNet, self).__init__()
        self.noise = noise
        self.test = False
        self.unit = n_units
        
        with self.init_scope():
            self.lin1 = NoisyLinear(obs_size,self.unit, 
                                    initialW=np.random.uniform(-1.0/np.sqrt(obs_size), 1.0/np.sqrt(obs_size), size=(self.unit,obs_size)))
            self.lin2 = NoisyLinear(self.unit,self.unit,
                                    initialW=np.random.uniform(-1.0/np.sqrt(self.unit), 1.0/np.sqrt(self.unit), size=(self.unit,self.unit)))
            self.lin3 = NoisyLinear(self.unit,n_actions,
                                    initialW=np.random.uniform(-1.0/np.sqrt(self.unit), 1.0/np.sqrt(self.unit), size=(n_actions,self.unit)))
            
    def __call__(self, x):
        h = self.lin1(x, noise=self.noise, test=self.test)
        h = F.leaky_relu(h)
        h = self.lin2(h, noise=self.noise, test=self.test)
        h = F.leaky_relu(h)
        h = self.lin3(h, noise=self.noise, test=self.test)
        a = chainerrl.action_value.DiscreteActionValue(h)
        return a

class MTModel():
    def __init__(self, gpu, flag_train, flag_resum, n_epoch, batchsize, noise):
        self.n_epoch = n_epoch
        self.flag_train = flag_train
        self.flag_resum = flag_resum
        self.gpu = gpu
        self.batchsize = batchsize
        self.noise = noise
        
    def run(self):
        self.env = gym.make('CartPole-v0')
        dirname = './cartpole-experiment-noise-on' if self.noise else './cartpole-experiment-noise-off'
        self.env = gym.wrappers.Monitor(self.env, dirname, force=True)
        
        print('observation space:', self.env.observation_space)
        print('action space:', self.env.action_space)
        
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n
        self.model = MTNNet(obs_size, n_actions, 100, self.noise)
        
        if self.gpu >= 0:
            chainer.cuda.get_device_from_id(self.gpu).use()
            self.model.to_gpu()
        
        self.optimizer = chainer.optimizers.Adam()
        self.optimizer.setup(self.model)
        
        gamma = 0.9
        epsilon = 0.0 if self.noise else 0.3
        
        explorer = chainerrl.explorers.ConstantEpsilonGreedy(
                epsilon=epsilon,
                random_action_func=self.env.action_space.sample)
        
        replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10**6)
        
        phi = lambda x: x.astype(np.float32, copy=False)
        
        agent = chainerrl.agents.DQN(
            self.model, self.optimizer, replay_buffer, gamma, explorer, 
            gpu=self.gpu, phi=phi,
            update_interval=1,
            replay_start_size=self.batchsize,
            minibatch_size=self.batchsize,
            target_update_interval=self.batchsize*10
            )
        
        if self.flag_resum:
            try:
                agent.load('agent')
                print('successfully resume model')
            except:
                print('ERROR: cannot resume model')
        
        n_episodes = self.n_epoch
        max_episode_len = 200
        R_train, R_test = [], []
        
        for i in range(1, n_episodes + 1):
            # train
            self.model.test = False
            obs = self.env.reset()
            reward = 0
            done = False
            R = 0
            t = 0
            while not done and t < max_episode_len:
                action = agent.act_and_train(obs, reward)
                obs, reward, done, _ = self.env.step(action)
                R += reward
                t += 1
            
            agent.stop_episode_and_train(obs, reward, done)
            R_train.append(R)
            
            print('episode:', i,
                  'R:', R,
                  'statistics:', agent.get_statistics())
            
            # test
            self.model.test = True
            obs = self.env.reset()
            reward = 0
            done = False
            R = 0
            t = 0
            while not done and t < max_episode_len:
                #self.env.render()
                action = agent.act(obs)
                obs, reward, done, _ = self.env.step(action)
                R += reward
                t += 1
            
            R_test.append(R)
            
            plt.plot(R_train, 'r-', label='train')
            plt.plot(R_test, 'b-', label='test')
            plt.xlabel('epoch')
            plt.ylabel('score')
            plt.legend(loc='best')
            plt.show()
            agent.stop_episode()
            
        print('Finished.')
        agent.save('agent')

if __name__ == '__main__':
    main()
