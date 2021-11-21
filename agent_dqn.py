import collections
import tflearn
import tensorflow as tf
import numpy as np
from Agent.dqnet import DQNetwork
from Utils.explorer import Explorer
from Utils.replaybuffer import ReplayBuffer
from Utils.summary import Summary


#for data plotting
#----------------------
import math
import time
from statistics import median,mean,mode,variance
import sys
import pandas as pd

sum_total_power = 0
sum_user_demand = 0
demand_counter = 0
total_power = []
user_demand = []
episode_rewards = []    #list of all episode rewards. We will plot this against number of episodes. This should be increasing as episodes increase.
previous_time = time.time()
slot_power = []
average_slot_power = []
episode_power = []
per_antenna_power=[]
#global Bandwidth #zakir
#----------------------

class DQNAgent:
    sum_user_demand=0
    demand_counter=0
    sum_total_power=0




    def __init__(self, env, config):
        self._sess = tf.Session()
        self._env = env
        self.flage_train=False
        self._dqn = DQNetwork(self._sess, env.dim_state, env.dim_action, config.lr)

        self._dir_mod_full = '{0}/{1}-dqn'.format(config.dir_mod, config.run_id)
        dir_sum_full = '{0}/{1}-dqn'.format(config.dir_sum, config.run_id)
        self._dir_log_full = '{0}/{1}-{2}.txt'.format(config.dir_log, config.run_id, 'dqn')
        
        print("\n----\n",self._dir_log_full,"\n----\n")
        
        self._summer = Summary(self._sess)
        self._summer.add_writer(dir_sum_full, name="dqn")
        self._summer.add_writer(dir_sum_full + '-max', name="max")
        self._summer.add_writer(dir_sum_full + '-min', name="min")
        self._summer.add_writer(dir_sum_full + '-rnd', name="rnd")
        self._summer.add_variable(name='ep-sum-reward')
        self._summer.add_variable(name='ep-mean-power')
        self._summer.add_variable(name='ep-loss')
        self._summer.add_variable(name='ep-rrh')
        self._summer.build()
    
        self._f_out = open(self._dir_log_full, 'w')
        self._store_args(config, self._f_out)

        self._replay_buffer = ReplayBuffer(config.buffer_size)
        self._explorer = Explorer(config.epsilon_init, config.epsilon_final, config.epsilon_steps)

        self._saver = tf.train.Saver(max_to_keep=5)

        self._train_flag = not config.load_id
        self._max_test_episodes = config.tests
        if config.load_id:
            self.flage_train=True
            self._load(config.dir_mod, config.load_id)
        else:
            #self._sess.run(tf.global_variables_initializer())
            self._sess.run(tf.compat.v1.global_variables_initializer())
        
        tflearn.is_training(self._train_flag, session=self._sess)
       
        self._OBVS = config.observations
        self._BATCH = config.mini_batch
        self._GAMMA = config.gamma

        self._num_rrh = config.num_rrh
        self._num_usr = config.num_usr

        self._max_episodes = config.episodes
        self._max_ep_sts = config.epochs
        self._max_steps = config.update

        self._ep = 0
        self._st = 0
        self._save_ep = config.save_ep
        self.reset_log()

    def reset_log(self):

        self._ep_reward = {
            'drl': [],
            'rnd': [],
            'min': [],
            'max': [],
        }

        self._ep_power = {
            'drl': [],
            'rnd': [],
            'min': [],
            'max': [],
        }

        self._ep_maxq = 0
        self._ep_loss = []
        self._actions = []
        self._rnd_ons = []

    def predict(self, state):
        q_value = self._dqn.predict([state])[0]
        self._ep_maxq = np.max(q_value)
        #print('q_value for DQN:',q_value )
        if self._train_flag:
            act = self._explorer.get_action(q_value)
            #print('training:')
        else:
            act = self._explorer.get_pure_action(q_value)
            #print("testing************valuescomingfasterthantraining:")
        return act
    def printWeights(self,nm):
        self._dqn.printWeights(nm)
    def work(self):
        
        #for data plotting
        #-----------------------
        global sum_total_power
        global sum_user_demand
        global demand_counter
        global episode_power
        global episode_rewards
        #global Bandwidth
        sum_total_power = 0
        sum_user_demand = 0
        demand_counter = 0
        episode_power.clear()
        #-----------------------

        per_antenna_power1=self._env._pow_tsm #zakir
        #Bandwidth=self._env._BAND #zakir
        init_state = state = self._env.reset_state()
        self.max_episodes = self._max_episodes if self._train_flag else self._max_test_episodes
        reset_state_ep = self._save_ep if self._train_flag else 20
       # print("ttttttttttttttttttttttttttttttttt")
        for _ in range(self.max_episodes):
            epi_reward = 0   #cumulative episode reward
            self._env.reset_demand()
            self._env.run_fix_solution()
            if not self._ep % reset_state_ep:
                init_state = state = self._env.reset_state()

            self._ep += 1
            self._explorer.decay()

            for ep_st in range(self._max_ep_sts):#max episode states
                self._st += 1

                sum_user_demand += sum(state[self._num_rrh:self._num_rrh + self._num_usr])


                action = self.predict(state)
                state_next, power, reward, done = self._env.step(action)
                #print("POWER:", power)
                #print("State: ", state, "Sum_user_demand:", sum_user_demand, "Action: ", action, "Next State: ", state_next, "Reward: ", reward, "Power: ", power, "Done: ", done)

                epi_reward += reward   #every reward collected in an episode needs to be added to cumulative epiode reward
#                print("\n\nstate-action: ", action,'\n\n')
#                print("\nstate: ",state)
#                print("\n\nnext-state: ", state_next,power, reward, done,'\n\n')

                _, power_max, reward_max = self._env.max_rrh_reward
                _, power_min, reward_min = self._env.min_rrh_reward
                on_rnd, power_rnd, reward_rnd = self._env.rnd_rrh_reward

                self._rnd_ons.append(on_rnd)

                self._ep_reward['drl'].append(reward)
                self._ep_reward['rnd'].append(reward_rnd)
                self._ep_reward['max'].append(reward_max)
                self._ep_reward['min'].append(reward_min)

                self._ep_power['drl'].append(power)
                self._ep_power['rnd'].append(power_rnd)
                self._ep_power['max'].append(power_max)
                self._ep_power['min'].append(power_min)

                self._actions.append(np.argmax(action)) ####Greedy action taken by an agent###update 
                self.flage_train=True
                self._train_batch((state, action, reward, state_next, done))
               # print("\n\nstate-> ",state,"\nstate_next-> ", state_next)
                self._write_log(state, state_next)
                state = state_next
                
                

                if done:
                    #print("RUPALI")
                    episode_rewards.append(
                        epi_reward)  # cumulative episode reward is appended to the list of all episode rewards
                    #print(episode_rewards)
                    break

            if self._train_flag and not self._ep % self._save_ep:
                self.save()
                self._write_log(init_state, state)
                init_state = state
                self.reset_log()
                
                
        #for data plotting
        #----------------------     
        global total_power
        global user_demand
        global per_antenna_power

        mean_total_power = 0
        if (sum_total_power != 0):  ######if sum of total power is not equal to zero####
            mean_total_power = sum_total_power/demand_counter
            mean_user_demand = sum_user_demand/demand_counter

            total_power.append(mean_total_power)
            per_antenna_power.append(per_antenna_power1)
            #Bandwidth=Bandwidth
            #ppr.append(ppr)#zakir
            #print("per anntna power by zakir",per_antenna_power)
            #total_power.append(self._env._P_MAX)
            print("DQN total power:", total_power)
            user_demand.append(mean_user_demand)
            print("DQN user demand:", user_demand)
            #user_demand.append(self._env._DM_MAX/1.e6)
            #for j in range(len(total_power)):
             #   print ('power', total_power[j])
        #----------------------
        #breakpoint()
        return total_power, user_demand, episode_rewards
        

    def _train_batch(self, sample):
        self._replay_buffer.add_samples([sample])

        if len(self._replay_buffer) < self._OBVS or len(self._replay_buffer) < self._BATCH:
            return False

        batch_state, batch_action, batch_reward, batch_state_next, batch_done = \
            self._replay_buffer.sample_batch(self._BATCH)

        q_values = self._dqn.predict_target(batch_state_next)
        # q_values = self._dqn.predict(batch_state_next)
        batch_y = []

        for q, reward, done in zip(q_values, batch_reward, batch_done):
            if done:
                batch_y.append(reward)
            else:
                batch_y.append(reward + self._GAMMA * np.max(q))

        _, loss = self._dqn.train(batch_state, batch_action, batch_y)
        self._ep_loss.append(loss)

        self._dqn.update_target()

        return True

    def _write_log(self, last_state, state):

        total_epochs = len(self._ep_reward['drl'])

        reward = np.array([self._ep_reward['drl'], self._ep_reward['rnd'], self._ep_reward['min'], self._ep_reward['max']])
        power = np.array([self._ep_power['drl'], self._ep_power['rnd'], self._ep_power['min'], self._ep_power['max']])

        index_non_zeros = (power[0, :] != 0) #& (power[1, :] != 0)

        reward = reward[:, index_non_zeros]
        power = power[:, index_non_zeros]
        
         
        total_epochs_non_0 = len(reward[0])

        reward = np.mean(reward, axis=1)
        power = np.mean(power, axis=1)

        reward = {'drl': reward[0], 'rnd': reward[1], 'min': reward[2], 'max': reward[3]}

        power = {'drl': power[0], 'rnd': power[1], 'min': power[2], 'max': power[3]}

        counter = collections.Counter(self._actions)
        init_state = ['{0:.0f}'.format(i) for i in last_state][:self._env.num_rrh]
        final_state = ['{0:.0f}'.format(i) for i in state][:self._env.num_rrh]
        if self._train_flag:
           # print('in training log>>>>>>>>>>>>>>>>')
            tmp = ' '.join(['| Episode: {0:.0f}'.format(self._ep),
                            '| Demand: {0}'.format(self._env.demand),
                            '| Epsilon: {0:.4f}'.format(self._explorer.epsilon),
                            '| Agent-steps: %i' % self._st,
                            '| Length: before {0} after {1}'.format(total_epochs, total_epochs_non_0),
                            '| Ep-max-reward: {0:.4f}'.format(reward['max']),
                            '| Ep-min-reward: {0:.4f}'.format(reward['min']),
                            '| Ep-rnd-reward: {0:.0f} {1:.4f}'.format(self._rnd_ons[-1], reward['rnd']),
                            '| Ep-reward: {0:.4f}'.format(reward['drl']),
                            '| Ep-max-power: {0:.4f}'.format(power['max']),
                            '| Ep-min-power: {0:.4f}'.format(power['min']),
                            '| Ep-rnd-power: {0:.4f}'.format(power['rnd']),
                            '| Ep-power: {0:.4f}'.format(power['drl']),
                            '| Num-rrh-on: %i' % self._env.num_rrh_on,
                            '| Ep-action: {0}'.format([(k, counter[k]) for k in sorted(counter.keys())]),
                            '| Init-state: {0}'.format('-'.join(init_state)),
                            '| Final-state: {0}'.format('-'.join(final_state))])

           
        else:
           csv_data = pd.read_csv('train_power.csv', skipinitialspace=True)
           Ep_power_csv=csv_data['Ep power']
          # print('power ',Ep_power_csv[0])
           tmp = ' '.join(['| Episode: {0:.0f}'.format(self._ep),
                        '| Demand: {0}'.format(self._env.demand),
                        '| Epsilon: {0:.4f}'.format(self._explorer.epsilon),
                        '| Agent-steps: %i' % self._st,
                        '| Length: before {0} after {1}'.format(total_epochs, total_epochs_non_0),
                        '| Ep-max-reward: {0:.4f}'.format(reward['max']),
                        '| Ep-min-reward: {0:.4f}'.format(reward['min']),
                        '| Ep-rnd-reward: {0:.0f} {1:.4f}'.format(self._rnd_ons[-1], reward['rnd']),
                        '| Ep-reward: {0:.4f}'.format(reward['drl']),
                        '| Ep-max-power: {0:.4f}'.format(power['max']),
                        '| Ep-min-power: {0:.4f}'.format(power['min']),
                        '| Ep-rnd-power: {0:.4f}'.format(power['rnd']),
                        '| Ep-power: {0:.4f}'.format(power['drl']),
 #                       '| Ep power:{0:.4f}'.format(Ep_power_csv[0]),
                        '| Num-rrh-on: %i' % self._env.num_rrh_on,
                        '| Ep-action: {0}'.format([(k, counter[k]) for k in sorted(counter.keys())]),
                        '| Init-state: {0}'.format('-'.join(init_state)),
                        '| Final-state: {0}'.format('-'.join(final_state))])
   
        #for data plotting
        #----------------------
        
        global slot_power   
        global slot_episode_counter
        global previous_time
        
        global sum_total_power
        global demand_counter
        
        global episode_power
        
        if (math.isnan(power['drl']) == True):
            episode_power.append('I')
        
        if (math.isnan(power['drl']) == False):
            
            episode_power.append(power['drl'])
            
            if(len(average_slot_power) < 20):
                slot_power.append(power['drl'])
                current_time = time.time()
                if ((current_time - previous_time)>=5):
                    average_slot_power.append(median(slot_power))
                    previous_time = time.time()
                  #  print('saved')
           # print('epo',self._ep) 
           # print('totla',self._max_ep_sts)
            if self._train_flag :
                self.traind_save=pd.DataFrame(columns=["Ep power"])
                self.traind_save=self.traind_save.append({"Ep power":str(power['drl'])}, ignore_index=True)
                self.traind_save.to_csv('train_power.csv')
               # print("DDQN Episode:", self._ep, " Power:",power['drl'])

            #print("DQN Episode:", self._ep, " Power:",power['drl'])
            sum_total_power += power['drl']
            #sum_user_demand +=
            demand_counter += 1
            #print( self._env._DM_MAX)
            #print('sum_total_power: ', sum_total_power)
            #print('demand_counter: ', demand_counter)
        
        
        
        #----------------------
    
    
        #print(tmp)
        self._f_out.write(tmp + '\n')
        self._f_out.flush()

        if len(self._ep_loss) > 0:
            self._summer.run(feed_dict={
                'ep-loss': np.mean(self._ep_loss),
                'ep-rrh': self._env.num_rrh_on,
                'ep-sum-reward': reward['drl'],
                'ep-mean-power': power['drl'],
            }, name='dqn', step=self._ep)
            self._summer.run(feed_dict={
                'ep-sum-reward': reward['max'],
                'ep-mean-power': power['max'],
            }, name='max', step=self._ep)
            self._summer.run(feed_dict={
                'ep-sum-reward': reward['min'],
                'ep-mean-power': power['min'],
            }, name='min', step=self._ep)
            self._summer.run(feed_dict={
                'ep-sum-reward': reward['rnd'],
                'ep-mean-power': power['rnd'],
            }, name='rnd', step=self._ep)

    def save(self):
        save_path = self._saver.save(self._sess, self._dir_mod_full + '/model',
                                     global_step=self._ep, write_meta_graph=False)
        #tf.logging.info("Model saved in file: {0}".format(save_path))
        tf.compat.v1.logging.info("Model saved in file: {0}".format(save_path))

    def _load(self, dir_mod, load_id):
        self._saver.restore(self._sess, tf.train.latest_checkpoint(dir_mod + '/' + load_id +'-dqn'))
        #tf.logging.info('Model restored from {0}'.format(load_id))
        tf.compat.v1.logging.info('Model restored from {0}'.format(load_id))

    @staticmethod
    def _store_args(config, f_out):
        tmp = ''
        for k in sorted(config.__dict__.keys()):
            tmp += '{0:<15} : {1}\n'.format(k, config.__dict__[k])
        print(tmp)
        f_out.write(tmp)
        f_out.flush()
