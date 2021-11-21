import sys
sys.path.append("../Agent")
#from Agent.dqnet import DQNetwork
from Agent.agent_dqn import DQNAgent, user_demand, total_power, average_slot_power, episode_power#, episode_rewards,per_antenna_power
from Agent.agent_duelingdqn import DDQNAgent, us_demand, ttl_power, avg_slt_pwr, ep_power
from Agent.agent_doubledqn import DoubleDQNAgent, u_demand, t_power, a_s_p, e_power#, doubledqn_episode_rewards
#from Agent.agent_doubleqn import DoubleDQNAgent, cost_his
from Env.env import Env
from parsers import CRANParser
import matplotlib.pyplot as plt
import tensorflow as tf
import copy
import math
import numpy as np

#DQN
user_demandN=[]
total_powerN=[]

#Double DQN
u_demandN=[]
t_powerN=[]
#Duel DQN
us_demandN=[]
ttl_powerN=[]


#class AllDQN:
    #def __init__(self,args):
        #p_min, p_max, n_rrh, n_usr, n_epochs = args
    #self.DQN_Start(args)
    #self.DDQN_Start(args)
    #self.DoubleDQN_Start(args)

def DQN_Start(args):
    #n_userList=np.array([3,4,5,6,7,8])
    #n_userList=[3,4,5,6,7,8]
    p_min, p_max, n_rrhList,n_usr, n_epochs = args
    #print("Number of user:",n_usr)
    #breakpoint()
    print("DQNStart")
    #for i in range(len(p_min)):
    for i in range(len(n_rrhList)):
        parser = CRANParser()
        #parser.set_defaults(power_min=p_min[i], power_max=p_max[i], num_rrh=n_rrh, num_usr=n_usr, epochs=n_epochs, pow_tsm = p_max[i]/n_rrh)
        parser.set_defaults(num_rrh=n_rrhList[i], num_usr=n_usr, epochs=n_epochs, pow_tsm=p_max)
        #parser.set_defaults(num_rrh=n_rrhList[i], num_usr=n_usr, epochs=n_epochs, pow_tsm=p_max[i] / n_rrh)
        #parser.set_defaults(num_rrh=n_rrh, num_usr=n_usr, epochs=n_epochs, pow_tsm=1e-3 * (10. **((p_max[i] / n_rrh)/ 10)))
        #parser.set_defaults(num_rrh=n_rrh, num_usr=n_userList[i], epochs=n_epochs, pow_tsm=10 *math.log10 ((10 **3)* (p_max[i] / n_rrh)))# covert to dBm
        config = parser.parse_args()
        #print("Eenie")
        env = Env('master', config)
        #print("Meenie")
        #gains = env._get_gainps(n_rrh, n_usr)
        #print (gains)
        agent = DQNAgent(env, config)
        #print("Minie")
        tf.reset_default_graph()
        tpower,usrdemd,episodrewd=agent.work()

        meanuserdmd=sum(usrdemd)/len(usrdemd)
        user_demandN.append(meanuserdmd)
        meantotalpowr=sum(tpower)/len(tpower)
        total_powerN.append(meantotalpowr)
        agent.save()
        agent.printWeights("DQN")
        print("DQNEND")


def DDQN_Start(args):
    #n_userList=np.array([3,4,5,6,7,8])
    p_min, p_max, n_rrhList,n_usr, n_epochs = args
    print("DDQNStart")
    #for i in range(len(p_min)):
    for i in range(len(n_rrhList)):

        parser = CRANParser()
        #parser.set_defaults(power_min=p_min[i], power_max=p_max[i], num_rrh=n_rrh, num_usr=n_usr, epochs=n_epochs, pow_tsm = p_max[i]/n_rrh)
        parser.set_defaults(num_rrh=n_rrhList[i], num_usr=n_usr, epochs=n_epochs, pow_tsm=p_max)
        #parser.set_defaults(num_rrh=n_rrhList[i], num_usr=n_usr, epochs=n_epochs, pow_tsm=p_max[i] / n_rrh)
        #parser.set_defaults(num_rrh=n_rrh, num_usr=n_usr, epochs=n_epochs, pow_tsm=1e-3 * (10. ** ((p_max[i] / n_rrh) / 10)))
        #parser.set_defaults(num_rrh=n_rrh, num_usr=n_userList[i], epochs=n_epochs, pow_tsm=10 *math.log10 ((10 **3)* (p_max[i] / n_rrh)))# covert to dBm
        config = parser.parse_args()
        #print("Eenie")
        env = Env('master', config)
        #print("Meenie")
        #gains = env._get_gains(n_rrh, n_usr)
        #print (gains)
        agent = DDQNAgent(env, config)
        #print("Minie")
        tf.reset_default_graph()
        tpower,usrdemd,episodrewd=agent.work()

        meanuserdmd=sum(usrdemd)/len(usrdemd)
        us_demandN.append(meanuserdmd)
        meantotalpowr=sum(tpower)/len(tpower)
        ttl_powerN.append(meantotalpowr)
        agent.save()
        agent.printWeights("DDQN")
        print("DDQNEND")


def DoubleDQN_Start(args):
    #n_userList=np.array([3,4,5,6,7,8])
    p_min, p_max, n_rrhList,n_usr, n_epochs = args
    print("DoubleDQNStart")

    #for i in range(len(p_min)):
    for i in range(len(n_rrhList)):
        parser = CRANParser()
        #parser.set_defaults(power_min=p_min[i], power_max=p_max[i], num_rrh=n_rrh, num_usr=n_usr, epochs=n_epochs)
        parser.set_defaults(num_rrh=n_rrhList[i], num_usr=n_usr, epochs=n_epochs, pow_tsm=p_max)
        #parser.set_defaults(num_rrh=n_rrhList[i], num_usr=n_usr, epochs=n_epochs, pow_tsm=p_max[i] / n_rrh)
        #parser.set_defaults(num_rrh=n_rrh, num_usr=n_usr, epochs=n_epochs, pow_tsm=1e-3 * (10. ** ((p_max[i] / n_rrh) / 10)))
        #parser.set_defaults(num_rrh=n_rrh, num_usr=n_userList[i], epochs=n_epochs, pow_tsm=10 *math.log10 ((10 **3)* (p_max[i] / n_rrh)))# covert to dBm
        config = parser.parse_args()
        #print("Eenie")
        env = Env('master', config)
        #print("Meenie")
        #gains = env._get_gains(n_rrh, n_usr)
        #print (gains)
        agent = DoubleDQNAgent(env, config)
        #print("Minie")
        #tf.reset_default_graph()
        tf.compat.v1.reset_default_graph()
        tpower,usrdemd,episodrewd=agent.work()
        #tpower,usrdemd=agent.work()

        meanuserdmd=sum(usrdemd)/len(usrdemd)
        u_demandN.append(meanuserdmd)
        meantotalpowr=sum(tpower)/len(tpower)
        t_powerN.append(meantotalpowr)
        #temp = agent.work()
        agent.save()
        agent.printWeights("DoubleQN")

        print("DoubleDQNEND")
