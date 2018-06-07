#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=30,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')
    input = tf.placeholder(shape = (None,111),dtype = tf.float32)
    label = tf.placeholder(shape = (None,8),dtype = tf.float32)
    
    layer1 = tf.layers.dense(input,units= 256,activation = tf.nn.relu)
    layer2 = tf.layers.dense(layer1,units=128,activation= tf.nn.relu)
    layer3 = tf.layers.dense(layer2,units=64,activation=tf.nn.relu)
    output = tf.layers.dense(layer3,units=8,activation=None)
    loss = tf.losses.mean_squared_error(labels = label,predictions=output)
    opt = tf.train.AdamOptimizer(learning_rate = 0.001)
    train_op = opt.minimize(loss=loss)
    init = tf.initialize_all_variables()    
    with tf.Session() as session:
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit
        
        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if  args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}
        
        num_iteration = 100
        datalen = len(expert_data['observations'])
        session.run(init)
        for i in range(num_iteration):
            perm = np.random.permutation(datalen)
            observ = expert_data['observations'][perm]
            act = expert_data['actions'][perm]
            for j in range(int(datalen/32)):
                batch_x = observ[j:j+32]
                batch_y = act[j:j+32].reshape((32,8))
                session.run(train_op,feed_dict={input:batch_x,label:batch_y})
        ret = []
        for i in range(args.num_rollouts):
            obs = env.reset()
            done = False
            totalr = 0
            step = 0
            while not done:
                obs= obs.reshape((1,obs.shape[0]))
                action = session.run(output,feed_dict={input : obs})
                obs,r,done,_ = env.step(action)
                totalr += r
                step+=1
                # env.render()
            ret.append(totalr)
        print('returns', ret)
        print('mean return', np.mean(ret))
        print('std of return', np.std(ret))

if __name__ == '__main__':
    main()
