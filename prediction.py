from __future__ import division

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import json
import tensorflow as tf
import numpy as np

def train(dict_games,dict_heroes):
    #create the model time
    #first our input
    in_xa = tf.placeholder(tf.float32, [None,5])
    in_xb = tf.placeholder(tf.float32, [None,5])

    in_x = tf.placeholder(tf.float32, [None,10,5])

    # in_xa = tf.placeholder(tf.float32, [5])
    # in_xb = tf.placeholder(tf.float32, [5])
    #
    # in_x = tf.placeholder(tf.float32, [10,5])

    # in_x0,in_x1,in_x2,in_x3,in_x4,in_x5,in_x6,in_x7,in_x8,in_x9 = (tf.placeholder(tf.float32, [None,5]) for i in range(10))
    # in_x_list = [in_x0,in_x1,in_x2,in_x3,in_x4,in_x5,in_x6,in_x7,in_x8,in_x9]

    g_r = tf.Variable(tf.zeros([5,2]))
    g_d = tf.Variable(tf.zeros([5,2]))
    b = tf.Variable(tf.zeros([2]))

    s = tf.Variable(tf.zeros([10,5,2]))

    # s_0,s_1,s_2,s_3,s_4,s_5,s_6,s_7,s_8,s_9 = (tf.Variable(tf.zeros([5,2])) for i in range(10))
    # s_list = [s_0,s_1,s_2,s_3,s_4,s_5,s_6,s_7,s_8,s_9]

    g_sum = tf.matmul(in_xa,g_r) + tf.matmul(in_xb,g_d)
    s_sum = tf.matmul(in_x[0],s[0])
    for i in range (1,10):
        s_sum += tf.matmul(in_x[i],s[i])
    out_y =  g_sum + s_sum + b

    # g_sum = (in_xa * g_r) + (in_xb * g_d)
    # s_sum = (in_x[0] * s[0])
    # for i in range (1,10):
    #     s_sum += (in_x[i] * s[i])
    # out_y =  g_sum + s_sum + b

    #defining loss and optimizer
    out_y_ = tf.placeholder(tf.float32,[None,2])

    #cross-entropy formulation
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=out_y_, logits=out_y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    #training
    count = 0
    data_y_test = []
    data_xa_test = []
    data_xb_test = []
    data_x_test = []

    data_y_final = []
    data_xa_final = []
    data_xb_final = []
    data_x_final = []

    for game_id in dict_games.keys():
        if count < 1:
            print("getting test cases")
            #get 1 game to test
            data_y_test,data_x_temp = get_data(game_id,dict_games,dict_heroes)
            for x in range(len(data_x_temp)):
                if x < 5:
                    data_xa_test.append(data_x_temp[x][0])
                else:
                    data_xb_test.append(data_x_temp[x][0])
                data_x_test.append(data_x_temp[x][1:])
            count +=1
            continue
        data_y,data_x_temp = get_data(game_id,dict_games,dict_heroes)
        data_xa = []
        data_xb = []
        data_x = []
        for x in range(len(data_x_temp)):
            if x < 5:
                data_xa.append(data_x_temp[x][0])
            else:
                data_xb.append(data_x_temp[x][0])
            data_x.append(data_x_temp[x][1:])
        # sess.run(train_step, feed_dict={in_xa: np.asarray(data_xa),in_xb: np.asarray(data_xb), in_x: np.asarray(data_x), out_y_: np.asarray(data_y)})
        data_y_final.append(data_y)
        data_xa_final.append(data_xa)
        data_xb_final.append(data_xb)
        data_x_final.append(data_x)

    sess.run(train_step, feed_dict={in_xa: np.asarray(data_xa_final),in_xb: np.asarray(data_xb_final), in_x: np.asarray(data_x_final), out_y_: np.asarray(data_y_final)})

    #test trained model
    correct_prediction = tf.equal(tf.argmax(out_y,1),tf.argmax(out_y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(accuracy, feed_dict={in_xa: np.asarray(data_xa_test),in_xb: np.asarray(data_xb_test), in_x: np.asarray(data_x_test), out_y_: np.asarray(data_y_test)}))

    sess.close()

def get_data(game_id,dict_games,dict_heroes):
    who_won = dict_games[game_id][0]
    if who_won:
        game_outcome = [1,0]
    else:
        game_outcome = [0,1]
    game_heroes = dict_games[game_id][1]
    winrates = []
    for x in range(0,10):
        hero_rate = []
        cur_hero = dict_heroes[game_heroes[x]]
        #first the general winrate in 0
        general_wr = float(cur_hero['total_win']/(cur_hero['total_loss']+cur_hero['total_win']))
        hero_rate.append(general_wr)
        if x < 5: #radiant
            #the specific winrates vs other team in 1,2,3,4,5
            for y in range(0,5):
                enemy_name = game_heroes[5+y]
                if enemy_name in cur_hero['vs_hero']['w'] and enemy_name in cur_hero['vs_hero']['l']:
                    specific_win = cur_hero['vs_hero']['w'][enemy_name]
                    specific_lose = cur_hero['vs_hero']['l'][enemy_name]
                    specific_wr = float(specific_win/(specific_lose+specific_win))
                else:
                    specific_wr = 0.5
                hero_rate.append(specific_wr)
        else: #dire
            #the specific winrates vs other team in 1,2,3,4,5
            for y in range(0,5):
                enemy_name = game_heroes[y]
                if enemy_name in cur_hero['vs_hero']['w'] and enemy_name in cur_hero['vs_hero']['l']:
                    specific_win = cur_hero['vs_hero']['w'][enemy_name]
                    specific_lose = cur_hero['vs_hero']['l'][enemy_name]
                    specific_wr = float(specific_win/(specific_lose+specific_win))
                else:
                    specific_wr = 0.5
                hero_rate.append(specific_wr)
        winrates.append(hero_rate)
    return game_outcome,winrates




def read_from_file():
    file_p = open("players.txt",'r+')
    file_h = open('heroes.txt','r+')
    file_g = open('games.txt','r+')

    list_players = json.load(file_p)
    dict_games = json.load(file_g)
    dict_heroes = json.load(file_h)

    file_h.close()
    file_p.close()
    file_g.close()

    return dict_games,list_players,dict_heroes

def main():
    dict_heroes = {}
    list_players = []
    dict_games = {}
    # initialise()
    dict_games,list_players,dict_heroes = read_from_file()
    # a,b = get_data('3260993188',dict_games,dict_heroes)
    # print(b)
    train(dict_games,dict_heroes)


if __name__ == '__main__':
    main()
