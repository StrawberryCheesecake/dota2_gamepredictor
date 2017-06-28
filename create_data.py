import sys
import dota2api
import json

api = dota2api.Initialise("BD04FDD08330AF1982CCE0280E4086C9")


def initialise(dict_heroes):
    v_heroes = api.get_heroes()
    for hero in v_heroes['heroes']:
        dict_heroes[hero['localized_name']] = {'ID':hero['id'],'total_win':0,'total_loss':0,'vs_hero':{'w':{},'l':{}}}

def travel_player(player_id,p_history,dict_games,dict_heroes,list_players):
    for game in p_history['matches']:
        if (game['match_id'] in dict_games):
            continue
        game_details = api.get_match_details(game['match_id'])
        #bot game we dont care skip it
        if (game_details['human_players'] != 10):
            continue

        radiant_win = game_details['radiant_win']
        game_heroes = []

        count = 0
        for player in game_details['players']:
            game_heroes.append(player['hero_name'])
            #radiant team
            if (count <= 5):
                if radiant_win:
                    dict_heroes[player['hero_name']]['total_win'] += 1
                    for x in range(5,10):
                        if game_details['players'][x]['hero_name'] in dict_heroes[player['hero_name']]['vs_hero']['w']:
                            dict_heroes[player['hero_name']]['vs_hero']['w'][game_details['players'][x]['hero_name']] += 1
                        else:
                            dict_heroes[player['hero_name']]['vs_hero']['w'][game_details['players'][x]['hero_name']] = 1
                else:
                    dict_heroes[player['hero_name']]['total_loss'] += 1
                    for x in range(5,10):
                        if game_details['players'][x]['hero_name'] in dict_heroes[player['hero_name']]['vs_hero']['l']:
                            dict_heroes[player['hero_name']]['vs_hero']['l'][game_details['players'][x]['hero_name']] += 1
                        else:
                            dict_heroes[player['hero_name']]['vs_hero']['l'][game_details['players'][x]['hero_name']] = 1
            #dire team
            else:
                if radiant_win:
                    dict_heroes[player['hero_name']]['total_loss'] += 1
                    for x in range(0,5):
                        if game_details['players'][x]['hero_name'] in dict_heroes[player['hero_name']]['vs_hero']['l']:
                            dict_heroes[player['hero_name']]['vs_hero']['l'][game_details['players'][x]['hero_name']] += 1
                        else:
                            dict_heroes[player['hero_name']]['vs_hero']['l'][game_details['players'][x]['hero_name']] = 1
                else:
                    dict_heroes[player['hero_name']]['total_win'] += 1
                    for x in range(0,5):
                        if game_details['players'][x]['hero_name'] in dict_heroes[player['hero_name']]['vs_hero']['w']:
                            dict_heroes[player['hero_name']]['vs_hero']['w'][game_details['players'][x]['hero_name']] += 1
                        else:
                            dict_heroes[player['hero_name']]['vs_hero']['w'][game_details['players'][x]['hero_name']] = 1
            #keep track of teams, first 5 are radiant
            count += 1
        #got all the data from the game add the game to the list
        dict_games[game['match_id']] = [radiant_win,game_heroes]


def get_players(dict_games,dict_heroes,list_players):
    count = 0
    pro_plays = api.get_top_live_games()
    for game in pro_plays['game_list']:
        for player in game['players']:
            count += 1
            if (count % 10 == 0):
                write_to_file(dict_games,dict_heroes,list_players)
            player_id = player['account_id']
            if player_id in list_players:
                print(str(player_id) + "player already loaded")
                continue
            list_players.append(player_id)
            print("added player:"+str(player_id))
            try:
                p_history = api.get_match_history(player_id)
            except:
                print("error getting match history")
                continue
            try:
                travel_player(player_id,p_history,dict_games,dict_heroes,list_players)
            except:
                print("error traveling player")
                print(sys.exc_info()[0])
                continue

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

def write_to_file(dict_games,dict_heroes,list_players):
    print("Writing Data")
    file_p = open("players.txt",'w+')
    file_h = open('heroes.txt','w+')
    file_g = open('games.txt','w+')

    print(len(dict_games.keys()))

    json.dump(list_players,file_p)
    json.dump(dict_heroes,file_h)
    json.dump(dict_games,file_g)

    file_h.close()
    file_p.close()
    file_g.close()

def main():
    dict_heroes = {}
    list_players = []
    dict_games = {}
    initialise(dict_heroes)
    dict_games,list_players,dict_heroes = read_from_file()
    get_players(dict_games,dict_heroes,list_players)
    write_to_file(dict_games,dict_heroes,list_players)

if __name__ == '__main__':
    main()
