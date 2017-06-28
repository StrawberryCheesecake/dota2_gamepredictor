import dota2api
api = dota2api.Initialise("BD04FDD08330AF1982CCE0280E4086C9")
heros = api.get_heroes()

# print(heros['heroes'][0]['localized_name'])
# print(heros['heroes'][0]['id'])

history = api.get_match_history('404163973')
# print(history['matches'][0]['match_id'])

game = api.get_match_details(history['matches'][0]['match_id'])
for player in game['players']:
    print(player['hero_name'])
