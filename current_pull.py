import requests




r = requests.get('https://www.baseball-reference.com/leagues/majors/2023.shtml#teams_standard_batting')

print(r.text)