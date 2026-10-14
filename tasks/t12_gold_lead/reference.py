ROWS = [{'game': 'G01', 'team': 'GiantX', 'side': 'blue', 'result': 'loss', 'gold_diff_15': 1060},
 {'game': 'G01', 'team': 'Movistar KOI', 'side': 'red', 'result': 'win', 'gold_diff_15': -1060},
 {'game': 'G02', 'team': 'Movistar KOI', 'side': 'blue', 'result': 'win', 'gold_diff_15': 2660},
 {'game': 'G02', 'team': 'GiantX', 'side': 'red', 'result': 'loss', 'gold_diff_15': -2660},
 {'game': 'G03', 'team': 'Karmine Corp', 'side': 'blue', 'result': 'win', 'gold_diff_15': 75},
 {'game': 'G03', 'team': 'Fnatic', 'side': 'red', 'result': 'loss', 'gold_diff_15': -75},
 {'game': 'G04', 'team': 'Movistar KOI', 'side': 'blue', 'result': 'win', 'gold_diff_15': -2285},
 {'game': 'G04', 'team': 'Fnatic', 'side': 'red', 'result': 'loss', 'gold_diff_15': 2285},
 {'game': 'G05', 'team': 'SK Gaming', 'side': 'blue', 'result': 'loss', 'gold_diff_15': -2251},
 {'game': 'G05', 'team': 'Team Vitality', 'side': 'red', 'result': 'win', 'gold_diff_15': 2251},
 {'game': 'G06', 'team': 'Team Heretics', 'side': 'blue', 'result': 'loss', 'gold_diff_15': -4163},
 {'game': 'G06', 'team': 'GiantX', 'side': 'red', 'result': 'win', 'gold_diff_15': 4163},
 {'game': 'G07', 'team': 'Team Heretics', 'side': 'blue', 'result': 'loss', 'gold_diff_15': -556},
 {'game': 'G07', 'team': 'GiantX', 'side': 'red', 'result': 'win', 'gold_diff_15': 556},
 {'game': 'G08', 'team': 'GiantX', 'side': 'blue', 'result': 'loss', 'gold_diff_15': 1777},
 {'game': 'G08', 'team': 'Karmine Corp', 'side': 'red', 'result': 'win', 'gold_diff_15': -1777},
 {'game': 'G09', 'team': 'Team Heretics', 'side': 'blue', 'result': 'loss', 'gold_diff_15': -3144},
 {'game': 'G09', 'team': 'Team Vitality', 'side': 'red', 'result': 'win', 'gold_diff_15': 3144},
 {'game': 'G10', 'team': 'Movistar KOI', 'side': 'blue', 'result': 'loss', 'gold_diff_15': -1096},
 {'game': 'G10', 'team': 'G2 Esports', 'side': 'red', 'result': 'win', 'gold_diff_15': 1096}]


def q1():
    wins = [r['gold_diff_15'] for r in ROWS if r['result'] == 'win']
    return sum(wins) / len(wins)


def q2():
    return sum(r['result'] == 'win' and r['gold_diff_15'] < 0 for r in ROWS)


def q3():
    ahead = [r for r in ROWS if r['gold_diff_15'] > 0]
    return sum(r['result'] == 'win' for r in ahead) / len(ahead)
