ROWS = [{'player': 'Forest', 'champion': 'Viego', 'result': 'win', 'deaths': 3},
 {'player': 'Popon', 'champion': 'Vi', 'result': 'win', 'deaths': 0},
 {'player': 'Racon', 'champion': 'Nocturne', 'result': 'win', 'deaths': 1},
 {'player': 'HALdesu', 'champion': "Bel'Veth", 'result': 'loss', 'deaths': 4},
 {'player': 'Norwegian', 'champion': 'Viego', 'result': 'win', 'deaths': 0},
 {'player': 'VEL Skinny', 'champion': 'Vi', 'result': 'loss', 'deaths': 5},
 {'player': 'Brucee', 'champion': 'Skarner', 'result': 'win', 'deaths': 1},
 {'player': '89', 'champion': 'Vi', 'result': 'loss', 'deaths': 6},
 {'player': 'DM SUISEI', 'champion': 'Mordekaiser', 'result': 'win', 'deaths': 2},
 {'player': 'RPG Shawshank', 'champion': 'Xin Zhao', 'result': 'loss', 'deaths': 5},
 {'player': 'EL', 'champion': 'Zyra', 'result': 'win', 'deaths': 0},
 {'player': 'DG hatinyang', 'champion': 'Xin Zhao', 'result': 'win', 'deaths': 4},
 {'player': 'Sassappel', 'champion': 'Vi', 'result': 'win', 'deaths': 3},
 {'player': 'Kiwi', 'champion': 'Viego', 'result': 'win', 'deaths': 3},
 {'player': 'Mutton', 'champion': 'Vi', 'result': 'loss', 'deaths': 6},
 {'player': 'DG socceryakyu', 'champion': 'Nocturne', 'result': 'win', 'deaths': 3},
 {'player': 'HNS Skinny', 'champion': 'Xin Zhao', 'result': 'loss', 'deaths': 6},
 {'player': 'VLY hachim1', 'champion': 'Vi', 'result': 'win', 'deaths': 5},
 {'player': 'SQG Lunatic', 'champion': 'Yone', 'result': 'loss', 'deaths': 5},
 {'player': 'QTD Van1', 'champion': 'Vi', 'result': 'win', 'deaths': 2}]


def q1():
    return len({r['champion'] for r in ROWS})


def q2():
    games = {}
    for r in ROWS:
        games.setdefault(r['champion'], []).append(r['result'] == 'win')
    rates = {c: sum(g) / len(g) for c, g in games.items() if len(g) >= 2}
    return max(rates, key=rates.get) if rates else None


def q3():
    losses = [r['deaths'] for r in ROWS if r['result'] == 'loss']
    return sum(losses) / len(losses)
