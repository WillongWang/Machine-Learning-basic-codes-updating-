from z3 import *

rank_Lisa = Int("rank_Lisa")
rank_Mary = Int("rank_Mary")
rank_Bob = Int("rank_Bob")
rank_Jim = Int("rank_Jim")

solver = Solver()

solver.add(And(1 <= rank_Lisa, rank_Lisa <= 4))
solver.add(And(1 <= rank_Mary, rank_Mary <= 4))
solver.add(And(1 <= rank_Bob, rank_Bob <= 4))
solver.add(And(1 <= rank_Jim, rank_Jim <= 4))
solver.add(Distinct(rank_Lisa, rank_Mary, rank_Bob, rank_Jim))
solver.add(Abs(rank_Lisa - rank_Bob) != 1)

is_bio_Lisa = Bool("is_bio_Lisa")
is_bio_Mary = Bool("is_bio_Mary") #会导致is_bio_Mary,is_bio_Bob,is_bio_Jim为none 
is_bio_Bob = Bool("is_bio_Bob")
is_bio_Jim = Bool("is_bio_Jim")

#solver.add(is_bio_Jim)
#solver.add(Not(is_bio_Jim))

solver.add(Or(is_bio_Lisa, is_bio_Mary))
solver.add(Or(
    And(is_bio_Lisa, rank_Jim == rank_Lisa - 1),
    And(is_bio_Mary, rank_Jim == rank_Mary - 1),
    And(is_bio_Bob, rank_Jim == rank_Bob - 1)
))

solver.add(rank_Bob == rank_Jim - 1)
solver.add(Or(rank_Lisa == 1, rank_Mary == 1))


rankings = []
while solver.check() == sat:
    model = solver.model() #每次只给一个答案，不全
    print("model: ",model)
    ranking = {
        "Lisa": model[rank_Lisa].as_long(),
        "Mary": model[rank_Mary].as_long(),
        "Bob": model[rank_Bob].as_long(),
        "Jim": model[rank_Jim].as_long(),
        "are Lisa,Mary,Bob,Jim Bio Majors?": [model[is_bio_Lisa],model[is_bio_Mary],model[is_bio_Bob],model[is_bio_Jim]]
    }
    if ranking in rankings:
        break
    rankings.append(ranking)
    print(f"Solution found: {ranking}")
    solver.add(Or(
        rank_Lisa != model[rank_Lisa],
        rank_Mary != model[rank_Mary],
        rank_Bob != model[rank_Bob],
        rank_Jim != model[rank_Jim],
        is_bio_Lisa != model[is_bio_Lisa],
        is_bio_Mary != model[is_bio_Mary],
        is_bio_Bob != model[is_bio_Bob],
        is_bio_Jim != model[is_bio_Jim]
    ))

if rankings:
    for i, r in enumerate(rankings, 1):
        print(f"Ranking {i}: {r}")
else:
    print("No valid rankings found!")

#Solution found: {'Lisa': 4, 'Mary': 1, 'Bob': 2, 'Jim': 3, 'are Lisa,Mary,Bob,Jim Bio Majors?': [True, True/False, True/False, True/False]}