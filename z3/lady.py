from z3 import *

l1 = Bool("l1") #lady in room 1
l2 = Bool("l2")
l3 = Bool("l3")

sign1 = Bool("sign1") #Room I sign: "A TIGER IS IN THIS ROOM"
sign2 = Bool("sign2") #Room II sign: "A LADY IS IN THIS ROOM"
sign3 = Bool("sign3") #Room III sign: "A TIGER IS IN ROOM II"

solver = Solver()
solver.add(Or(
    And(l1, Not(l2), Not(l3)),
    And(Not(l1), l2, Not(l3)),
    And(Not(l1), Not(l2), l3),
))
solver.add(sign1 == Not(l1))  
solver.add(sign2 == l2)       
solver.add(sign3 == Not(l2))  
solver.add(AtMost(sign1, sign2, sign3, 1))

# Solve the problem
if solver.check() == sat:
    model = solver.model()
    print("Room 1 contains the lady:", model.evaluate(l1))
    print("Room 2 contains the lady:", model.evaluate(l2))
    print("Room 3 contains the lady:", model.evaluate(l3))
else:
    print("No solution found!")

#lady in room 1

