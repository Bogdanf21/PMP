from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork as network
from pgmpy.factors.discrete import TabularCPD


def gain(e0, e1, e2, e3, e4):
    return -2 * e0 - e1 + e3 + 2 * e4  # expected gain for p1


betting_model = network([
    ("P1_draw", "P2_draw"),
    ("P1_draw", "P1_call"),
    ("P2_draw", "P2_call"),
    ("P1_call", "P2_call"),
    ("P1_call", "P1_call2"),
    ("P2_call", "P1_call2"),
    ("P1_draw", "P1_call2"),
    ("P1_draw", "win"),
    ("P2_draw", "win"),
    ("P1_call_def", "earn"),
    ("P2_call", "earn"),
    ("win", 'earn'),
    ("P1_call", "P1_call_def"),
    ("P1_call2", "P1_call_def")
])
cpd_P1_draw = TabularCPD(
    variable="P1_draw", variable_card=5, values=[[0.2], [0.2], [0.2], [0.2], [0.2]]
)

cpd_P2_draw = TabularCPD(
    variable="P2_draw",
    variable_card=5,
    values=[[0, 0.25, 0.25, 0.25, 0.25],
            [0.25, 0, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0, 0.25],
            [0.25, 0.25, 0.25, 0.25, 0]],
    evidence=["P1_draw"],
    evidence_card=[5]
)
cpd_P1_call = TabularCPD(
    variable="P1_call",
    variable_card=2,
    values=[[0.95, 0.6, 0.55, 0.3, 0.1], [0.1, 0.2, 0.3, 0.8, 0.99]],
    # changed for b)
    evidence=["P1_draw"],
    evidence_card=[5]
)

cpd_P2_call = TabularCPD(
    variable="P2_call",
    variable_card=2,
    values=[[1, 1, 0.8, 0.75, 0.4, 0.35, 0.2, 0.15, 0, 0], [0, 0, 0.15, 0.2, 0.45, 0.45, 0.75, 0.8, 1, 1]],
    evidence=["P2_draw", "P1_call"],
    evidence_card=[5, 2]
)
cpd_P1_call2 = TabularCPD(
    variable="P1_call2",
    variable_card=2,
    values=[[1, 1, 1, 1, 0.85, 1, 0.8, 1, 0.5, 1, 0.45, 1, 0.2, 1, 0.15, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0.15, 0, 0.2, 0, 0.5, 0, 0.55, 0, 0.8, 0, 0.85, 0, 1, 0, 1, 0]],
    # changed for b)
    evidence=["P1_draw", "P2_call", "P1_call"],
    evidence_card=[5, 2, 2]
)
cpd_P1_call_def = TabularCPD(
    variable="P1_call_def",
    variable_card=2,
    values=[[1, 0, 0, 0], [0, 1, 1, 1]],  # if p1 betted at least once
    evidence=["P1_call", "P1_call2"],
    evidence_card=[2, 2]
)
cpd_win = TabularCPD(
    variable="win",
    variable_card=2,
    values=[[0, 0, 0, 0, 0,
             1, 0, 0, 0, 0,
             1, 1, 0, 0, 0,
             1, 1, 1, 0, 0,
             1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1,
             0, 1, 1, 1, 1,
             0, 0, 1, 1, 1,
             0, 0, 0, 1, 1,
             0, 0, 0, 0, 1]],
    evidence=["P1_draw", "P2_draw"],
    evidence_card=[5, 5]
)
cpd_earn = TabularCPD(
    variable="earn",
    variable_card=5,
    values=[[0, 0, 0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]],
    evidence=["win", "P1_call_def", "P2_call"],
    evidence_card=[2, 2, 2]
)

print(cpd_P1_draw)
print(cpd_P2_draw)
print(cpd_P1_call)
print(cpd_P1_call2)
print(cpd_P2_call)
print(cpd_earn)
print(cpd_P1_call_def)
print(cpd_win)


betting_model.add_cpds(cpd_P1_draw, cpd_P2_draw, cpd_P1_call, cpd_P2_call, cpd_P1_call2, cpd_win, cpd_earn,
                       cpd_P1_call_def)
infer = VariableElimination(betting_model)
q1 = infer.query(variables=["earn"], evidence={"P1_draw": 3, "P1_call_def": 1})
q2 = infer.query(variables=["earn"], evidence={"P1_draw": 3, "P1_call_def": 0})
print(gain(q1.values[0], q1.values[1], q1.values[2], q1.values[3], q1.values[4]) -
      gain(q2.values[0], q2.values[1], q2.values[2], q2.values[3],
                    q2.values[4]))  # >0 so first gives p1 an advantage
q1 = infer.query(variables=["earn"], evidence={"P2_draw": 2, "P1_call_def": 1, "P2_call": 1})
q2 = infer.query(variables=["earn"], evidence={"P2_draw": 2, "P1_call_def": 1, "P2_call": 0})
print(gain(q1.values[0], q1.values[1], q1.values[2], q1.values[3], q1.values[4]) -
      gain(q2.values[0], q2.values[1], q2.values[2], q2.values[3],
                    q2.values[4]))  # >0 so p2 should leave game
# In order for player 2 to gain an advantage when betting (at point b), we can reverse player 1's strategy (to bet
# more often when he has weaker cards that is, 1-P(call) for each decision of player 1 compared to how they are set
# now, the inference gives the result below < 0:
# There is no instance in which p1 is advised to give up with a king of spades

print(gain(0.7293, 0, 0, 0, 0.2707) - gain(0, 0, 0, 1, 0))
