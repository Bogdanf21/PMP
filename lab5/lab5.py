from pgmpy.models import BayesianNetwork as network
from pgmpy.factors.discrete import TabularCPD

betting_model = network([
    ("P1_draw", "P2_draw"),
    ("P1_draw", "P1_call"),
    ("P2_draw", "P2_call"),
    ("P1_call2", "P2_call"),
    ("P1_call2", "P1_call"),
    ("P1_draw", "win"),
    ("P2_draw", "win"),
    ("P1_call", "earn"),
    ("P2_call", "earn"),
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
    variable="P2_call",
    variable_card=2,
    values=[[0.95, 0.75, 0.6, 0.3, 0.03], [0.05, 0.25, 0.4, 0.7, 0.97]],
    evidence=["P1_draw"],
    evidence_card=[5]
)

cpd_P2_call = TabularCPD(
    variable="P2_call",
    variable_card=2,
    values=[[1, 1, 0.85, 0.8, 0.5, 0.45, 0.2, 0.15, 0, 0], [0, 0, 0.15, 0.2, 0.5, 0.55, 0.8, 0.85, 1, 1]],
    evidence=["P2_draw", "P1_call"],
    evidence_card=[5, 2]
)
cpd_P1_call2 = TabularCPD(
    variable="P2_call2",
    variable_card=2,
    values=[[1, 0, 1, 0, 0.85, 0, 0.8, 0, 0.5, 0, 0.45, 0, 0.2, 0, 0.15, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.15, 0, 0.2, 0, 0.5, 0, 0.55, 0, 0.8, 0, 0.85, 0, 1, 0, 1, 0]],
    evidence=["P1_draw", "P2_call", "P1_call"],
    evidence_card=[5, 2, 2]
)

print(cpd_P1_call)
print(cpd_P1_draw)
print(cpd_P1_call2)
print(cpd_P2_draw)
print(cpd_P2_call)

betting_model.add_cpds(cpd_P1_call, cpd_P1_draw, cpd_P1_call2, cpd_P2_draw, cpd_P2_call)
betting_model.check_model()
