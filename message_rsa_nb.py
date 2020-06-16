# %%

%run rsa/message_rsa.py

# %%

from exh import *

# %%

rsa = MessageRSA(Universe(fs = [a, b]), 
                 messages = [
                             ("",          [a | ~a]),
                             ("or",        [a | b]),
                             ("or both",   [a | b]),
                             ("both",      [a & b])
                            ],
                  states = [
                     ("none",     ~a & ~b),
                     ("excl",     (a | b) & ~ (a & b)),
                     ("ignorant", a | b),
                     ("all",      a & b)
                  ],
                 costs = np.array([4, 1, 3, 1], dtype = "float"),
                 rationality = 1.)


# %%

rsa.compute(2)
rsa.heatmap_listener()
rsa.heatmap_speaker()

# %%

rsa.compute(1, False)
rsa.heatmap_listener()
# rsa.heatmap_speaker()

# %%

rsa.compute(1, False)
rsa.heatmap_speaker()
# rsa.heatmap_listener()




# %%

def justsome(var, scope):
	return (E(var) > scope) & ~(A(var) > scope)

d = Pred(3, name = "d", depends = "x")



rsa = MessageRSA(Universe(fs = [Ex > d]), 
                 messages = [
                             ("",                  [Ex > d | ~ (Ex > d)]),
                             ("some",              [Ex > d, justsome("x", d)]),
                             ("some or all",       [Ex > d]),
                             ("all",               [Ax > d])
                            ],
                 states = [
                    ("none",     ~(Ex > d)),
                    ("excl",     justsome("x", d)),
                    ("ignorant", Ex > d),
                    ("all",     Ax > d)
                 ],
                 costs = np.array([4, 1, 3, 1], dtype = "float"))

# %%

rsa.compute(2)
rsa.heatmap_listener()
rsa.heatmap_speaker()

# %%

rsa.compute(2, False)
rsa.heatmap_listener()
rsa.heatmap_speaker()





# %%