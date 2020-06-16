# %%

%run rsa/simple_rsa.py

# %%

from exh import *

# %%
################## SYMMMETRY PROBLEM ###############################
rsa = RSA(Universe(fs = [a, b]), 
          alts = [
                 a | ~a,
                 a | b,
                 (a | b) & ~(a & b), 
                 a & b
                 ],
           states = [
              ("none",     ~a & ~b),
              ("excl",     (a | b) & ~ (a & b)),
              ("ignorant", a | b),
              ("all",      a & b)
           ],
          costs = np.array([4, 1, 3, 1], dtype = "float"),
          rationality = 10.)


# %%

rsa.compute(3)
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
################## M IMPLICATURES  #############################



# %%