from utils import *
from plots import * 
from expectation_values import *

#surface_plots(2,params=None)
#ratio_plots(2)
#sanity check(check with the formulas in the nonclassicality paper that results are the same)
N=2
sigma= V_tms([0.5,2],[0],[0,0], params=None,ordering='xxpp')
#print(sigma)
#print(expectationvalue(sigma,['adag','adag','a','a'],[1,1,1,1])/expectationvalue(sigma,['adag','a'],[1,1]))
print(K_ng(sigma,[-1]))
print(expvalN_ng(sigma,[-1]))
beep()