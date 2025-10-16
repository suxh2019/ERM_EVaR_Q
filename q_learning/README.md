## Reused the code from the previous paper
We reuse the code of Q-learning and other parts from the previous paper. We include the authors' name, paper title, and the link to the code of the paper.

Jia Lin Hau, Erick Delage, Esther Derman,Mohammad Ghavamzadeh, and Marek Petrik. Q-learning for Quantile MDPs: A Decomposition, Performance, and Convergence Analysis. The link to code is https://github.com/MonkieDein/DRA-Q-LA .


## Preliminary
(1) Install Julia and VSCode.
(2) Open VSCode and then open q_learning.
(3) ```julia requirement.jl``` :  Install all required julia libraries.

Note that explainations of some files are copied from the previous paper above.
## File Structure
- code/
    - experiment.jl : General functions for experiments, includes solveVI, evaluations, simplifyEvals and getTargetVaR.
    - onlineMDP.jl : Functions to execute policy in a monte carlo simulation. Type of policies include (Markov, QuantileDependent) as well as their time dependent variant.
    - riskMeasure.jl : Functions to create a discrete random variable and compute their risk (mean, min, max, q⁻, VaR, CVaR, ERM, EVaR).
    - TabMDP.jl : Define MDP and Objective structure. Solve nested, quantile, ERM, EVaR, and distributional (markovQuantile) Dynamic Program (DP) a.k.a Value Iteration (VI).
    - utils.jl : Commonly used functions for checking directory, decaying coefficient function and multi-dimensions function-applicator.
    - others/
        - csv2MDP.jl : Code to convert csv MDP files to MDP objects.
    - experiments/
         - q_cliff_0.2.jl: use algorithm 2 to Compute the optimal EVaR policy with EVaR risk level \alpha = 0.2 on cliff walking(CW) domain; the optimal policy is for Figure 1 and figure 3
         -  q_cliff_0.6.jl: use algorithm 2 to Compute the optimal EVaR policy with EVaR risk level \alpha = 0.6 on cliff walking(CW) domain; the optimal policy is for Figure 2 and figure 3
         -  c_d_estimation_cw.jl:  estimate the c value and d value in algorithm 3 on cliff walking(CW) domain
         -  c_d_estimation_gr.jl: Estimate the c value and d value in algorithm 3 on gambler ruin(GR) domain
         -  mean_std_cliff_0.2.jl:  Using algorithm 2 to Compute the optimal EVaR policy and calculate its EVaR value on cliff walking(CW) domain.  Manually change the random seed and the number of samples to calculate the EVaR values. EVaR risk level α = 0.2. This is for figure 4.
         -  plot_lp_q_cliff.jl: plot the difference in EVaR values computed by linear program and Q-learning on cliff walking(CW) domain. The EVaR risk level α = 0.2. This is for figure 5.
         -  plot_lp_q_gambler.jl: plot the difference in EVaR values computed by linear program and Q-learning on gambler ruin(GR) domain. The EVaR risk level α = 0.2. This is for figure 6.
         - q_cliff_multiple.jl: Compute EVaR values for different number of samples on cliff walking(CW) domain. The risk level α of EVaR is 0.2. This is for figure 4
         - q_gambler_0.2.jl: Compute EVaR values for different number of samples on gambler ruin(GR)domain. The risk level α of EVaR is 0.2. This is for figure 5.

- experiment/
    - domain/
        - csv/ : CSV file containing domain transition and reward.
        - domains_info.csv : CSV file containing discount factor and initial state.
        - MDP/ : MDP JLD2 files (Generate from: ``` code/others/csv2MDP.jl```)

- figures/
   -figure 1/:  q_cliff_0.2.jl
   -figure 2/: q_cliff_0.6.jl
   -figure 3/: q_cliff_0.2.jl ,q_cliff_0.6.jl,simulate_returns.jl
   -figure 4/:  mean_std_cliff_0.2.jl,mean-std.jl
   -figure 5/:  plot_lp_q_cliff.jl,  q_cliff_multiple.jl ,LP_cliff.jl
   -figure 6/:   plot_lp_q_gambler.jl, q_gambler_0.2.jl,  LP_gambler.jl






- -----------|-------------------------------------|--------------------------|
-Figures      |                q_learning           |     linear_program       |
-------------|-------------------------------------|--------------------------|
-Figure 1     |              q_cliff_0.2.jl         |                          |
-------------|-------------------------------------|--------------------------|
-Figure 2     |              q_cliff_0.6.jl         |                          |
-------------|-------------------------------------|--------------------------|
-Figure 3     |              q_cliff_0.2.jl         |     simulate_returns.jl  |     
             |              q_cliff_0.6.jl         |                          |
-------------|-------------------------------------|--------------------------|
-Figure 4     |              mean_std_cliff_0.2.jl  |     mean-std.jl          |
-------------|-------------------------------------|--------------------------|
-Figure 5     |              plot_lp_q_cliff.jl     |    LP_cliff.jl           |
             |             q_cliff_multiple.jl     |                          |
-------------|-------------------------------------|--------------------------|
-Figure 6     |              plot_lp_q_gambler.jl   |    LP_gambler.jl         |
             |              q_gambler_0.2.jl       |                          |
-------------|-------------------------------------|--------------------------|