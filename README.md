# Spontaneous-Optimal-Mixing
This repository contains the flow solver, viscometric curve analysis, and worldline tracking code for the data presented in "Spontaneous Optimal Mixing via Defect-Vortex Coupling in Confined Active Nematics"

flow-solver:
Produces Q-tensor and flow field data for a range of nematic coherence length and active length scales. Boundaries are modified in "set_boundary" where circular, or epitrochoid boundaries can be selected.
the "net_charge" variable in "apply_Q_boundary_conditions" should be set to the variable 'q' as presented in the manuscript, with 2/2 corresponding to tangential anchoring. This also produces the videos shown in Supplemental Videos 7 and 8.

braid_tracker:
Produces a visualization of trajectories and worldlines shown in Supplemental Videos 1, 2, and 3 given Q-tensor data and flow field data.

viscometric_analysis:
Produces the plots shown in Supplemental Videos 4, 5, and 6 given Q-tensor data and flow field data.
