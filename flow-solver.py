#!/usr/bin/env python
# coding: utf-8

# # 2D active nematic hydrodynamics simulation

# ## About

# This notebook documents a trial finite-difference approach to simulating Beris-Edwards nematic hydrodynamics with activity in 2D. The coupled fields being simulated are the nematic $Q$-tensor and an incompressible velocity field ${\bf u}$.
#
# The flow solver portion of the code is adapted from the ["12 Steps to Navier-Stokes"](https://github.com/barbagroup/CFDPython) tutorial.

# ### Latest changes
#
# * In "Imports and Setup", replace shell `mkdir` command with `os.makedirs` call
# * Rename `Π` to `Π_S` and `skew_symmetric_stress` to `Π_A`
# * Remove viscosity term from pressure relaxation. (Barba's tutorial doesn't include it, rightly noting that it's among higher-order derivative terms to be ignored.)
# * Clarify pressure relaxation writeup.
# * Apply the boundary conditions of `Q` to `H` tensor before using `H` to calculate stress (thanks to Brandon for this)
# * Remove errors in upwind convective term calculation left over from switching between first- and second-order versions (thanks to Kevin for this)
# * In parameters, remove explicit use of `Lscale` to avoid confusion about length scales
# * Add random seed to parameters for reproducibility
# * Correct mistake in adding randomness to director initialization (won't affect fully randomized angles)
# * Convert .ipynb notebook to .py file (same content) for better version control and for scripting usage

# In[ ]:


# Uncomment and run this line to convert notebook to .py file
# ! jupyter nbconvert --to script flow_solver-nematic.ipynb


# ### Usage notes

# * The live animation runs more smoothly in Chrome than in Firefox.
# * I recommend creating a conda environment to use this notebook. Assuming you have Anaconda installed, from a command line, type:
#
#             conda create -n "myenv"
#             conda activate myenv
#             conda install numpy scipy numba matplotlib ipympl
#             conda install -c numba icc_rt
#             python3 -m ipykernel install --user --name="myenv"
#     For explanation of the `icc_rt` line, see [here](https://numba.pydata.org/numba-doc/latest/user/performance-tips.html). The last line makes our environment available as a Jupyter notebook kernel; make sure to choose "myenv" as the kernel when you open this notebook in Jupyter.

# ### TODO

# * saddle-splay plots: mask boundary points
# * look into replacing stopping condition for pressure-Poisson relaxation with threshold total div u.
# * Beris Edwards theory: does S_0 need to be 1? Is the theory right?

# ## Imports and Setup

# In[ ]:


import numpy as np
import numba as nb
from IPython.core.display import display

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as colors
import matplotlib.cm as cm
import glob
import os
import scipy

# number of threads for numba multithreading;
# generally set to one less than number of threads available
nb.set_num_threads(25)

# Matplotlib settings
rcParams['font.size'] = 14
for prop in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
    rcParams[prop] = 'black'
fs = {'fontsize': 24}
rcParams['font.family'] = 'sans-serif'
rcParams['text.usetex'] = False


# Create Results and Images subfolders if they don't exist
def make_directory_if_needed(pathname):
    if len(glob.glob(pathname)) == 0:
        os.makedirs(pathname)
    return pathname


resultspath = "./"
make_directory_if_needed(resultspath);

imgpath = "./"
make_directory_if_needed(imgpath);


# ## Equations of motion

# Our dynamical variables are a second-rank, traceless, symmetric tensor field $Q_{ij}({\mathbf r})$ and a flow velocity field $\mathbf{u}(\mathbf r)$.

# ### $Q$ update

# $Q$ is updated according to
#
# $$ \frac{D Q_{ij}}{Dt} = \frac{\partial Q_{ij}}{\partial t} + u_k \partial_k Q_{ij}  = \gamma^{-1} H_{ij} + S_{ij}  $$
#
# Rearranging to solve for $\partial Q_{ij} / \partial t$, we use a second-order upwind finite difference calculation in the convective term to do the update.

# In[ ]:


@nb.njit(parallel=True, fastmath=True, nogil=True)
def get_Q_update(dQ, Q, H, S, u, γ, bounds):
    """ Calculate update to Q tensor """
    # S2 = 4*(Q[:,:,0]**2 + Q[:,:,1]**2)
    # f = 2*(np.exp(-S2[:]/2/(1*.3)**2)-1)/S2[:]
    # TrQJH = 2*(H[:,:,0]*Q[:,:,1]-H[:,:,1]*Q[:,:,0])
    # Hprime = H[:]
    # Hprime[:,:,0] += +f*TrQJH*Q[:,:,1]
    # Hprime[:,:,1] += -f*TrQJH*Q[:,:,0]

    # dQ[:] = (1/γ) * Hprime[:] + S[:]
    dQ[:] = (1 / γ) * H[:] + S[:]
    upwind_advective_term(u, Q, dQ, bounds)  # subtract (u•∇)Q
    # f = open("hdat.txt", 'a+')
    # for x in range(Lx):
    #     for y in range(Ly):
    #         f.write(f"{}\n")
    # f.close()
    # obj_as_table = ((H[:,:,0]**2 + H[:,:,1]**2)**0.5 - (S[:,:,0]**2 + S[:,:,1]**2)**0.5).flatten()
    # np.savetxt('/Results/hdat/' + (str(stepcount)+'.txt').replace(' ','0'), obj_as_table)


# ### Velocity update

# The velocity field will update according to Navier-Stokes:
#
# $$ \frac{\partial {\bf u}}{\partial t}  = - ({\bf u} \cdot \nabla) {\bf u} + \nu \nabla^2 {\bf u}  + \frac{1}{\rho} {\bf F} - \frac{1}{\rho} \nabla p $$
#
# On the RHS,
# * The first term is the convective term. *Note: Giomi PRX 2015 ignores this because it's higher-order in ${\bf u}$ and we are assuming low Reynolds number. We will keep this term.*
# * The second term is the viscous term with kinematic viscosity $\nu$.
# * In the third term, ${\bf F}$ is the force density of elastic and active stresses, related to the stress tensor $\Pi = \Pi^E + \Pi^A$ by
#     $$ \begin{align}
#         F_i &= \partial_j \Pi_{ij} \\
#         \Rightarrow F_x &= \partial_x \Pi_{xx} + \partial_y \Pi_{xy}, \\
#         F_y &= \partial_x \Pi_{yx} + \partial_y \Pi_{yy}
#     \end{align} $$
# * In the fourth term, the presure enforces incompressibility,
# $$ \nabla \cdot {\bf u} = 0 $$
#
# Any part of $\Pi$ proportional to the identity would produce a non-zero $\nabla \cdot \mathbf{F}$ and so will be automatically canceled by $\nabla p$ to ensure $\nabla \cdot \mathbf{u} = 0$.
#
# Any antisymmetric part of $\Pi$ ($\Pi_{xx}=\Pi_{yy}=0$, $\Pi_{yx} = - \Pi_{xy}$)) will give a contribution
# $$ \begin{align}
#     F_x &= \partial_y \Pi_{xy} \\
#     F_y &= -\partial_x \Pi_{xy}
#     \end{align}
# $$
# whose divergence is manifestly zero; thus only the symmetric part of $\Pi$ contributes to the pressure

# In[ ]:


@nb.njit
def get_u_update(dudt, u, p, ρ, Π_S, Π_A, ν, bounds):
    """ update flow field (after solving for pressure field) """
    Laplacian_vector(u, dudt, bounds, coeff=ν)  # viscous term
    upwind_advective_term(u, u, dudt, bounds)  # convective term
    # pressure and stress tensor (active + elastic) contributions
    u_update_p_Π_terms(dudt, p, ρ, Π_S, Π_A, bounds)


@nb.njit(parallel=True, fastmath=True, nogil=True)
def u_update_p_Π_terms(dudt, p, ρ, Π_S, Π_A, bounds):
    """ add velocity update terms from pressure and active+elastic forces """
    Lx, Ly = p.shape[:2]
    for point in nb.prange(len(bounds)):
        x, y = bounds[point, 0], bounds[point, 1]
        xup = (x + 1) % Lx
        xdn = (x - 1)
        yup = (y + 1) % Ly
        ydn = (y - 1)
        # pressure + force density from elastic and active stresses:
        dudt[x, y, 0] += 0.5 * 1 / ρ * (
                -(p[xup, y] - p[xdn, y])  # [-grad(p)]_x
                # F_x = dx Πxx + dy Πxy:
                + (Π_S[xup, y, 0] - Π_S[xdn, y, 0])  # dx Πxx
                + (  # dy Πxy
                        (Π_S[x, yup, 1] + Π_A[x, yup])
                        - (Π_S[x, ydn, 1] + Π_A[x, ydn])
                )
        )
        dudt[x, y, 1] += 0.5 * 1 / ρ * (
                -(p[x, yup] - p[x, ydn])  # [-grad(p)]_y
                # F_y = dx Πyx + dy Πyy = dx Πyx - dy Πxx :
                + (  # dx Πyx
                        (Π_S[xup, y, 1] - Π_A[xup, y])
                        - (Π_S[xdn, y, 1] - Π_A[xdn, y])
                )
                - (Π_S[x, yup, 0] - Π_S[x, ydn, 0])  # - dy Πxx
        )


# ### Pressure calculation for incompressible flow

# At each timestep, we relax the pressure field $p$ toward that which will create, in combination with the other forces, a divergence-free update to ${\bf u}$.
#
# Ideally, ${\bf u}$ from the last timestep was exactly divergence-free. But, accounting for some numerical error, our desire for the updated ${\bf u}$ to be divergence-free:
#
# $$ \nabla \cdot {\bf u}_{t+\Delta t} = 0 $$
#
# requires that
#
# $$ \nabla \cdot \frac{\partial {\bf u}}{\partial t} = \frac{\partial}{\partial t} (\nabla \cdot {\bf u}) \approx  \frac{\nabla \cdot {\bf u}_{t+\Delta t} - \nabla \cdot {\bf u}_t}{\Delta t}  = - \frac{\nabla \cdot {\bf u}_t}{\Delta t}  $$
#
#
# Taking the divergence of the ${\bf u}$ update equation, and dropping 2nd- or higher-order derivatives of ${\bf u}$, we get another expression for $\nabla \cdot (\partial {\bf u}/\partial t)$:
#
# $$ \nabla \cdot \frac{\partial {\bf u}}{\partial t} = -\partial_i u_j \partial_j u_i +  \frac{1}{\rho} \nabla \cdot \mathbf{F} - \frac{1}{\rho} \nabla^2 p $$
#
# Note that Giomi PRX 2015 drops the first term on the RHS because of low Reynolds number.
#
# With some rearrangement, and combining our expressions for $\nabla \cdot (\partial {\bf u}/\partial t)$, this becomes a Poisson equation for $p$:
#
# $$ \nabla^2 p \approx \nabla \cdot \mathbf{F} - \rho \partial_i u_j \partial_j u_i   + \rho \frac{\nabla \cdot {\bf u}_t}{\Delta t} $$
#
# If we break up Π into a piece proportional to the identity, a trace-free diagonal piece, a symmetric piece with zeros on the diagonal,  and an antisymmetric piece, we have
# $$
# \Pi = \left(\begin{array}{c c} a +b & c + d  \\ c - d & a - b \end{array}\right)
# $$
# $$\begin{align*}
#     \nabla \cdot \mathbf{F} &= \partial_i \partial_j \Pi_{ij} \\
#     &= \partial^2_x \Pi_{xx} + \partial_x \partial_y (\Pi_{xy} + \Pi_{yx}) + \partial_y^2 \Pi_{yy}  \\
#     &= \partial^2_x (a+b) + \partial_x \partial_y (2c) + \partial^2_y(a-b)  \\
#     &= \nabla^2 a + (\partial^2_x - \partial^2_y) b + 2 \partial_x \partial_y c
# \end{align*}$$
#
# Thus, any antisymmetric component (the d term) in Π has no effect on p. The term proportional to the identity (the a term) immediately cancels from the equations of motion, as if $$\nabla^2 p'(\vec r) = f(\vec r) + \nabla^2 a(\vec r),$$ then p′(r)=p(r)+a(r) is a solution where $\nabla^2 p(r)=f(r)$. In the calculation of $\partial_t ​u$, $F'=F+\nabla a$, and $\nabla p'=\nabla p+\nabla a$. Therefore $F'−\nabla p'=F−\nabla p$. Thus only the traceless, symmetric part of the stress tensor,
# $$ \Pi_S = \left(\begin{array}{c c} b & c \\ c & -b \end{array}\right), $$
# contributes to p.
#
#
# We solve this Poisson equation on the grid by the method described in [Lorena Barba's tutorial](https://nbviewer.org/github/barbagroup/CFDPython/blob/master/lessons/13_Step_10.ipynb), relaxing $p$ in pseudo-time until its relative change falls below a stopping criterion threshold. Note that our choice of units sets $\rho=1$.
#
# Using the choice of 9-point Laplacian stencil mentioned below, we have
#
# $$
# \begin{align}
#     \nabla^2 p & \approx  \frac{1}{6}  \left(  p_{i+1,j+1} + p_{i+1,j-1} + p_{i-1,j-1} + p_{i-1,j+1}
#                 + 4 \left(
#                     p_{i+1,j} +  p_{i-1,j}  +  p_{i,j+1} +  p_{i,j-1}\right) - 20 p_{ij}
#                  \right)  \\
#        \nabla^2 p & \approx \nabla \cdot \mathbf{F} - \rho \partial_i u_j \partial_j u_i   + \rho \frac{\nabla \cdot {\bf u}_t}{\Delta t}
#     % & = \frac{1}{20} \left(p_{i+1,j+1} + p_{i+1,j-1} + p_{i-1,j-1} + p_{i-1,j+1}
#     %     + 4 \left(
#     %         p_{i+1,j} +  p_{i-1,j}  +  p_{i,j+1} +  p_{i,j-1}
#     %      \right)
#     %      - 6 \left(
#     %          \nabla \cdot \mathbf{F} - \rho \partial_i u_j \partial_j u_i - \rho \nu \nabla^2 (\nabla \cdot {\bf u})
#     %      \right)
#     %    \right)
# \end{align}
# $$
#
# where
#
# $$
# \begin{align}
#     \nabla \cdot \mathbf{F} & = (\partial^2_x - \partial^2_y) b + 2 \partial_x \partial_y c \\
#     b &= \Pi_{S,xx} \\
#     c &= \Pi_{S,xy}
# \end{align}
# $$

# In[ ]:


@nb.njit
def relax_pressure(
        u, ρ, p, Π_S, Π_A, ν, p_aux, pressure_poisson_RHS,
        dt, target_rel_change, boundary, bounds,
        max_p_iters=-1
):
    """ Find pressure field that maintains incompressibility of flow field """
    div_vector(u, pressure_poisson_RHS, bounds)  # RHS = ∇•u
    ### Laplacian(pressure_poisson_RHS, p_aux, coeff=ν)  # p_aux = ν∇²∇•u
    pressure_poisson_RHS *= ρ / dt  # RHS = (∇•u)/∆t
    ### pressure_poisson_RHS += p_aux                    # RHS = (∇•u)/∆t + ν∇²∇•u
    # RHS += -d_i u_j d_j u_i + ∇•F :
    calculate_pressure_terms(u, ρ, Π_S, pressure_poisson_RHS, bounds)
    p_iters = 0
    rel_change = target_rel_change + 1
    # end pressure relaxation if number of pseudotimesteps exceeds max
    # or end pressure relaxation if relative change per step falls below
    # threshold
    while (p_iters < max_p_iters or max_p_iters < 0) and rel_change > target_rel_change:
        # copy p_aux <- p
        p_aux[:] = p
        # the pseudotimestep update of the pressure field:
        relax_pressure_inner_loop(p, p_aux, pressure_poisson_RHS, bounds)
        apply_p_boundary_conditions(p, p_aux, boundary, u, ρ, Π_S, Π_A, ν)
        rel_change = np.sum(np.abs(p_aux - p)) / np.abs(1e-7 + np.sum(p_aux))
        p_iters += 1
    # subtract_p_avg(p,bounds)
    # print(p_iters)

    return p_iters


@nb.njit(parallel=True, fastmath=True, nogil=True)
def subtract_p_avg(p, bounds):
    p_avg = np.sum(p) / len(bounds)
    for point in nb.prange(len(bounds)): p[bounds[point, 0], bounds[point, 1]] -= p_avg


@nb.njit(parallel=True, fastmath=True, nogil=True)
def calculate_pressure_terms(u, ρ, Π_S, pressure_poisson_RHS, bounds):
    """ Calculate right-hand side of the pressure Poisson equation """
    Lx, Ly = u.shape[:2]
    for point in nb.prange(len(bounds)):
        x, y = bounds[point, 0], bounds[point, 1]
        xup = (x + 1) % Lx
        xdn = (x - 1)
        yup = (y + 1) % Ly
        ydn = (y - 1)
        dudx = 0.5 * (u[xup, y, 0] - u[xdn, y, 0])
        dvdy = 0.5 * (u[x, yup, 1] - u[x, ydn, 1])
        # d_i u_j d_j u_i  = (d_x u_x)^2 + (d_y u_y)^2 + 2 d_y u_x d_x u_y:
        pressure_poisson_RHS[x, y] += (
            # ∇•F = ∂ᵢ∂ⱼΠᵢⱼ = (∂x^2 - ∂y^2) Π_{xx} + 2 ∂x ∂y Π_{xy}
                (Π_S[xup, y, 0] + Π_S[xdn, y, 0] - Π_S[x, yup, 0] - Π_S[x, ydn, 0])
                + 0.5 * (
                        Π_S[xup, yup, 1] - Π_S[xup, ydn, 1]
                        - Π_S[xdn, yup, 1] + Π_S[xdn, ydn, 1]
                ) - ρ * (dudx * dudx + dvdy * dvdy + 0.5 * (u[x, yup, 0] - u[x, ydn, 0]) * (
                    u[xup, y, 1] - u[xdn, y, 1]))
        )


@nb.njit(parallel=True, fastmath=True, nogil=True)
def relax_pressure_inner_loop(p, p_aux, pressure_poisson_RHS, bounds):
    """
    Evolve pressure field toward satisfying the pressure Poisson equation
    """
    Lx, Ly = p.shape[:2]
    for point in nb.prange(len(bounds)):
        x, y = bounds[point, 0], bounds[point, 1]
        xup = (x + 1) % Lx
        xdn = (x - 1)
        yup = (y + 1) % Ly
        ydn = (y - 1)
        # coefficients depend on choice of Laplacian stencil,
        # solved for the central term
        p[x, y] = 0.05 * (
                -6 * pressure_poisson_RHS[x, y]
                + 4 * (
                        p_aux[xup, y] + p_aux[x, yup]
                        + p_aux[x, ydn] + p_aux[xdn, y]
                )
                + p_aux[xup, yup] + p_aux[xup, ydn]
                + p_aux[xdn, yup] + p_aux[xdn, ydn]
        )


# ### Molecular field and advection tensors

# Here we calculate two tensorial terms needed for our updates: The molecular field $H_{ij}$ and the generalized advective term $S_{ij}$ (not to be confused with the scalar degree of order $S$).
#
# Molecular field:
#
# $$ H_{ij} = - \frac{\delta F_{\mathrm{LdG}}}{\partial Q_{ij}} = - Q_{ij} (A + C \mathrm{Tr}[Q^2]) + K \nabla^2 Q_{ij} $$
#
# Generalized advective term:
#
# $$ S_{ij} = \lambda S E_{ij} + Q_{ik} \omega_{kj} - \omega_{ik} Q_{kj} $$
# where $S$ is the nematic degree of order. Here we use the strain rate and vorticity tensors:
#
# $$ E_{ij} = \frac{1}{2} \left(\partial_i u_j + \partial_j u_i \right) $$
# $$ \omega_{ij} = \frac{1}{2} \left(\partial_i u_j - \partial_j u_i \right) $$
#
# Using the antisymmetry of $\omega$ and the symmetry and tracelessness of $Q$, the independent components of $S_{ij}$ are:
#
# $$ \begin{aligned}
#     S_{xx} &= \lambda S E_{xx} + Q_{xy} \omega_{yx} - \omega_{xy} Q_{yx} = \lambda S E_{xx} - 2 Q_{xy} \omega_{xy} \\
#     S_{xy} &= \lambda S E_{xy} + Q_{xx} \omega_{xy} - \omega_{xy} Q_{yy} = \lambda S E_{xy} + 2Q_{xx}\omega_{xy}
# \end{aligned} $$

# In[ ]:


@nb.njit(parallel=True, fastmath=True, nogil=True)
def H_S_from_Q(u, Q, H, S, A, C, K, λ, bounds):
    """ Calculations for molecular field and co-rotation tensors """
    Laplacian_vector(Q, H, bounds, coeff=K)  # H = K * ∇²Q
    Lx, Ly = u.shape[:2]
    for point in nb.prange(len(bounds)):
        x, y = bounds[point, 0], bounds[point, 1]
        xup = (x + 1) % Lx
        xdn = (x - 1)
        dxux = 0.5 * (u[xup, y, 0] - u[xdn, y, 0])
        dxuy = 0.5 * (u[xup, y, 1] - u[xdn, y, 1])
        dyux = 0.5 * (u[x, (y + 1) % Ly, 0] - u[x, (y - 1), 0])
        ωxy = 0.5 * (dxuy - dyux)
        trQsq = 2 * (Q[x, y, 0] ** 2 + Q[x, y, 1] ** 2)
        λS = λ * np.sqrt(2 * trQsq)
        H[x, y, :] -= (A + C * trQsq) * Q[x, y, :]
        TrQE = 2 * Q[x, y, 0] * dxux + Q[x, y, 1] * (dyux + dxuy)
        # using E_xx = du/dx:
        S[x, y, 0] = λS * dxux - 2 * ωxy * Q[x, y, 1] - 2 * TrQE * Q[x, y, 0]
        # using E_xy = (du/dy + dv/dx)/2:
        S[x, y, 1] = λS * 0.5 * (dxuy + dyux) + 2 * ωxy * Q[x, y, 0] - 2 * TrQE * Q[x, y, 1]

    # ### Stress tensor


# The stress tensor is the sum of the elastic stress $\Pi^E$ and active stress $\Pi^A$:
# $$ \Pi = \Pi^E + \Pi^A $$
# (The contributions of viscosity and pressure are added separately)
#
# Elastic stress term:
# $$ \Pi^E_{ij} = - \lambda H_{ij} + Q_{ik} H_{kj} - H_{ik} Q_{kj}  +2\mathrm{Tr}[QH]Q_{ij}- K \partial_i Q_{kl} \partial_j Q_{kl}$$
#
# The Ericksen Stress $-K \partial_i Q_{kl} \partial_j Q_{kl}$ is not traceless, but is symmetric, hence, we can decompose it in the following way
#
# $$\Pi^{Ericksen} =  \frac{\mathrm{Tr}[\Pi^{Ericksen}]}{2} \mathbb{I} + \begin{pmatrix}
# b & c \\
# c & -b
# \end{pmatrix},$$
#
# where $$b = \frac{1}{2}[\Pi^{Ericksen}_{11} - \Pi^{Ericksen}_{22}] \\= \frac{-K}{2}[\partial_x Q_{kl} \partial_x Q_{kl} - \partial_y Q_{kl} \partial_y Q_{kl}] \\=  -K[(\partial_x Q_{xx})^2 + (\partial_x Q_{xy})^2 - (\partial_y Q_{xx})^2 - (\partial_y Q_{xy})^2]$$
# and
# $$c = \partial_x Q_{kl} \partial_y Q_{kl} = -2K [\partial_x Q_{xx}\partial_y Q_{xx} + \partial_x Q_{xy}\partial_y Q_{xy}].$$
#
# Using the symmetry and tracelessness of both $Q$ and $H$, the independent components are:
#
# $$ \begin{aligned}
#      \Pi^E_{xx} & = - \lambda H_{xx} + Q_{xx} H_{xx} + Q_{xy} H_{yx} - H_{xx} Q_{xx} - H_{xy} Q_{yx} -K[(\partial_x Q_{xx})^2 + (\partial_x Q_{xy})^2 - (\partial_y Q_{xx})^2 - (\partial_y Q_{xy})^2]\\
#      &= - \lambda H_{xx} +2\mathrm{Tr}[QH]Q_{xx} -K[(\partial_x Q_{xx})^2 + (\partial_x Q_{xy})^2 - (\partial_y Q_{xx})^2 - (\partial_y Q_{xy})^2] \\
#      \Pi^E_{xy} &= - \lambda H_{xy} + Q_{xx} H_{xy} + Q_{xy} H_{yy} - H_{xx} Q_{xy} - H_{xy} Q_{yy} +2\mathrm{Tr}[QH]Q_{xy} -2K [\partial_x Q_{xx}\partial_y Q_{xx} + \partial_x Q_{xy}\partial_y Q_{xy}]\\
#      &= -\lambda H_{xy} + 2 (Q_{xx} H_{xy} - H_{xx} Q_{xy}) -2K [\partial_x Q_{xx}\partial_y Q_{xx} + \partial_x Q_{xy}\partial_y Q_{xy}]
#  \end{aligned} $$
#
# Active stress term:
# $$ \Pi^A_{ij} = - \zeta Q_{ij} $$
#
# We can rewrite $\Pi$ as a sum of a symmetric, traceless component, an antisymmetric component, and a component proportional to the identity:
# $$ \begin{aligned}
#      \Pi &= \Pi_S + \Pi_A +\Pi_I\\
#      \Pi_S &= - \lambda H_{ij} + (2\mathrm{Tr}[QH] - \zeta) Q_{ij} + \Pi^{Ericksen} -  \frac{\mathrm{Tr}[\Pi^{Ericksen}]}{2} \mathbb{I}\\
#      \Pi_A &= Q_{ik} H_{kj} - H_{ik} Q_{kj}\\
#      \Pi_I &= \frac{\mathrm{Tr}[\Pi^{Ericksen}]}{2} \mathbb{I}
# \end{aligned} $$
#
# $\Pi_S$ has two independent components, $\Pi_{xx}$ and $\Pi_{xy}$, while $\Pi_A$ has one independent component:
# $$ \begin{aligned}
#      \Pi_{A,xy} &= Q_{xx} H_{xy} - H_{xx} Q_{xy} + Q_{xy} H_{yy} - H_{xy} Q_{yy} \\
#      &= Q_{xx} H_{xy} - H_{xx} Q_{xy} - Q_{xy} H_{xx} + H_{xy} Q_{xx} \\
#      &= 2(Q_{xx} H_{xy} - H_{xx} Q_{xy})
# \end{aligned} $$

# In[ ]:


@nb.njit(parallel=True, fastmath=True, nogil=True)
def get_Erickson_stress(Q, K, Π_S, bounds):
    Lx, Ly = Q.shape[:2]
    for point in nb.prange(len(bounds)):
        x, y = bounds[point, 0], bounds[point, 1]
        xup = (x + 1) % Lx
        xdn = (x - 1)
        yup = (y + 1) % Ly
        ydn = (y - 1)
        dxQxx, dxQxy = 0.5 * (Q[xup, y] - Q[xdn, y])
        dyQxx, dyQxy = 0.5 * (Q[x, yup] - Q[x, ydn])
        Π_S[x, y, 0] -= K * ((dxQxx) ** 2 + (dxQxy) ** 2 - (dyQxx) ** 2 - (dyQxy) ** 2)
        Π_S[x, y, 1] -= 2 * K * ((dxQxy * dyQxy) + (dxQxx * dyQxx))


@nb.njit(parallel=True, fastmath=True, nogil=True)
def get_TrQH_term(Q, H, Π_S):
    TrQH = 2 * (Q[:, :, 0] * H[:, :, 0] + Q[:, :, 1] * H[:, :, 1])
    Π_S[:, :, 0] += TrQH * Q[:, :, 0]
    Π_S[:, :, 1] += TrQH * Q[:, :, 1]


@nb.njit(parallel=True, fastmath=True, nogil=True)
def calculate_Π(Π_S, Π_A, H, Q, λ, ζ, K, bounds):
    """ Calculation of stress tensor (elastic + active contributions) """
    # symmetric traceless component (two elements)
    Π_S[:] = -λ * H - ζ * Q
    get_Erickson_stress(Q, K, Π_S, bounds)
    get_TrQH_term(Q, H, Π_S)

    # antisymmetric component (one element)
    Π_A[:] = 2 * (Q[:, :, 0] * H[:, :, 1] - H[:, :, 0] * Q[:, :, 1])


# ### Boundary conditions

# - We define a boundary as a set $\{\hat \nu\}$ comprising of the normal vectors pointing out of the enclosed liquid crystal at every point. We can find $\{\hat \nu\}$ by first finding the normalized tangent vectors to the boundary at every point, and then rotating them by $\pi / 2$. if we have a surface parameterized by $\vec x(u)$, then by definition, the tangent vectors to the surface are $\partial_u \vec x(u)$. In two dimensions, rotation of these tangent vectors can be seen as $$(\partial_u x(u), \partial_u y(u)) \to (\partial_u y(u), -\partial_u x(u))$$ Normalizing by $\sqrt{(\partial_u x(u))^2 + (\partial_u y(u))^2}$ we therefore define a boundary set as
#  $$\hat \nu  =  \frac{1}{\sqrt{(\partial_u x(u))^2 + (\partial_u y(u))^2}}(\partial_u y(u) \hat i - \partial_u x(u) \hat j) $$
# - Partial derivatives at the boundary are given in first and second order as follows
#
# $$\partial_x f(x,y) = a(f(x,y) - f(x-a,y))/h$$
# $$\partial_y f(x,y) = b(f(x,y) - f(x,y-b))/h$$
# $$\partial^2_x f(x,y) = (f(x,y) - 2f(x-a,y) + f(x-2a,y))/h^2$$
# $$\partial^2_y f(x,y) = (f(x,y) - 2f(x,y-b) + f(x,y-2b))/h^2$$
#
# Where $a = sign(\nu_x)$ and $b = sign(\nu_y)$, and $$
# sign(\lambda)=
# \begin{cases}
# 1 &\text{if } \lambda > 0,\\
# -1 &\text{if } \lambda < 0,\\
# 0 &\text{if } \lambda = 0,\\
# \end{cases}
# $$
#
# - We use strong parallel anchoring on $Q_{ij}$ at the boundary. In particular at $t = 0$, we set $\vec n|_\partial = \hat \tau$. We implement this as
# $$Q_{ij} = S(n_in_j - \frac{\delta_{ij}}{2}) \to \boxed{ Q_{ij}(t = 0)|_\partial =  S(-\nu_x\nu_y - \frac{\delta_{ij}}{2})}$$
# In order to maintain this anchoring as we integrate the system forward in time, we find the coressponding molecular tensor that provides the fluid advection necessary to ensure that $$ \frac{\partial Q_{ij}}{\partial t}|_\partial = 0$$ which requires
# $$ \frac{D Q_{ij}}{Dt} = \frac{\partial Q_{ij}}{\partial t} + u_k \partial_k Q_{ij}  = \gamma^{-1} H_{ij} + S_{ij}  \to \frac{\partial Q_{ij}}{\partial t}|_\partial = 0 = - u_k \partial_k Q_{ij} + \gamma^{-1} H_{ij} + S_{ij}  \to \boxed{H_{ij}|_\partial = \gamma [u_k \partial_k Q_{ij} - S_{ij}]}$$
#
# - The flow field obeys a no flux condition $\oint {\bf u} \cdot \hat \nu = 0$, which given incompressibility, requires that $u_\nu|_\partial = 0$. We can examine the effects of this by considering a rotation $(\hat x, \hat y) \to (\hat \tau, \hat \nu)$
#  $$\partial_t \vec u_\nu = 0 = \hat \nu \cdot [-(\vec u \cdot \nabla)\vec u + \eta \nabla^2 \vec u + \vec F - \nabla p]$$
#  Yielding $$\hat \nu \cdot \nabla p|_\partial = \hat \nu \cdot [-(\vec u \cdot \nabla)\vec u + \eta \nabla^2 \vec u + \vec F]$$
#  Yet we find on closer inspection that $$-(\vec u \cdot \nabla)\vec u_\nu = -(\vec u \cdot \nabla)u_\nu = -u_\nu \partial_\nu u_\nu - u_\tau \partial_\tau u_\nu = -(0)\partial_\nu u_\nu - u_\tau (0) = 0$$
#  Applying the finite difference scheme gives a unique solution for $p|_\partial$
#
# $$\boxed{p[x,y]|_\partial = \frac{a\nu_xp[x-a,y] + b\nu_yp[x,y-b] + [\hat \nu \cdot [\eta \nabla^2 \vec u + \vec F]]}{a\nu_x + b\nu_y}}$$
#
# - For molecular fluids, the no-slipping condition is the theoretically and experimentally verified consensus on fluid-boundary interfaces. That is
# $$\boxed{{\bf u}|_\partial = 0}$$
# However, we operate, instead, at the macroscipic continuum of rods within the fluid, for which the local definition of the flow fields is not bounded similarly, we merely obtain that for an incompressible fluid in fixed area that as mandated by Stokes's Theorem $\oint {\bf u} \cdot \hat \nu = 0$, and presuming no inflows, or outflows, that
# $u_\nu |_\partial = 0$.\\\\ It is interesting, given this fact, to consider the {\it vorticity} in the frame of the local normal and tangent vectors given by $$\vec \omega = \vec \nabla \times \vec u = (\partial_\nu u_\tau - \partial_\tau u_\nu) (\hat \nu \times \hat \tau)$$
# We see that at the boundary the second term vanishes and
# $$\vec \omega|_\partial = \partial_\nu u_\tau|_\partial (\hat \nu \times \hat \tau)$$
# We can imagine a scheme in which $u_\tau$ copies the tangential component from neighbors pointing along $\hat \nu$ into the bulk. Following this idea, Lions boundary conditions propose that the slipping velocity has no directional derivative in $\hat \nu$, and thus that $\partial_\nu u_\tau|_\partial = 0 \to \vec \omega|_\partial = 0$.
# We can write this with finite differences as follows
# $$\hat \nu \cdot \vec u |_\partial= \nu_x u_x + \nu_y u_y = 0 \to u_x = -\frac{\nu_y u_y}{\nu_x}$$
# $$(\nabla \times \vec u)|_\partial = (\partial_x u_y - \partial_y u_x)|_\partial = 0 \to a(u_y(x,y) - u_y(x-a,y)) = b(u_x(x,y) - u_x(x,y-b))$$
# Note that only $\vec u(x,y)$ is assumed to be on the boundary. Combining these constraints yields
# $$a\nu_x(u_y(x,y) - u_y(x-a,y)) = -b\nu_y u_y(x,y) - b\nu_x u_x(x,y-b)) \to$$
# $$u_y(x,y) = \nu_x \frac{a u_y(x-a,y) - b u_x(x,y-b)}{a\nu_x + b\nu_y}$$
# In all, we find
# $$\boxed{\vec u|_\partial = \frac{b u_x(x,y-b) - a u_y(x-a,y)}{a\nu_x + b\nu_y} \hat \tau}$$

# In[ ]:


# Note that boundary conditions on u, p, and Q need to be applied to not only one but two nearest neighbors.
# one for H and one for Π

@nb.njit
def apply_u_boundary_conditions(u, boundary):
    Lx, Ly = u.shape[:2]
    for l in range(2):
        for x in range(Lx):
            for y in range(Ly):
                # u[x,y,:] = 0
                if boundary[l, x, y, 0] or boundary[l, x, y, 1]:
                    nx, ny = boundary[l, x, y]
                    a = 0 if nx == 0 else round(nx / abs(nx))
                    b = 0 if ny == 0 else round(ny / abs(ny))
                    u[x, y] = np.array([ny, -nx]) * (b * u[x, y - b, 0] - a * u[x - a, y, 1]) / (a * nx + b * ny)
                    u[x, y, :] = 0


@nb.njit
def apply_ss_boundary_conditions(ss, boundary):
    Lx, Ly = ss.shape[:2]
    for x in range(Lx):
        for y in range(Ly):
            if boundary[0, x, y, 0] or boundary[0, x, y, 1] or boundary[1, x, y, 0] or boundary[1, x, y, 1]: ss[
                x, y] = 0


@nb.njit
def apply_p_boundary_conditions(p, p_aux, boundary, u, ρ, Π_S, Π_A, ν):
    Lx, Ly = p.shape[:2]
    for l in range(2):
        for x in range(Lx):
            for y in range(Ly):
                if boundary[l, x, y, 0] or boundary[l, x, y, 1]:
                    nx, ny = boundary[l, x, y]
                    a = 0 if nx == 0 else round(nx / abs(nx))
                    b = 0 if ny == 0 else round(ny / abs(ny))
                    Fx = a * (Π_S[x, y, 0] - Π_S[x - a, y, 0]) + b * (
                                Π_S[x, y, 1] + Π_A[x, y] - Π_S[x, y - b, 1] - Π_A[x, y - b])
                    Fy = a * (Π_S[x, y, 1] - Π_A[x, y] - Π_S[x - a, y, 1] + Π_A[x - a, y]) - b * (
                                Π_S[x, y, 0] - Π_S[x, y - b, 0])
                    F = np.array([Fx, Fy])
                    lapu = 2 * u[x, y] - 2 * (u[x - a, y] + u[x, y - b]) + u[x - 2 * a, y] + u[x, y - 2 * b]
                    p[x, y] = (np.dot(boundary[l, x, y], F + ρ * ν * lapu) + a * nx * p_aux[x - a, y] + b * ny * p_aux[
                        x, y - b]) / (a * nx + b * ny)


@nb.njit
def apply_Q_boundary_conditions(Q, boundary):
    Lx, Ly = Q.shape[:2]
    for l in range(2):
        for x in range(Lx):
            for y in range(Ly):
                if boundary[l, x, y, 0] or boundary[l, x, y, 1]:
                    nx, ny = boundary[l, x, y, 0], boundary[l, x, y, 1]
                    theta = np.arccos(nx)
                    if ny < 0: theta = 2 * np.pi - theta
                    net_charge = 2 / 2  # change for artificial winding index
                    nnx, nny = np.cos(theta * net_charge), np.sin(theta * net_charge)
                    Q[x, y, 0] = S0 * (nny ** 2 - 1 / 2)  # Set director to (nx, -ny)
                    Q[x, y, 1] = S0 * (-nnx * nny)


@nb.njit
def apply_H_boundary_conditions(H, γ, Q, u, S, boundary):
    Lx, Ly = H.shape[:2]
    for l in range(2):
        for x in range(Lx):
            for y in range(Ly):
                if boundary[0, x, y, 0] or boundary[0, x, y, 1] or boundary[1, x, y, 0] or boundary[1, x, y, 1]:
                    nx, ny = boundary[l, x, y]
                    a = 0 if nx == 0 else round(nx / abs(nx))
                    b = 0 if ny == 0 else round(ny / abs(ny))
                    H[x, y, :] = γ * (
                                a * u[x, y, 0] * (Q[x, y] - Q[x - a, y]) + b * u[x, y, 1] * (Q[x, y] - Q[x, y - b]) - S[
                                                                                                                      x,
                                                                                                                      y,
                                                                                                                      :])

                # ## Numerical implementation


# ### Main simulation loop

# In[ ]:


def run_active_nematic_sim(u, Q, p, boundary, bounds, consts_dict, runlabel='test'):
    consts = tuple(consts_dict.values())

    run_results_path = make_directory_if_needed(resultspath + runlabel + "/")
    Q_results_path = make_directory_if_needed(run_results_path + "Q/")
    u_results_path = make_directory_if_needed(run_results_path + "u/")
    run_img_path = make_directory_if_needed(imgpath + runlabel + "/")

    # save parameters to text file
    Lx, Ly = u.shape[:2]
    save_params = dict(Lx=Lx, Ly=Ly)
    save_params.update(consts_dict)
    with open(run_results_path + runlabel + "_consts.txt", "w") as f:
        f.write(str(save_params))

    arrs = declare_auxiliary_arrays(u, Q, p)
    # div_u = np.zeros((Lx, Ly))  # holds divergence of velocity field

    # other initializations
    udiff = udiff_thresh + 1
    stepcount = 0
    t = 0
    fig, plots_dict = create_plot(u, Q, p, bounds)

    stepcount_last_save = -save_every_n_steps

    apply_Q_boundary_conditions(Q, boundary)
    apply_u_boundary_conditions(u, boundary)
    apply_p_boundary_conditions(p, p.copy(), boundary, u, consts[-3], arrs[-2], arrs[-1], consts[1])

    while udiff > udiff_thresh and stepcount < max_steps and t < max_t:
        if stepcount - stepcount_last_save >= save_every_n_steps:
            for (label, obj, path) in [
                ("Q", Q, Q_results_path),
                ("u", u, u_results_path)
            ]:
                obj_as_table = np.stack([
                    obj[:, :, i].flatten()
                    for i in range(2)
                ], axis=-1)
                np.savetxt(
                    path + label + f'_{stepcount:10d}.txt'.replace(' ', '0'),
                    obj_as_table
                )
            save_plot = True
            stepcount_last_save = stepcount
        else:
            save_plot = False
            stepcount, t, udiff = update_step(
                arrs, consts, stepcount, t, boundary, bounds,
                # do one step in first loop to catch errors and get printout
                n_steps=jit_loops if t > 0 else 1
            )
            # div_vector(u, div_u)
        update_plot(
            fig, plots_dict, u, Q, p, boundary, bounds, stepcount,
            do_save_plot=save_plot, savefileprefix=run_img_path
        )
        # print(
        #     f'\rsteps: {stepcount},  time: {t:.5f},  '
        #     f'|Δu|/|u|: {udiff:.5E},  <p iters>: {avg_p_iters},  '
        #     f'<|div(u)|/|u|> = {np.average(np.abs(div_u))/np.average(np.sqrt(np.sum(u**2, axis=-1))):.5E}'
        # )


# ### Euler update

# At each timestep, we do an Euler update:
#
# * Calculate the strain and vorticity tensors from ${\bf u}$.
# * Calculate $H_{ij}$, $S_{ij}$, and $\Pi_{ij}$ from $Q_{ij}$.
# * Calculate the pressure field $p$ needed to ensure a divergence-free update to ${\bf u}$.
# * As mentioned above, the antisymmetric component of $\Pi$ does not contribute to $p$.
# * Calculate $\partial_t Q_{ij}$ and $\partial_t {\bf u}$ as given above.
# * Update:
#     $$ Q_{ij} \leftarrow Q_{ij} + (\Delta t) \partial_t Q_{ij} $$
#     $$ {\bf u}  \leftarrow {\bf u} + (\Delta t) \partial_t {\bf u}$$

# In[ ]:


@nb.njit
def update_step(arrs, consts, stepcount, t, boundary, bounds, n_steps=1):
    """Loop over Euler updates to velocity u, pressure p, and the Q-tensor"""
    p_iters = 0
    for _ in range(n_steps):
        p_iters += update_step_inner(arrs, consts, bounds, boundary)  # Euler update
        stepcount += 1
        t += consts[0]
    u, dudt = arrs[:2]
    du_tot_sum = np.sqrt(np.sum(dudt ** 2))
    udiff_denom = np.sqrt(np.sum(u ** 2))
    if udiff_denom != 0:
        udiff = du_tot_sum / udiff_denom
    else:
        udiff = 0.
    return stepcount, t, udiff


@nb.njit
def update_step_inner(arrs, consts, bounds, boundary):
    """ Euler update """
    dt, ν, A, C, K, λ, ζ, γ, ρ, p_target_rel_change, max_p_iters = consts
    u, dudt, Q, dQdt, p, p_aux, pressure_poisson_RHS, S, H, Π_S, Π_A = arrs

    H_S_from_Q(u, Q, H, S, A, C, K, λ, bounds)  # update H and S
    apply_H_boundary_conditions(H, γ, Q, u, S, boundary)

    calculate_Π(Π_S, Π_A, H, Q, λ, ζ, K, bounds)  # update Π

    # relax pressure to ensure incompressibility
    p_iters = relax_pressure(
        u, ρ, p, Π_S, Π_A, ν, p_aux, pressure_poisson_RHS,
        dt, p_target_rel_change, boundary, bounds, max_p_iters=max_p_iters
    )

    # calculate dQdt
    get_Q_update(dQdt, Q, H, S, u, γ, bounds)

    # calculate dudt
    get_u_update(dudt, u, p, ρ, Π_S, Π_A, ν, bounds)

    # update Q, u
    Q += dt * dQdt
    u += dt * dudt
    apply_Q_boundary_conditions(Q, boundary)  # should already be obeyed via H boundary but good for numeric precision
    apply_u_boundary_conditions(u, boundary)

    return p_iters


# ### Some utilities for finite difference calculus

# #### Laplacian

# Here we implement the Laplacian using the following 9-point [stencil](https://en.wikipedia.org/wiki/Discrete_Laplace_operator):
#
# $$ \nabla^2 f \approx \frac{1}{6} \begin{array}{|c | c | c|} \hline 1 & 4 & 1 \\ \hline 4 & -20 & 4 \\ \hline 1 & 4 & 1 \\ \hline \end{array} $$
#
# for both scalars and vectors. There's also a function for calculating the divergence of a vector field.

# In[ ]:


@nb.njit(parallel=True, fastmath=True, nogil=True)
def Laplacian(arr, out, bounds, coeff=1.):
    """ Laplacian of a scalar array """
    coeff /= 6
    Lx, Ly = arr.shape[:2]
    for point in nb.prange(len(bounds)):
        x, y = bounds[point, 0], bounds[point, 1]
        xup = (x + 1) % Lx
        xdn = (x - 1)
        yup = (y + 1) % Ly
        ydn = (y - 1)
        out[x, y] = coeff * (
                -20 * arr[x, y]
                + 4 * (
                        arr[xup, y]
                        + arr[x, ydn] + arr[x, yup]
                        + arr[xdn, y]
                )
                + arr[xup, ydn] + arr[xup, yup]
                + arr[xdn, ydn] + arr[xdn, yup]
        )


@nb.njit(parallel=True, fastmath=True, nogil=True)
def Laplacian_vector(arr, out, bounds, coeff=1.):
    """ Laplacian of an array of vectors """
    coeff /= 6
    Lx, Ly = arr.shape[:2]
    for point in nb.prange(len(bounds)):
        x, y = bounds[point, 0], bounds[point, 1]
        xup = (x + 1) % Lx
        xdn = (x - 1)
        yup = (y + 1) % Ly
        ydn = (y - 1)
        out[x, y, :] = coeff * (
                -20 * arr[x, y, :]
                + 4 * (
                        arr[xup, y, :]
                        + arr[x, ydn, :] + arr[x, yup, :]
                        + arr[xdn, y, :]
                )
                + arr[xup, ydn, :] + arr[xup, yup, :]
                + arr[xdn, ydn, :] + arr[xdn, yup, :]
        )


@nb.njit(parallel=True, fastmath=True, nogil=True)
def div_vector(arr, out, bounds):
    """ calculate divergence of vector field """
    Lx, Ly = arr.shape[:2]
    for point in nb.prange(len(bounds)):
        x, y = bounds[point, 0], bounds[point, 1]
        xup = (x + 1) % Lx
        xdn = (x - 1)
        out[x, y] = 0.5 * (
                (arr[xup, y, 0] - arr[xdn, y, 0])  # dx vx
                + (arr[x, (y + 1) % Ly, 1] - arr[x, (y - 1), 1])  # dy vy
        )


# #### Upwind advective derivative

# We also calculate $-(\mathbf{u} \cdot \nabla) f$ using a second-order [upwind advective derivative](https://en.wikipedia.org/wiki/Upwind_scheme), which is an asymmetric finite differencing scheme using points from the direction opposite the local flow direction:
#
# $$ u_x \partial_x {f} \approx \begin{cases}
#      \frac{1}{2} u_{x,i,j}\left( 3 f_{i,j} - 4 f_{i-1,j} + f_{i-2,j} \right) & u_x > 0 \\
#     -\frac{1}{2} u_{x,i,j}\left( 3 f_{i,j} - 4 f_{i+1,j} + f_{i+2,j} \right) & u_x < 0
#     \end{cases}
# $$
#
# (and likewise for $u_y \partial_y f$).

# In[ ]:


@nb.njit(parallel=True, fastmath=True, nogil=True)
def upwind_advective_term(u, arr, out, bounds, coeff=-1):
    """
    calculate second-order upwind advective derivative -(u•∇)[arr]
    and add this to 'out'
    """
    coeff *= 0.5
    Lx, Ly = arr.shape[:2]
    for point in nb.prange(len(bounds)):
        x, y = bounds[point, 0], bounds[point, 1]
        xup = (x + 1) % Lx
        xdn = (x - 1)
        xupup = (x + 2) % Lx
        xdndn = (x - 2)
        # -ux dx A_k
        tmp = coeff * u[x, y, 0]
        if u[x, y, 0] > 0:
            out[x, y, :] += tmp * (
                    3 * arr[x, y, :] - 4 * arr[xdn, y, :] + arr[xdndn, y, :]
            )

        else:
            out[x, y, :] += tmp * (
                    -3 * arr[x, y, :] + 4 * arr[xup, y, :] - arr[xupup, y, :]
            )

        # -uy dy A_k
        tmp = coeff * u[x, y, 1]
        if u[x, y, 1] > 0:
            ydn = (y - 1)
            ydndn = (y - 2)
            out[x, y, :] += tmp * (
                    3 * arr[x, y, :] - 4 * arr[x, ydn, :] + arr[x, ydndn, :]
            )
        else:
            yup = (y + 1) % Ly
            yupup = (y + 2) % Ly
            out[x, y, :] += tmp * (
                    -3 * arr[x, y, :] + 4 * arr[x, yup, :] - arr[x, yupup, :]
            )


# ## Plotting

# We output:
# * a quiver plot for the director field, overlaid with...
#     * disclinations, colored red $+1/2$ or blue $-1/2$ as determined by the saddle-splay energy density
#     * and a heatmap of nematic order $S$
# * a quiver plot for the velocity field, colored by vorticity
# * a heatmap for the pressure
#

# In[ ]:


def ss_plot_function(ss, vmax):
    """ Threshold on abs. val. of saddle-splay to show defects """
    return (np.ma.masked_where(np.abs(ss) < vmax, ss)).T


def n_do_from_Q(Q):
    """Extract nematic director and degree of order from Q-tensor"""
    do = np.sqrt(2 * np.sum(Q * Q, axis=2))  # nematic degree of order (d.o.); argument of sqrt is trQsq
    a = Q[:, :, 0]  # Qxx values
    b = Q[:, :, 1]  # Qxy values
    denom = np.sqrt(b * b + (a + np.sqrt(a * a + b * b)) ** 2)  # normalization for n
    bad_ptsX, bad_ptsY = np.where(denom == 0)  # trick to avoid divisions by zero
    denom[bad_ptsX, bad_ptsY] = 1
    nx = (a + np.sqrt(a * a + b * b)) / denom  # n_x values
    ny = b / denom  # n_y values
    nx[bad_ptsX, bad_ptsY] = 0
    ny[bad_ptsX, bad_ptsY] = 0
    return nx, ny, do


def create_plot(u, Q, p, bounds):
    Lx, Ly = u.shape[:2]
    X, Y = np.meshgrid(np.arange(Lx), np.arange(Ly))
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(5 * figheight * Lx / Ly, figheight))
    E = np.zeros((Lx, Ly, 2))
    ω = np.zeros((Lx, Ly))
    t_ω = np.zeros((Lx, Ly))
    r_ω = np.zeros((Lx, Ly))
    mask = np.zeros((Lx, Ly))
    for point in nb.prange(len(bounds)): mask[bounds[point, 0], bounds[point, 1]] = 1
    EΩ_from_u(u, E, ω, bounds)
    extent = [0, (Lx - 1), (Ly - 1), 0]
    plots_dict = {
        "X": X,
        "Y": Y,
        "ax1": ax1,
        "ax2": ax2,
        "ax3": ax3,
        "strain_tensor": E,
        "vorticity": ω,
        "avg_vorticity": t_ω,
        "rms_vorticity": r_ω,
    }

    fig.tight_layout(pad=5.0)
    ax1.set_box_aspect(Ly / Lx)
    ax2.set_box_aspect(Ly / Lx)
    ax3.set_box_aspect(Ly / Lx)
    ax4.set_box_aspect(Ly / Lx)
    ax5.set_box_aspect(Ly / Lx)
    ax1.set_title('Nematic Director Field')
    ax2.set_title('Velocity Field')
    ax3.set_title('Pressure Field')
    ax4.set_title('Signed Average Vorticity')
    ax5.set_title('RMS Vorticity')

    # quiver plot for velocity field
    ωscale = 0.5 * uscale / Lx
    plots_dict['ures'] = ures = max(1, int(min(Lx, Ly) / 20))  # how many velocity arrows
    plots_dict['u_quiv'] = u_quiv = ax2.quiver(
        X.T[::ures, ::ures], Y.T[::ures, ::ures],
        u[::ures, ::ures, 0], u[::ures, ::ures, 1],
        ω[::ures, ::ures],  # color velocity arrows by vorticity
        scale=uscale,
        angles='xy',
        cmap='jet',
        clim=[-ωscale, ωscale],
        alpha=mask[::ures, ::ures]
    )
    ax2.set_facecolor('white')
    ax2.set_xlim(*tuple(extent[:2]))
    ax2.set_ylim(*tuple(extent[2:]))

    plt.colorbar(
        u_quiv,
        ax=ax2,
        fraction=0.046,
        pad=0.04,
        label='ω',
        location='right',
        extend='both'
    )

    # quiver plot for average velocity field
    plots_dict['ua_quiv'] = ua_quiv = ax4.imshow(plots_dict['avg_vorticity'], cmap='jet', vmin=-2 * ωscale,
                                                 vmax=2 * ωscale, alpha=mask.T)
    ax4.set_facecolor('white')
    ax4.set_xlim(*tuple(extent[:2]))
    ax4.set_ylim(*tuple(extent[2:]))

    plt.colorbar(
        ua_quiv,
        ax=ax4,
        label=r'$\langle \omega \rangle$',
        location='right',
        extend='both',
    )

    # quiver plot for rms velocity field
    plots_dict['ur_quiv'] = ur_quiv = ax5.imshow(plots_dict['rms_vorticity'], cmap='jet', vmin=0, vmax=2 * ωscale,
                                                 alpha=mask.T)
    ax5.set_facecolor('white')
    ax5.set_xlim(*tuple(extent[:2]))
    ax5.set_ylim(*tuple(extent[2:]))

    plt.colorbar(
        ur_quiv,
        ax=ax5,
        label=r'$\sqrt {\langle \omega^2 \rangle }$',
        location='right',
        extend='both',
    )

    # quiver plot for director field
    plots_dict['nres'] = nres = max(1, int(Lx / n_rods))  # how many director lines
    nx, ny, do = n_do_from_Q(Q)  # get nematic director and degree of order
    plots_dict['n_quiv'] = ax1.quiver(
        X.T[::nres, ::nres], Y.T[::nres, ::nres],
        nx[::nres, ::nres], ny[::nres, ::nres],
        linewidth=.2,
        headwidth=0,
        scale=n_quiver_scale,
        pivot='middle',
        angles='xy',
        alpha=mask[::nres, ::nres]
    )
    ax1.set_facecolor('white')

    # degree of order plot
    plots_dict['do_plot'] = do_plot = ax1.imshow(
        do.T / S0,
        vmin=0,
        vmax=1,
        cmap='Greens_r',
        interpolation='none',
        extent=extent,
        alpha=mask.T
    )
    plt.colorbar(
        do_plot,
        ax=ax1,
        fraction=0.046,
        pad=0.04,
        label=r'$S / S_0$',
        location='right',
        extend='max'
    )

    # saddle-splay markers for defects
    Lx, Ly = Q.shape[:2]
    plots_dict['ss'] = ss = np.zeros((Lx, Ly))
    get_saddle_splay(Q, plots_dict['ss'], bounds)
    vmax = ss_scale
    plots_dict['ss_plot'] = ax1.imshow(
        ss_plot_function(ss, vmax),
        vmin=-vmax,
        vmax=vmax,
        cmap='bwr_r',
        interpolation='none',
        extent=extent,
        alpha=mask.T
    )

    # pressure plot
    plots_dict['p_plot'] = p_plot = ax3.imshow(
        p.T,
        cmap='inferno',
        interpolation='none',
        extent=extent,
        alpha=mask.T
        # vmin = -2*uscale,
        # vmax = 2*uscale
    )
    ax3.set_facecolor('white')
    plots_dict['p_cbar'] = plt.colorbar(
        p_plot,
        ax=ax3,
        fraction=0.046,
        pad=0.04,
        label=r'$p$',
        location='right',
        extend='max'
    )

    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    ax4.invert_yaxis()
    ax5.invert_yaxis()

    ax1.set_frame_on(False)
    ax2.set_frame_on(False)
    ax3.set_frame_on(False)
    ax4.set_frame_on(False)
    ax5.set_frame_on(False)

    try:
        display(fig)
    except NameError:
        pass
    return fig, plots_dict


def update_plot(fig, plots_dict, u, Q, p, boundary, bounds, stepcount, do_save_plot=True, savefileprefix=''):
    nres = plots_dict['nres']
    ures = plots_dict['ures']
    nx, ny, deg_ord = n_do_from_Q(Q)  # get nematic director and degree of order
    EΩ_from_u(u, plots_dict['strain_tensor'], plots_dict['vorticity'], bounds)
    plots_dict['avg_vorticity'] += plots_dict['vorticity']
    plots_dict['rms_vorticity'] += plots_dict['vorticity'] ** 2

    plots_dict['n_quiv'].set_UVC(nx[::nres, ::nres], ny[::nres, ::nres])  # update director plot
    # update velocity plot
    plots_dict['u_quiv'].set_UVC(
        u[::ures, ::ures, 0], u[::ures, ::ures, 1],
        plots_dict['vorticity'][::ures, ::ures]
    )
    plots_dict['ua_quiv'].set_data(plots_dict['avg_vorticity'].T / (stepcount / 100))
    plots_dict['ur_quiv'].set_data((plots_dict['rms_vorticity'].T / (stepcount / 100)) ** 0.5)
    get_saddle_splay(Q, plots_dict['ss'], bounds)
    apply_ss_boundary_conditions(plots_dict['ss'], boundary)
    vmax = plots_dict['ss_plot'].get_clim()[1]
    plots_dict['ss_plot'].set_data(
        ss_plot_function(plots_dict['ss'], vmax)
    )
    plots_dict['do_plot'].set_data(deg_ord.T / S0)
    plots_dict['p_plot'].set_data(p.T)
    plt.close()
    try:
        # this line is specific to the IPython environment
        display(fig, clear=True)
    except NameError:
        pass

    if do_save_plot:
        fig.savefig(savefileprefix + f'{stepcount:10d}.png'.replace(' ', '0'))


@nb.njit(parallel=True, fastmath=True, nogil=True)
def get_saddle_splay(Q, out, bounds):
    """ calculate saddle-splay energy density (for plotting defects) """
    Lx, Ly = Q.shape[:2]
    for point in nb.prange(len(bounds)):
        x, y = bounds[point, 0], bounds[point, 1]
        xup = (x + 1) % Lx
        xdn = (x - 1)
        yup = (y + 1) % Ly
        ydn = (y - 1)
        twice_dxQxx = Q[xup, y, 0] - Q[xdn, y, 0]
        twice_dxQxy = Q[xup, y, 1] - Q[xdn, y, 1]
        twice_dyQxx = Q[x, yup, 0] - Q[x, ydn, 0]
        twice_dyQxy = Q[x, yup, 1] - Q[x, ydn, 1]
        out[x, y] = twice_dxQxy * twice_dyQxx - twice_dxQxx * twice_dyQxy
        ## (diQjk dkQij - diQij dkQjk)/2 -> 2 dxQxy dy Qxx - 2 dxQxx dyQxy


@nb.njit(parallel=True, fastmath=True, nogil=True)
def EΩ_from_u(u, E, ω, bounds):
    """
    Compute the strain rate tensor E and vorticity ω from the flow field u
    (only used for plots)
    """
    Lx, Ly = u.shape[:2]
    for point in nb.prange(len(bounds)):
        x, y = bounds[point, 0], bounds[point, 1]
        xup = (x + 1) % Lx
        xdn = (x - 1)
        yup = (y + 1) % Ly
        ydn = (y - 1)
        dxuy = u[xup, y, 1] - u[xdn, y, 1]  # delay multiplying by 1/2
        dyux = u[x, yup, 0] - u[x, ydn, 0]  # delay multiplying by 1/2
        E[x, y, 0] = 0.5 * (u[xup, y, 0] - u[xdn, y, 0])  # du_x/dx
        E[x, y, 1] = 0.25 * (dxuy + dyux)
        ω[x, y] = 0.25 * (dxuy - dyux)

    # ### Initialization auxiliary functions


# In[ ]:


def initialize_Q_from_θ(Q, theta_initial, S0):
    nx_initial = np.cos(theta_initial)
    ny_initial = np.sin(theta_initial)
    Q[:, :, 0] = S0 * (nx_initial ** 2 - 0.5)
    Q[:, :, 1] = S0 * (nx_initial * ny_initial)
    # Q *= 0.01


def set_boundary(boundary, sim_points, bc_label, Lx, Ly):
    outer_bound = set()

    if bc_label == 'circular':
        radius = Lx // 2 - 1

        # marks in bound points
        for x in range(Lx):
            for y in range(Ly):
                if (x - radius) ** 2 + (y - radius) ** 2 <= radius ** 2: sim_points.add((x, y))
        # in bound points that border out of bound points are outer boundary
        for x, y in sim_points:
            if {(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)} - sim_points:
                boundary[1, x, y, 0] = round((x - radius) / ((x - radius) ** 2 + (y - radius) ** 2) ** (0.5), 4)
                boundary[1, x, y, 1] = round((y - radius) / ((x - radius) ** 2 + (y - radius) ** 2) ** (0.5), 4)
                outer_bound.add((x, y))
        # in bound - non outer boundary points points with neighbors that are in outer boundary are inner boundary
        for x, y in sim_points:
            if {(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)} & outer_bound and (x, y) not in outer_bound:
                boundary[0, x, y, 0] = round((x - radius) / ((x - radius) ** 2 + (y - radius) ** 2) ** (0.5), 4)
                boundary[0, x, y, 1] = round((y - radius) / ((x - radius) ** 2 + (y - radius) ** 2) ** (0.5), 4)

    if bc_label == 'posts':
        for x in range(Lx):
            for y in range(Ly): sim_points.add((x, y))

        post_list = [(Lx - 50, Ly - 50, 15), (Lx - 50, 50, 15), (50, Ly - 50, 15),
                     (50, 50, 15)]  # center_x, center_y, r
        for pt in post_list:
            in_r = set()
            # marks in bound points
            for x in range(Lx):
                for y in range(Ly):
                    if (x - pt[0]) ** 2 + (y - pt[1]) ** 2 <= pt[2] ** 2:
                        in_r.add((x, y))
                        sim_points.remove((x, y))
            # in bound points that border out of bound points are inner boundary, negative signs point away from bulk
            for x, y in in_r:
                if {(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)} - in_r:
                    boundary[0, x, y, 0] = -round((x - pt[0]) / ((x - pt[0]) ** 2 + (y - pt[1]) ** 2) ** (0.5), 4)
                    boundary[0, x, y, 1] = -round((y - pt[1]) / ((x - pt[0]) ** 2 + (y - pt[1]) ** 2) ** (0.5), 4)
                    outer_bound.add((x, y))
            # in bound - non outer boundary points points with neighbors that are in inner boundary are outer boundary, negative signs point away from bulk
            for x, y in in_r:
                if {(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)} & outer_bound and (x, y) not in outer_bound:
                    boundary[1, x, y, 0] = -round((x - pt[0]) / ((x - pt[0]) ** 2 + (y - pt[1]) ** 2) ** (0.5), 4)
                    boundary[1, x, y, 1] = -round((y - pt[1]) / ((x - pt[0]) ** 2 + (y - pt[1]) ** 2) ** (0.5), 4)

    if bc_label == 'cardioid':
        # marks in bound points
        a = Lx // 3
        for xs in range(Lx):
            for ys in range(Ly):
                x = xs - Lx // 4
                y = ys - Ly // 2
                if (x ** 2 + y ** 2) ** 0.5 <= a * (1 + x / (x ** 2 + y ** 2 + 1e-10) ** 0.5): sim_points.add((xs, ys))
        # in bound points that border out of bound points are outer boundary
        for x, y in sim_points:
            if {(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)} - sim_points:
                outer_bound.add((x, y))
                if x == Lx // 4 and y == Ly // 2:
                    boundary[1, x, y] = [-1, 0]
                else:
                    c = (x - Lx // 4) / ((x - Lx // 4) ** 2 + (y - Ly // 2) ** 2) ** (0.5)
                    s = (y - Ly // 2) / ((x - Lx // 4) ** 2 + (y - Ly // 2) ** 2) ** (0.5)
                    n = ((x - Lx // 4) ** 2 + (y - Ly // 2) ** 2 + (a * s) ** 2) ** (0.5)
                    boundary[1, x, y, 0] = round((x - Lx // 4 - a * s ** 2) / n, 4)
                    boundary[1, x, y, 1] = round((y - Ly // 2 + a * s * c) / n, 4)
        # in bound - non outer boundary points points with neighbors that are in outer boundary are inner boundary
        for x, y in sim_points:
            if {(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)} & outer_bound and (x, y) not in outer_bound:
                if x == Lx // 4 + 1 and y == Ly // 2:
                    boundary[0, x, y] = [-1, 0]
                else:
                    c = (x - Lx // 4) / ((x - Lx // 4) ** 2 + (y - Ly // 2) ** 2) ** (0.5)
                    s = (y - Ly // 2) / ((x - Lx // 4) ** 2 + (y - Ly // 2) ** 2) ** (0.5)
                    n = ((x - Lx // 4) ** 2 + (y - Ly // 2) ** 2 + (a * s) ** 2) ** (0.5)
                    boundary[0, x, y, 0] = round((x - Lx // 4 - a * s ** 2) / n, 4)
                    boundary[0, x, y, 1] = round((y - Ly // 2 + a * s * c) / n, 4)

    if bc_label == 'tanh':
        k = 1.0015  # between 1 an and 2.pertrusion sharpness (cusp at 1 ~ circular at 2)
        b = 1000  # between 1 and Lx. petrusion dispersion
        n = 1  # integer. number of cusps.
        radius = Lx // 2 - 1
        # marks in bound points
        for x in range(Lx):
            for y in range(Ly):
                if ((x - radius) ** 2 + (y - radius) ** 2) ** 0.5 <= radius * np.tanh(
                    b * (k - np.cos(n * np.arctan2(y - radius, x - radius)))): sim_points.add((x, y))
        # in bound points that border out of bound points are outer boundary
        for x, y in sim_points:
            if {(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)} - sim_points:
                outer_bound.add((x, y))
                t = np.arctan2(y - radius, x - radius)
                s = np.sin(t)
                c = np.cos(t)
                r = np.tanh(b * (k - np.cos(n * t)))
                dr = (1 - (np.tanh(b * (k - np.cos(n * t)))) ** 2) * np.sin(n * t) * n * b
                norm = (r ** 2 + dr ** 2) ** (0.5)
                boundary[1, x, y, 0] = round((c * r + s * dr) / norm, 4)
                boundary[1, x, y, 1] = round((s * r - c * dr) / norm, 4)
        # in bound - non outer boundary points points with neighbors that are in outer boundary are inner boundary
        for x, y in sim_points:
            if {(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)} & outer_bound and (x, y) not in outer_bound:
                t = np.arctan2(y - radius, x - radius)
                s = np.sin(t)
                c = np.cos(t)
                r = np.tanh(b * (k - np.cos(n * t)))
                dr = (1 - (np.tanh(b * (k - np.cos(n * t)))) ** 2) * np.sin(n * t) * n * b
                norm = (r ** 2 + dr ** 2) ** (0.5)
                boundary[0, x, y, 0] = round((c * r + s * dr) / norm, 4)
                boundary[0, x, y, 1] = round((s * r - c * dr) / norm, 4)

    if bc_label == 'epitrochoid':
        from scipy.optimize import fsolve
        d = .99  # between 0 and 1. 0 -> circle, 1-> epicycloid (d/r) on wikipedia definition
        k = 1  # 2(q-1) = number of cusps
        radius = Lx // 2 - 1
        for x in range(Lx):
            for y in range(Ly):
                f = lambda u: np.arctan2(y - radius, x - radius) - np.arctan2(
                    (k + 1) * np.sin(u) + d * np.sin((k + 1) * u), (k + 1) * np.cos(u) + d * np.cos((k + 1) * u))
                if (x - radius) ** 2 + (y - radius) ** 2 <= radius ** 2 / (k + 2) ** 2 * (
                        (k + 1) ** 2 + d ** 2 + 2 * (k + 1) * d * np.cos(k * fsolve(f, 0.1)[0])): sim_points.add((x, y))
                # in bound points that border out of bound points are outer boundary
        for x, y in sim_points:
            if {(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)} - sim_points:
                outer_bound.add((x, y))
                f = lambda u: np.arctan2(y - radius, x - radius) - np.arctan2(
                    (k + 1) * np.sin(u) + d * np.sin((k + 1) * u), (k + 1) * np.cos(u) + d * np.cos((k + 1) * u))
                u = fsolve(f, 0.1)[0]
                norm = (1 + d ** 2 + 2 * d * np.cos(k * u)) ** (0.5)
                boundary[1, x, y, 0] = round((np.cos(u) + d * np.cos((k + 1) * u)) / norm, 4)
                boundary[1, x, y, 1] = round((np.sin(u) + d * np.sin((k + 1) * u)) / norm, 4)
        # in bound - non outer boundary points points with neighbors that are in outer boundary are inner boundary
        for x, y in sim_points:
            if {(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)} & outer_bound and (x, y) not in outer_bound:
                f = lambda u: np.arctan2(y - radius, x - radius) - np.arctan2(
                    (k + 1) * np.sin(u) + d * np.sin((k + 1) * u), (k + 1) * np.cos(u) + d * np.cos((k + 1) * u))
                u = fsolve(f, 0.1)[0]
                norm = (1 + d ** 2 + 2 * d * np.cos(k * u)) ** (0.5)
                boundary[0, x, y, 0] = round((np.cos(u) + d * np.cos((k + 1) * u)) / norm, 4)
                boundary[0, x, y, 1] = round((np.sin(u) + d * np.sin((k + 1) * u)) / norm, 4)

    if bc_label == 'no_slip_channel':
        boundary[0, :, 1, 0] = 0
        boundary[0, :, 1, 1] = -1  # inner boundary bottom
        boundary[0, :, -2, 0] = 0
        boundary[0, :, -2, 1] = 1  # inner boundary top
        boundary[1, :, 0, 0] = 0
        boundary[1, :, 0, 1] = -1  # outer boundary bottom
        boundary[1, :, -1, 0] = 0
        boundary[1, :, -1, 1] = 1  # outer boundary top
        for x in range(Lx):
            for y in range(Ly): sim_points.add((x, y))

    if bc_label.endswith('.txt') or bc_label == 'periodic':
        for x in range(Lx):
            for y in range(Ly): sim_points.add((x, y))


def declare_auxiliary_arrays(u, Q, p):
    Lx, Ly = p.shape[:2]
    dudt = np.zeros((Lx, Ly, 2))  # velocity update
    dQdt = np.zeros((Lx, Ly, 2))  # Q update
    p_aux = np.zeros((Lx, Ly))  # auxiliary array for pressure updates
    H = np.zeros((Lx, Ly, 2))  # molecular field
    S = np.zeros((Lx, Ly, 2))  # rotational terms for Q
    Π_S = np.zeros((Lx, Ly, 2))  # stress tensor traceless-symmetric component
    Π_A = np.zeros((Lx, Ly))  # stress tensor antisymmetric component

    pressure_poisson_RHS = np.zeros((Lx, Ly))  # holds right-hand side of pressure-Poisson equation

    return (u, dudt, Q, dQdt, p, p_aux, pressure_poisson_RHS, S, H, Π_S, Π_A)


# ## Non-dimensionalization and scaling

# ### Comparison to Giomi PRX 2015

# [Giomi PRX 2015](https://journals.aps.org/prx/pdf/10.1103/PhysRevX.5.031003) scales out
# * length scale of system size $L$,
# * time scale of viscous dissipation $\tau=\rho L^2/\eta = L^2 / \nu$,
# * and stress scale of viscous stress  $\Sigma=\eta/\tau=\eta^2 / (\rho L^2) = \nu^2 \rho^2 / (\rho L^2) = \nu^2 \rho /L^2$.
#
# We will proceed similarly except using the lattice spacing $dx$ rather than $L$ as the length scale so that constants retain the same physical meaning when we change the system size. In comparing to Giomi's choices, we will thus take $L=256$ which was the size of his lattice.
#
# Setting $\tau = \Sigma  = 1$ and $L=256$, we have $\nu = 256^2$ and $\rho=1$, as well as:
#
# * $\zeta$ has units of stress, so in order for $\sqrt{K/\zeta}$ to have units of length, it follows that $K$ has units of stress x (length)^2.
#  Thus when Giomi says $K=1$, he means $K = \Sigma L^2 \rightarrow L^2$.
# * $\gamma$ has units of energy x time / (length)^2 = stress x length x time, so when Giomi says $\gamma=10$, he means $\gamma = 10 \Sigma L \tau \rightarrow 10 L $.
# * $\sqrt{K/C}$ is a length (the nematic coherence length) so $C$ has units of stress. When Giomi says $ C = 4\times 10^4$, he means $C=4\times 10^4 \Sigma \rightarrow 4\times10^4$.
#
# The defect core size (nematic coherence length) is $\sqrt{K/C} = L / \sqrt{4\times 10^4} = L / 200 = 1.28$.
#
#

# ### Reynolds number

# $$ \mathrm{Re} = \frac{\rho u L}{\eta} = \frac{u L}{\nu} $$
#
# A typical speed associated with $+1/2$ defects is $\sqrt{K \alpha} /\eta $, and a typical length is $\sqrt{K/\alpha}$, so
#
# $$ \mathrm{Re} = \frac{ \sqrt{K \alpha} \sqrt{K/\alpha}}{\rho \nu^2} = \frac{K}{\rho \nu^2} $$
#
# We can use this to express $\nu$ in terms of the Reynolds number and $K$:
#
# $$ \nu = \sqrt{\frac{K}{\rho \mathrm{Re}} } $$
#
# With the above rescalings, the Reynolds number becomes:
#
# $$ \mathrm{Re} = \frac{K}{\rho \nu^2} = \frac{K}{\Sigma L^2} \rightarrow \frac{K}{L^2} \rightarrow \frac{L^2}{L^2} = 1 $$
#
# Well, that's odd! Apparently Giomi's Reynolds number is $1$ whereas we'd expect $\ll 1$. (Especially since the paper repeatedly emphasizes its focus on "low Reynolds number turbulence".) Setting it to $0.01$ instead seems to work just fine for us.

# ### Length scales

# In my testing, `nematic_coherence_length` can be as small as 0.05. However, larger values closer to the `active_length_scale` make for more visible "tilt walls" of the kind seen in experiments (where fracture of microtubule bundles has occurred).
#
# `active_length_scale` $\ell_a$ ought to produce active flows when $\ell_a < L$ where $L$ is the linear system size. For reasons I don't understand yet, we need roughly $\ell_a \lesssim L / 20$ to see active flows.

# Define parameter ranges
ALS_range = np.arange(2.8,4.1,0.1)
NCL_range = np.arange(2.5,4.1,0.1)
resolution = 1
           
# System size
Lx = 200  # number of lattice sites along x direction
Ly = 200  # number of lattice sites along y direction

# ### Nematic properties
K = 2 ** 14  # Frank elastic constant (actually L from LdG free energy)
γ = 100  # rotational viscosity
λ = 1  # flow alignment parameter

# ### Fluid properties
Re = 1e-1  # Reynolds number, used to set viscosity
ν = np.sqrt(K / Re)

# ### Active properties
# ### Update parameters
jit_loops = 100  # number of update steps to do between outputs;
# higher -> faster; lower -> smoother animation
dt = 1e-4  # timestep

p_target_rel_change = 1e-4
# relax pressure until |Δp|/|p| per timestep < this
# max number of timestep iterations
# for pressure relaxation per timestep;
# set to <0 to ignore.
max_p_iters = -1

# ### Stopping conditions
udiff_thresh = -1  # relative change in velocity per timestep
max_steps = 1500000
max_t = 1500000
# reso scaling
Lx *= resolution
Ly *= resolution
ν *= resolution ** 2
γ /= resolution ** 2
Lx = int(Lx)
Ly = int(Ly)
ρ = 1 / resolution ** 4

for active_length_scale in [1, 2, 3, 4]:
    for nematic_coherence_length in [4, 5, 6, 7, 8]:   
        active_length_scale_p = resolution*active_length_scale     
        nematic_coherence_length_p = resolution*nematic_coherence_length

        # ### Derived parameters
        # These parameters are calculated from other ones.
        ζ = K / active_length_scale_p ** 2  # activity
        
        # LdG phase terms
        C = K / nematic_coherence_length_p ** 2
        A = -C
        S0 = np.sqrt(-2 * A / C)  # preferred bulk nematic degree of order
        
        # ### Figure options
        figheight = 6
        n_rods = min(max(Lx, Ly), 32)
        # scale for velocity arrows (higher value -> shorter arrows)
        uscale = 200
        # higher value -> SHORTER director lines
        n_quiver_scale = (2 / 3) * n_rods
        # scale for "defect density" (higher value -> smaller apparent defect core)
        ss_scale = 0.05 * S0
        
        # ### Random number generation
        seed = int(1000 * np.random.rand())
        
        # ### Options for saving data and images
        # # The simulation does $n_j=$ `jit_loops` timesteps per loop.
        # # To save every $N$th loop, set `save_every_n_steps` to $N*n_j$.
        save_every_n_steps = 10 * jit_loops
        
        # ### Package parameters as dictionary
        consts_dict = dict(
            dt=dt,
            ν=ν,
            A=A,
            C=C,
            K=K,
            λ=λ,
            ζ=ζ,
            γ=γ,
            ρ=ρ,
            p_target_rel_change=p_target_rel_change,
            max_p_iters=max_p_iters
        )
        
        # ## Declare arrays

        u = np.zeros((Lx, Ly, 2))  # velocity field
        p = np.zeros((Lx, Ly))  # pressure
        Q = np.zeros((Lx, Ly, 2))  # (Qxx,Qxy) at each site
        boundary = np.zeros((2, Lx, Ly, 2))  # boundary [Layer, X cord, Y cord, \nu_x, \nu_y]
        π = np.pi  # for convenience
        
        # ## Initial conditions (nothing written here)

        # random number generator
        rng = np.random.default_rng(seed=seed)
        
        u[:, :, :] = 0
        p[:, :] = 0
        
        # set the boundary terms
        bc_label = 'epitrochoid'
        sim_points = set()
        set_boundary(boundary, sim_points, bc_label, Lx, Ly)
        
        # director
        # randomly chosen, uniform direction
        theta_initial = π * rng.random() * np.ones((Lx, Ly))
        # add fluctuations
        theta_initial = + 1.0 * π * np.random.random((Lx, Ly))
        theta_mask = np.zeros((Lx, Ly))
        for x, y in sim_points: theta_mask[x, y] = theta_initial[x, y]
        theta_initial = theta_mask
        
        # set bounds array
        # for bounds to be compatible with numba, its size must be fixed.
        bounds = np.zeros((len(sim_points), 2), dtype=int)
        for i in range(len(bounds)): bounds[i] = sim_points.pop()
        
        #  convert to Q-tensor
        initialize_Q_from_θ(Q, theta_initial, S0)
        
        # ## Run!
        # override active length scale to make system passive:
        # consts_dict["ζ"] = 0.
        runname = f"als_{active_length_scale}_ncl_{nematic_coherence_length_p}"
        run_active_nematic_sim(u, Q, p, boundary, bounds, consts_dict, runname)
        # video
        os.system(f"ffmpeg -framerate 15 -pattern_type glob -i '{imgpath}{runname}/*.png'   -c:v libx264 -pix_fmt yuv420p {imgpath}{runname}/{runname}.mp4")
        os.system(f"rm {imgpath}{runname}/*.png")
        os.system(f"rm -r {resultspath}")
        
        
        


