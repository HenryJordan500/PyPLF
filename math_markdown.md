
Genral Particle Equation of Motion:

$\mathbf{\ddot{x}_p} = \beta \frac{D\mathbf{u}}{Dt} + \frac{1}{St}(\mathbf{u} - \mathbf{\dot{x}_p}) + \frac{1}{Fr}\mathbf{\hat{g}} + \frac{\beta}{3} (\mathbf{u} - \mathbf{\dot{x}_p}) \times \mathbf{\omega}$

$St$ - Stokes Number $(0,\infty)$. Large number is high inertia. How quickly particle responds to the flow. Formula depends on system 

$\beta$ - Dimensionless density ratio $\frac{3\rho_f}{\rho_f+2\rho_p}$. $0 < \beta < 1$, $ 1 < \beta < 3$ (Rising Particle). Same formula for all systems.

$Fr$ - Froude Number. Can be recast as $Fr = St/v_s$ where $v_s$ is the dimensionless settling speed.

$\omega = \nabla \times \mathbf{u}$. Vorticity. Zero if potential flow $u = -\nabla \phi$. Most analytical flows have no voriticity.

Note if no gravity in simulation Fr is very large. Itis best practice to remove the term from the equation. It is also best practice to remove this term for potential flow.  