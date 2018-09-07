import numpy as np
import matplotlib.pyplot as plt
from math import pi

def heat1d(dt, u_m, u_p, full_soln = False):
    """This function calculates the solution to the heat equation:
    du/dy = k d^2u/dx^2 where u has domain [-1,1]

    Parameters
    ----------
    dt : float
        timestep for problem
    u_m : np.array
        Array size [N] specifying left boundary condition
    u_p : np.array
        Array size [N] specifying right boundary condition
    full_soln : bool (optional)
        Flag indicating whether or not to return the full solution
        for each timestep

    Returns
    -------
    u_0 : np.array
        Array size [N] with values at x=0
    us : list of np.arrays
        List of full domain solutions for each timestep
    """
    # Set constants for problem
    k = 0.1
    # Size of mesh to solve using finite difference, use odd so there is a midpoint
    mesh_size = 41
    deltax = 2/mesh_size
    C = k*dt/(deltax**2)

    if u_m.size != u_p.size:
        raise ValueError('Boundary conditions must be same length')
    dt_max = deltax**2/(2*k)
    if dt > dt_max:
        raise ValueError('Forward-time center-space not stable with current'
                         'conditions. dt = '+str(dt)+' dt_max = '+str(dt_max))
    N = u_m.size
    u_0 = np.zeros(N)

    # initialize u vector
    u_vec_old = np.zeros([mesh_size,])

    #build finite difference matrix
    f_matrix = np.zeros([mesh_size,mesh_size])
    f_matrix[0,0] = 1
    f_matrix[-1,-1] = 1
    for j in range(1,mesh_size-1):
        f_matrix[j,j-1] = C
        f_matrix[j,j] = -2*C
        f_matrix[j,j+1] = C

    us = []
    u_0s = np.empty([0,])
    for i in range(N):
        u_vec_old[0] = u_m[i]
        u_vec_old[-1] = u_p[i]
        u_vec_new = np.matmul(f_matrix,u_vec_old)+u_vec_old
        # Set boundary conditions
        u_vec_new[0] = u_m[i]
        u_vec_new[-1] = u_p[i]
        #Find value at midpoint
        u_0 = u_vec_new[int((mesh_size-1)/2)]
        u_vec_old = u_vec_new
        us.append(u_vec_new)
        u_0s = np.append(u_0s,u_0)

    if not full_soln:
        return u_0s
    else:
        return us

    
# Run this script if this file is called directly
if __name__ == '__main__':
    t_total = 1
    steps = 100
    dts = np.linspace(0,t_total,steps)
    dt = t_total/steps
    u_m = np.ones(dts.size)
    u_p = -5*np.ones(dts.size)
    us = heat1d(dt, u_m, u_p, full_soln=True)
    us = heat1d(dt, u_m, u_p, full_soln=False)
    print(us)
    for i in range(steps):
        plt.plot(np.linspace(-1,1,41),us[i])
        plt.show()
        plt.clf()
        plt.cla()
        plt.close()



