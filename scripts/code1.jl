# A very simple Navier-Stokes solver for a drop falling in a rectangular
# domain. The viscosity is taken to be a constant and forward in time,
# centered in space discretization is used. The density is advected by a 
# simple upwind scheme.
using Plots
# domain size and physical variables
Lx=1.0; Ly=1.0; gx=0;gy= -100.0; rho1=1.0; rho2=2.0; m0=0.01; rro=rho1;
unorth=0; usouth=0; veast=0; vwest=0; mtime=0.0;
rad=0.15; xc=0.5; yc=0.7; # Initial drop size and location

# Numerical variables
nx=32; ny=32; dt=0.00125; nstep=100; maxit=200; maxError=0.001; beta=1.2;

# Zero various arrays
u=zeros(nx+1,ny+2); v=zeros(nx+2,ny+1); p=zeros(nx+2, ny+2);
ut=zeros(nx+1,ny+2); vt=zeros(nx+2,ny+1); tmp1=zeros(nx+2,ny+2);
uu=zeros(nx+1,ny+1); vv=zeros(nx+1,ny+1); tmp2=zeros(nx+2,ny+2);

# Set the grid
dx=Lx/nx; dy=Ly/ny;
x=[dx*(i-1.5) for i=1:nx+2]; y=[dy*(j-1.5) for j=1:ny+2];

# Set the density
r=ones(nx+2,ny+2)*rho1;
for i=2:nx+1 
    for j=2:ny+1
        if((x[i]-xc)^2 + (y[j]-yc)^2 < rad^2)
             r[i,j]=rho2
        end
    end 
end

# ======================START TIME LOOP================================
for is = 1: nstep
    # tangential velocity at boundaries
    u[1:nx+1,1] .= 2*usouth .- u[1:nx+1,2]
    u[1:nx+1,ny+2] .= 2*unorth .- u[1:nx+1,ny+1]
    v[1,1:ny+1] .= 2*vwest .- v[2,1:ny+1]
    v[nx+2,1:ny+1] .= 2*veast .- v[nx+1,1:ny+1]

    for i=2:nx
        for j=2:ny+1  # TEMPORARY u-velocity
            ut[i,j] = u[i,j] + dt*(-0.25*(((u[i+1,j]+u[i,j])^2-(u[i,j]+u[i-1,j])^2)/dx +
            ((u[i,j+1]+u[i,j])*(v[i+1,j]+v[i,j])-(u[i,j]+u[i,j-1])*(v[i+1,j-1]+v[i,j-1]))/dy) +
            m0/(0.5*(r[i+1,j]+r[i,j]))*((u[i+1,j]-2*u[i,j]+u[i-1,j])/dx^2 +
            (u[i,j+1]-2*u[i,j]+u[i,j-1])/dy^2 ) + gx)
        end
    end

    for i=2:nx+1
        for j=2:ny  # TEMPORARY v-velocity
            vt[i,j] = v[i,j] + dt*(-0.25*(((u[i,j+1]+u[i,j])*(v[i+1,j]+v[i,j]) -
            (u[i-1,j+1]+u[i-1,j])*(v[i,j]+v[i-1,j]))/dx +
            ((v[i,j+1]+v[i,j])^2-(v[i,j]+v[i,j-1])^2)/dy) +
            m0/(0.5*(r[i,j+1]+r[i,j]))*((v[i+1,j]-2*v[i,j]+v[i-1,j])/dx^2 +
            (v[i,j+1]-2*v[i,j]+v[i,j-1])/dy^2 ) + gy)
        end
    end

    # Compute source term and the coefficient for p(i,j)
    rt = copy(r)
    lrg = 1000
    rt[1:nx+2,1] .= lrg
    rt[1:nx+2,ny+2] .= lrg
    rt[1,1:ny+2] .= lrg
    rt[nx+2,1:ny+2] .= lrg

    for i=2:nx+1
        for j=2:ny+1
        tmp1[i,j] = (0.5/dt)*((ut[i,j]-ut[i-1,j])/dx+(vt[i,j]-vt[i,j-1])/dy)
        tmp2[i,j] = 1.0/((1.0/dx)*(1.0/(dx*(rt[i+1,j]+rt[i,j])) +
            1.0/(dx*(rt[i-1,j]+rt[i,j]))) +
            (1.0/dy)*(1.0/(dy*(rt[i,j+1]+rt[i,j])) +
            1.0/(dy*(rt[i,j-1]+rt[i,j]))))
        end
    end

    for it=1:maxit  # SOLVE FOR PRESSURE
        oldArray = copy(p)
        for i=2:nx+1
            for j=2:ny+1
            p[i,j] = (1.0-beta)*p[i,j] + beta*tmp2[i,j]*(
            (1.0/dx)*(p[i+1,j]/(dx*(rt[i+1,j]+rt[i,j])) +
            p[i-1,j]/(dx*(rt[i-1,j]+rt[i,j]))) +
            (1.0/dy)*(p[i,j+1]/(dy*(rt[i,j+1]+rt[i,j])) +
            p[i,j-1]/(dy*(rt[i,j-1]+rt[i,j]))) - tmp1[i,j])
            end
        end
        if maximum(abs.(oldArray .- p)) < maxError
            break
        end
    end

    for i=2:nx
        for j=2:ny+1  # CORRECT THE u-velocity
        u[i,j] = ut[i,j] - dt*(2.0/dx)*(p[i+1,j]-p[i,j])/(r[i+1,j]+r[i,j])
        end
    end

    for i=2:nx+1
        for j=2:ny  # CORRECT THE v-velocity
            v[i,j] = vt[i,j] - dt*(2.0/dy)*(p[i,j+1]-p[i,j])/(r[i,j+1]+r[i,j])
        end
    end

    # ADVECT DENSITY using centered difference plus diffusion
    ro = copy(r)
    for i=2:nx+1
        for j=2:ny+1
            r[i,j] = ro[i,j] - (0.5*dt/dx)*(u[i,j]*(ro[i+1,j] - ro[i,j]) -
            u[i-1,j]*(ro[i-1,j] - ro[i,j])) -
            (0.5*dt/dy)*(v[i,j]*(ro[i,j+1] - ro[i,j]) -
            v[i,j-1]*(ro[i,j-1] - ro[i,j])) +
            (m0*dt/dx/dx)*(ro[i+1,j] - 2.0*ro[i,j] + ro[i-1,j]) +
            (m0*dt/dy/dy)*(ro[i,j+1] - 2.0*ro[i,j] + ro[i,j-1])
        end
    end
    for i=2:nx
        for j=2:ny+1  # CORRECT THE u-velocity
            u[i,j] = ut[i,j] - dt*(2.0/dx)*(p[i+1,j]-p[i,j])/(r[i+1,j]+r[i,j])
        end
    end

    for i=2:nx+1
        for j=2:ny  # CORRECT THE v-velocity
            v[i,j] = vt[i,j] - dt*(2.0/dy)*(p[i,j+1]-p[i,j])/(r[i,j+1]+r[i,j])
        end
    end

    # ADVECT DENSITY using centered difference plus diffusion
    ro = copy(r)
    for i=2:nx+1
        for j=2:ny+1
            r[i,j] = ro[i,j] - (0.5*dt/dx)*(u[i,j]*(ro[i+1,j] + ro[i,j]) -
            u[i-1,j]*(ro[i-1,j] + ro[i,j])) -
            (0.5*dt/dy)*(v[i,j]*(ro[i,j+1] + ro[i,j]) -
            v[i,j-1]*(ro[i,j-1] + ro[i,j])) +
            (m0*dt/dx/dx)*(ro[i+1,j] - 2.0*ro[i,j] + ro[i-1,j]) +
            (m0*dt/dy/dy)*(ro[i,j+1] - 2.0*ro[i,j] + ro[i,j-1])
        end 
    end

    global mtime = mtime + dt  # plot the results
    uu[1:nx+1,1:ny+1] .= 0.5 .* (u[1:nx+1,2:ny+2] .+ u[1:nx+1,1:ny+1])
    vv[1:nx+1,1:ny+1] .= 0.5 .* (v[2:nx+2,1:ny+1] .+ v[1:nx+1,1:ny+1])
    xh = [dx*(i-1) for i=1:nx+1]
    yh = [dy*(j-1) for j=1:ny+1]

    # The following lines are for plotting which is typically done using packages like Plots.jl or PyPlot.jl in Julia.
    # Here is an example of how you might do it using Plots.jl:

    if is % 10 == 0
        println("time = ", mtime)
        contour(x, y, r', aspect_ratio=:equal, xlims=(0, Lx), ylims=(0, Ly))
        quiver!(xh, yh, quiver=(uu', vv'), color=:red)
        Plots.savefig("out_vis/step_$is.png")
    end
end

