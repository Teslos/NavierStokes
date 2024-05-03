# A very simple Navier-Stokes solver for a drop falling in a rectangular
# domain. The viscosity is taken to be a constant and forward in time,
# centered in space discretization is used. The density is advected by a 
# simple upwind scheme.
using CUDA, Printf, LinearAlgebra
using CairoMakie

function set_density!(r, xc, yc, x, y, rad, rho2)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    if(i > 1 && i <= size(r,1)-1 && j > 1 && j <= size(r,2)-1)
            if((x[i]-xc)^2 + (y[j]-yc)^2 < rad^2)
                r[i,j] = rho2
            end
    end
    return
end

function set_tangential!(u, usouth, unorth, v, vwest, veast, nx, ny)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    if (i <= size(u,1) && j <= size(v,2))
        u[i,1] = 2*usouth - u[i,2]
        u[i,ny+2] = 2*unorth - u[i,ny+1]
    
        v[1,j] = 2*vwest - v[2,j]
        v[nx+2,j] = 2*veast - v[nx+1,j]
    end 
    return
end

function temp_uvel!(ut, u, v, r, dt, dx, dy, gx, m0)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    if (i > 1 && i <= size(ut,1)-1 && j>1 && j <= size(ut,2)-1)
             # TEMPORARY u-velocity
            ut[i,j] = u[i,j] + dt*(-0.25*(((u[i+1,j]+u[i,j])^2-(u[i,j]+u[i-1,j])^2)/dx +
            ((u[i,j+1]+u[i,j])*(v[i+1,j]+v[i,j])-(u[i,j]+u[i,j-1])*(v[i+1,j-1]+v[i,j-1]))/dy) +
            m0/(0.5*(r[i+1,j]+r[i,j]))*((u[i+1,j]-2*u[i,j]+u[i-1,j])/dx^2 +
            (u[i,j+1]-2*u[i,j]+u[i,j-1])/dy^2 ) + gx)
    end
    return
end

function temp_vvel!(vt, u, v, r, dt, dx, dy, gy, m0)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    if (i > 1 && i <= size(vt,1)-1 && j > 1 && j <= size(vt,2)-1)
            # TEMPORARY v-velocity
            vt[i,j] = v[i,j] + dt*(-0.25*(((u[i,j+1]+u[i,j])*(v[i+1,j]+v[i,j]) -
            (u[i-1,j+1]+u[i-1,j])*(v[i,j]+v[i-1,j]))/dx +
            ((v[i,j+1]+v[i,j])^2-(v[i,j]+v[i,j-1])^2)/dy) +
            m0/(0.5*(r[i,j+1]+r[i,j]))*((v[i+1,j]-2*v[i,j]+v[i-1,j])/dx^2 +
            (v[i,j+1]-2*v[i,j]+v[i,j-1])/dy^2 ) + gy)
    end
    return
end

function set_boundary!(rt)
    lrg = 1000
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    if (i <= size(rt,1) && j <= size(rt,2))
        rt[i,1] = lrg
        rt[i,end] = lrg
        rt[1,j] = lrg
        rt[end,j] = lrg
    end
    return
end

function compute_sources!(tmp1, tmp2, rt, ut, vt, dt, dx, dy, nx, ny)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    if (i > 1 && i <= size(tmp1,1)-1 && j > 1 && j <= size(tmp1,2)-1)
        tmp1[i,j] = (0.5/dt)*((ut[i,j]-ut[i-1,j])/dx+(vt[i,j]-vt[i,j-1])/dy)
        tmp2[i,j] = 1.0/((1.0/dx)*(1.0/(dx*(rt[i+1,j]+rt[i,j])) +
            1.0/(dx*(rt[i-1,j]+rt[i,j]))) +
            (1.0/dy)*(1.0/(dy*(rt[i,j+1]+rt[i,j])) +
            1.0/(dy*(rt[i,j-1]+rt[i,j]))))
    end
    return
end

function solve_pressure!(p, tmp1, tmp2, rt, beta, dx, dy, nx, ny)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        j = (blockIdx().y-1) * blockDim().y + threadIdx().y
        if (i > 1 && i <= size(p,1)-1 && j > 1 && j <= size(p,2)-1)
                p[i,j] = (1.0-beta)*p[i,j] + beta*tmp2[i,j]*(
                (1.0/dx)*(p[i+1,j]/(dx*(rt[i+1,j]+rt[i,j])) +
                p[i-1,j]/(dx*(rt[i-1,j]+rt[i,j]))) +
                (1.0/dy)*(p[i,j+1]/(dy*(rt[i,j+1]+rt[i,j])) +
                p[i,j-1]/(dy*(rt[i,j-1]+rt[i,j]))) - tmp1[i,j])
        end
    return
end

function update_dPdτ!(p, dPdτ, tmp1, rt, dt, dτ, damp, dx, dy)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    if (i > 1 && i <= size(p,1)-1 && j > 1 && j <= size(p,2)-1)
    dPdτ[i,j] = dPdτ[i,j]*(1.0-damp) + dτ*(
    (1.0/dx)*((p[i+1,j]-p[i,j])/(dx*(rt[i+1,j]+rt[i,j])) -
    (p[i,j]-p[i-1,j])/(dx*(rt[i-1,j]+rt[i,j]))) +
    (1.0/dy)*((p[i,j+1]-p[i,j])/(dy*(rt[i,j+1]+rt[i,j])) -
    (p[i,j]-p[i,j-1])/(dy*(rt[i,j-1]+rt[i,j]))) - tmp1[i,j])
    end
    return
end

function update_p!(p, dPdτ, dτ)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    if (i > 1 && i <= size(p,1)-1 && j > 1 && j <= size(p,2)-1)
    p[i,j] = p[i,j] + dτ*dPdτ[i,j]
    end
    return
end

function correct_vvel!(v, vt, p, r, dt, dy)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    if (i > 1 && i <= size(v,1)-1 && j > 1 && j <= size(v,2)-1)
        v[i,j] = vt[i,j] - dt*(2.0/dy)*(p[i,j+1]-p[i,j])/(r[i,j+1]+r[i,j])
    end
    return
end

function correct_uvel!(u, ut, p, r, dt, dx)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    if (i > 1 && i <= size(u,1)-1 && j > 1 && j <= size(u,2)-1)
        u[i,j] = ut[i,j] - dt*(2.0/dx)*(p[i+1,j]-p[i,j])/(r[i+1,j]+r[i,j])
    end
    return
end

function advect_density!(r, ro, u, v, dt, dx, dy, m0, nx, ny)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    if (i > 1 && i <= size(r,1)-1 && j > 1 && j <= size(r,2)-1)
            r[i,j] = ro[i,j] - (0.5*dt/dx)*(u[i,j]*(ro[i+1,j] - ro[i,j]) -
            u[i-1,j]*(ro[i-1,j] - ro[i,j])) -
            (0.5*dt/dy)*(v[i,j]*(ro[i,j+1] - ro[i,j]) -
            v[i,j-1]*(ro[i,j-1] - ro[i,j])) +
            (m0*dt/dx/dx)*(ro[i+1,j] - 2.0*ro[i,j] + ro[i-1,j]) +
            (m0*dt/dy/dy)*(ro[i,j+1] - 2.0*ro[i,j] + ro[i,j-1])
    end
    return
end

function assign!(uu, u, vv, v)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    if (i <= size(uu,1)-1 && j <= size(uu,2)-1)
        uu[i,j] = 0.5*(u[i,j+1]+u[i,j])
        vv[i,j] = 0.5*(v[i+1,j]+v[i,j])
    end
    return
end

# domain size and physical variables
Lx=1.0; Ly=1.0; gx=0;gy= -100.0; rho1=1.0; rho2=2.0; m0=0.01; rro=rho1;
unorth=0; usouth=0; veast=0; vwest=0; mtime=0.0;
rad=0.15; xc=0.5; yc=0.7; # Initial drop size and location
CFLτ=0.9/sqrt(2)  # CFL number for τ
CFL_vis=1/4.1; CFL_adv=1.0;
vin = 1.0

# Numerics
BLOCKX = 8
BLOCKY = 8
GRIDX  = 8*4
GRIDY  = 8*4


# Numerical variables
nx=BLOCKX*GRIDX; ny=BLOCKY*GRIDY; dt=0.00125; nstep=6000; maxit=200; maxError=0.001; beta=1.2;
dx=Lx/nx; dy=Ly/ny;
damp=0.72; dτ=CFLτ*dy

# Zero various arrays
u=CUDA.zeros(nx+1,ny+2); v=CUDA.zeros(nx+2,ny+1); p=CUDA.ones(nx+2, ny+2); dPdτ=CUDA.zeros(nx+2,ny+2);
ut=CUDA.zeros(nx+1,ny+2); vt=CUDA.zeros(nx+2,ny+1); tmp1=CUDA.zeros(nx+2,ny+2);
uu=CUDA.zeros(nx+1,ny+1); vv=CUDA.zeros(nx+1,ny+1); tmp2=CUDA.zeros(nx+2,ny+2);

# Set the grid
x=CuArray([dx*(i-1.5) for i=1:nx+2]); y=CuArray([dy*(j-1.5) for j=1:ny+2])

dt = min(dt, min(dx, dy)/sqrt(gx^2 + gy^2)/8.1)
dt = min(CFL_vis*dx^2*rho1/m0, CFL_adv*dx/vin)
println("Time step is: ", dt)

cuthreads = (BLOCKX+2, BLOCKY+2, 1)
cublocks  = (GRIDX+2, GRIDY+2, 1)

# Set the density
r=CUDA.ones(nx+2,ny+2)*rho1;
@cuda blocks = cublocks threads = cuthreads set_density!(r, xc, yc, x, y, rad, rho2)
synchronize()
# ======================START TIME LOOP================================
for is = 1: nstep
    # tangential velocity at boundaries
    @cuda blocks = cublocks threads = cuthreads set_tangential!(u, usouth, unorth, v, vwest, veast, nx, ny)
    synchronize()
    # temporary u-velocity
    @cuda blocks = cublocks threads = cuthreads temp_uvel!(ut, u, v, r, dt, dx, dy, gx, m0)
    synchronize()
    #println("ut = ", ut)    
    # temporary v-velocity
    @cuda blocks = cublocks threads = cuthreads temp_vvel!(vt, u, v, r, dt, dx, dy, gy, m0)
    synchronize()
    #println("size vt = ", size(vt))
    #println("vt = ", vt)
    #exit()
    # Compute source term and the coefficient for p(i,j)
    rt = CUDA.copy(r)
    @cuda blocks = cublocks threads = cuthreads set_boundary!(rt)
    synchronize()
    #println("rt = ", rt)
    #exit()
    @cuda blocks = cublocks threads = cuthreads compute_sources!(tmp1, tmp2, rt, ut, vt, dt, dx, dy, nx, ny)
    synchronize()
    #println("tmp1 = ", tmp1)
    #println("tmp2 = ", tmp2)
    #exit()
    # Solve for pressure
    for it=1:maxit  # SOLVE FOR PRESSURE
        oldArray = CUDA.copy(p)
        #@cuda blocks = cublocks threads = cuthreads solve_pressure!(p, tmp1, tmp2, rt, beta, dx, dy, nx, ny)
        #synchronize()
        #max_diff = CUDA.@sync CUDA.reduce(max, abs.(oldArray.- p))
        @cuda blocks = cublocks threads = cuthreads update_dPdτ!(p, dPdτ, tmp1, rt, dt, dτ, damp, dx, dy)
        synchronize()
        @cuda blocks = cublocks threads = cuthreads update_p!(p, dPdτ, dτ)
        synchronize()
        max_diff = CUDA.@sync CUDA.reduce(max, abs.(oldArray.- p))
        #println("max_diff in pressure = ", max_diff)
        if max_diff < maxError
            break
        end
    end
    # correct u-velocity
    @cuda blocks = cublocks threads = cuthreads correct_uvel!(u, ut, p, r, dt, dx)
    synchronize()
    # correct v-velocity
    @cuda blocks = cublocks threads = cuthreads correct_vvel!(v, vt, p, r, dt, dy)
    synchronize()
    # ADVECT DENSITY using centered difference plus diffusion
    ro = CUDA.copy(r)
    @cuda blocks = cublocks threads = cuthreads advect_density!(r, ro, u, v, dt, dx, dy, m0, nx, ny)
    synchronize()
    global mtime = mtime + dt  # plot the results
    @cuda blocks = cublocks threads = cuthreads assign!(uu, u, vv, v) 
    synchronize()
    xh = [dx*(i-1) for i=1:nx+1]
    yh = [dy*(j-1) for j=1:ny+1]

    # The following lines are for plotting which is typically done using packages like Plots.jl or PyPlot.jl in Julia.
    # Here is an example of how you might do it using Plots.jl:

    if is % 100 == 0
        println("time = ", mtime)
        println("max(u) = ", maximum(u))
        println("max(v) = ", maximum(v))
        s = 4
        fig = Figure()
        ax = Axis(fig[1, 1])
        contour!(ax, Array(x), Array(y), Array(r), levels=20)
        arrows!(ax, xh[1:s:end], yh[1:s:end], Array(uu)[1:s:end,1:s:end], Array(vv)[1:s:end,1:s:end], 
            arrowsize=0.03, lengthscale=0.02, color=:red)
        limits!(ax, 0, Lx, 0, Ly)
        save("out_vis1/step_$is.png", fig)
    end
    #=
        contour(Array(x), Array(y), Array(r)', aspect_ratio=:equal, xlims=(0, Lx), ylims=(0, Ly))
        quiver!(xh, yh, quiver=(Array(uu)', Array(vv)'), color=:red)
        Plots.savefig("out_vis1/step_$is.png")
    =#
end

