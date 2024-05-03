# this solver is based on algorithm by DNS solver.
const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using LinearAlgebra, Printf
using MAT, Plots

#@views function main(; do_vis=true, do_save=false)
    # physics
    Lx = 1.0 # [m]
    Ly = 1.0 # [m]
    gx = 0.0
    gy  = -100.0
    ρ₁ = 1.0 # [kg/m^3]
    ρ₂ = 2.0 # [kg/m^3]
    m0 = 0.01
    unorth = 0.0
    usouth = 0.0
    veast  = 0.0
    vwest  = 0.0
    time = 0.0
    nstep = 1
    rad = 0.15
    xcenter = 0.5
    ycenter = 0.7

    # numerics
    nx = 32
    ny = 32
    dt = 0.00125
    nsteps = 100
    nvis = 10
    maxit = 200
    maxerr = 1e-3
    beta = 1.2

    # set the grid
    dx = Lx/nx
    dy = Ly/ny
    xc = LinRange(-0.5*dx, Lx+0.5*dx, nx+2)
    yc = LinRange(-0.5*dy, Ly+0.5*dy, ny+2)
    xv = LinRange(0.0, Lx, nx+1)
    yv = LinRange(0.0, Ly, ny+1)

    # allocate
    ρ = @zeros(nx+2, ny+2)
    ρₜ = @zeros(nx+2, ny+2)
    u = @zeros(nx+1, ny+2)
    v = @zeros(nx+2, ny+1)
    p = @zeros(nx+2, ny+2)
    ut = @zeros(nx+1, ny+2)
    vt = @zeros(nx+2, ny+1)
    uu = @zeros(nx+1, ny+1)
    vv = @zeros(nx+1, ny+1)
    tmp1 = @zeros(nx+2, ny+2)
    tmp2 = @zeros(nx+2, ny+2)

    ρ .= ρ₁
    @hide_communication (16,2,2) begin
        # set the density
        @parallel set_density!(ρ, xcenter, ycenter, xc, yc, rad, ρ₂)

        # start loop
        for is = 1:nsteps
            time += dt
            # tangential velocity
            @parallel set_tangential!(u, v, usouth, unorth, vwest, veast)
            # temporary u-velocity
            @parallel temp_uvel!(ut, u, ρ, dt, dx, dy, g, m0)
            # temporary v-velocity
            @parallel temp_vvel!(vt, v, ρ, dt, dx, dy, g, m0)
            # compute source term
            @parallel set_boundary!(ρₜ, ρ)
            @parallel compute_source!(tmp1, tmp2, ρ, ut, vt, dt, dx, dy)

            # solve pressure
            for it = 1:maxit
                oldp = copy(p)
                @parallel solve_pressure!(p, tmp1, tmp2, ρ, beta, dx, dy)
                if maximum(abs.(oldp-p)) < maxerr
                    break
                end
            end
            # correct u-velocity
            @parallel correct_uvel!(u, ut, p, ρ, dt, dx)
            # correct v-velocity
            @parallel correct_vvel!(v, vt, p, ρ, dt, dy)
            # advect density
            @parallel advect_density!(ρ, ρₜ, u, v, dt, dx, dy, m0)
            # calculate the velocity at the cell centers
            @parallel velocity!(u, v, uu, vv)
            if is % nvis == 0
                println("time = $time")
                p3=heatmap(xc,yc,Array(uu)';aspect_ratio=1,xlims=(0,Lx),ylims=(0,Ly),title="UU")
                p4=heatmap(xc,yc,Array(vv)';aspect_ratio=1,xlims=(0,Lx),ylims=(0,Ly),title="VV")
                display(plot(p3,p4))
            end
        end
    end

#end

@parallel_indices (ix,iy) function set_tangential!(u,v,usouth, unorth,vwest, veast )
    if ix > 1 && ix < size(u,1) && iy > 1 && iy < size(v,2)
        u[ix,1] = 2*usouth - u[ix,2]
        u[ix,end] = 2*unorth - u[ix,end-1]
        v[1,iy] = 2*vwest - v[2,iy]
        v[end,iy] = 2*veast - v[end-1,iy]
    end
    return
end
# set the density
@parallel_indices (ix,iy) function set_density!(ρ, xcenter,ycenter,xc,yc,rad,r2)
    if (ix>1 && ix<size(ρ,1) && iy>1 && iy<size(ρ,2))
        x = xc[ix]
        y = yc[iy]
        if (x-xcenter)^2 + (y-ycenter)^2 < rad^2
        ρ[ix,iy] = r2;
        end
    end
    return
end
@parallel_indices (ix,iy) function set_boundary!(ρₜ,ρ)
    if ix > 1 && ix < size(ρₜ,1) && iy > 1 && iy < size(ρₜ,2)
        ρₜ[ix,iy] = ρ[ix,iy]
        lrg = 1000.0
        ρₜ[ix,1] = lrg
        ρₜ[ix,end] = lrg
        ρₜ[1,iy] = lrg
        ρₜ[end,iy] = lrg
    end
    return
end
@parallel_indices (ix,iy) function compute_source!(tmp1,tmp2,ρₜ,ut,vt,dt,dx,dy)
    if ix > 1 && ix < size(tmp1,1) && iy > 1 && iy < size(tmp1,2)
    tmp1[ix,iy] =(0.5/dt) * ( (ut[ix,iy]-ut[ix-1,iy])/dx + (vt[ix,iy]-vt[ix,iy-1])/dy )
    tmp2[ix,iy] = 1/( (1/dx) *(1/(dx*(ρₜ[ix+1,iy]+ρₜ[ix,iy]))
                             + 1/(dx*(ρₜ[ix-1,iy]+ρₜ[ix,iy])))
     + 1/dy*(1/(dy*(ρₜ[ix,iy+1]+ρₜ[ix,iy])) + 1/(dy*(ρₜ[ix,iy-1]+ρₜ[ix,iy])) ) )
    end
    return
end

@parallel_indices (ix,iy) function temp_uvel!(ut,u,ρ,dt,dx,dy,gx,m0)
    if ix > 1 && ix < size(ut,1) && iy > 1 && iy < size(ut,2)
    ut[ix,iy] = u[ix,iy]+dt*(-0.25((u[ix+1,iy]+u[ix,iy])^2 
    - (u[ix,iy]+u[ix-1,iy])^2)/dx + ((u[ix,iy+1] + u[ix,iy])*(v[ix+1,iy] + v[ix,iy])
    - (u[ix,iy]+u[ix,iy-1])*(v[ix+1,iy-1]+v[ix,iy-1])/dy)
    + m0/(0.5*(ρ[ix+1,iy]+ρ[ix,iy]))*((u[ix+1,iy]-2*u[ix,iy]+u[ix-1,iy])/dx^2
    + (u[ix,iy+1]-2*u[ix,iy]+u[ix,iy-1])/dy^2) + gx )
    end
    return
end

@parallel_indices (ix,iy) function temp_vvel!(vt,v,ρ,dt,dx,dy,gy,m0)
    if ix > 1 && ix < size(vt,1) && iy > 1 && iy < size(vt,2)
    vt[ix,iy] = v[ix,iy]+dt*(-0.25*(((u[ix,iy+1]+u[ix,iy])*(v[ix+1,iy]+v[ix,iy])
    - (u[ix-1,iy+1]+u[ix-1,iy])*(v[ix,iy]+v[ix-1,iy]))/dx 
    + ((v[ix,iy+1]+v[ix,iy])^2 - (v[ix,iy]+v[ix,iy-1])^2)/dy)
    + m0/(0.5*(ρ[ix,iy+1]+ρ[ix,iy]))*((v[ix+1,iy]-2*v[ix,iy]+v[ix-1,iy])/dx^2
    + (v[ix,iy+1]-2*v[ix,iy]+v[ix,iy-1])/dy^2) + gy )
    end
    return
end

@parallel_indices (ix,iy) function solve_pressure!(p,tmp1,tmp2,ρₜ,beta,dx,dy)
    if ix > 1 && ix < size(p,1) && iy > 1 && iy < size(p,2)
    p[ix,iy] = (1.0 - beta)*p[ix,iy] + beta*tmp2[ix,iy]
   *((1/dx)*(p[ix+1,iy]/(dx*(ρₜ[ix+1,iy]+ρₜ[ix,iy])) + p[ix-1,iy]/(dx*(ρₜ[ix-1,iy]+ρₜ[ix,iy]))) + 
     (1/dy)*(p[ix,iy+1]/(dy*(ρₜ[ix,iy+1]+ρₜ[ix,iy])) + p[ix,iy-1]/(dy*(ρₜ[ix,iy-1]+ρₜ[ix,iy])))
    -tmp1[ix,iy])
    end
    return
end
@parallel_indices (ix,iy) function correct_uvel!(u,ut,p,ρ,dt,dx)
    if ix > 1 && ix < size(u,1) && iy > 1 && iy < size(u,2)
    u[ix,iy] = ut[ix,iy] - dt*(2.0/dx)*(p[ix+1,iy]-p[ix,iy])/(ρ[ix+1,iy]+ρ[ix,iy])
    end
    return
end

@parallel_indices (ix,iy) function correct_vvel!(v,vt,p,ρ,dt,dy)
    if ix > 1 && ix < size(v,1) && iy > 1 && iy < size(v,2)
    v[ix,iy] = vt[ix,iy] - dt*(2.0/dy)*(p[ix,iy+1]-p[ix,iy])/(ρ[ix,iy+1]+ρ[ix,iy])
    end
    return
end

@parallel_indices (ix,iy) function advect_density!(ρ,ρo,u,v,dt,dx,dy,m0)
    if ix > 1 && ix < size(ρ,1) && iy > 1 && iy < size(ρ,2)
    ρ[ix,iy] = ρo[ix,iy] - dt*(0.5/dx)*(u[ix,iy]*(ρo[ix+1,iy]+ρo[ix,iy]) - u[ix-1,iy]*(ρo[ix-1,iy]+ρo[ix,iy]))
    - dt*(0.5/dy)*(v[ix,iy]*(ρ[ix,iy+1]+ρ[ix,iy]) - v[ix,iy-1]*(ρ[ix,iy-1]+ρ[ix,iy]))
    +(m0*dt/(dx*dx))*(ρo[ix+1,iy]-2*ρo[ix,iy]+ρo[ix-1,iy])
    +(m0*dt/(dy*dy))*(ρo[ix,iy+1]-2*ρo[ix,iy]+ρo[ix,iy-1])
    end
    return
end

@parallel_indices (ix,iy) function velocity!(u,v,uu,vv)
    if ix > 1 && ix < size(u,1) && iy > 1 && iy < size(u,2)
        uu[ix,iy] = 0.5*(u[ix,iy+1]+u[ix,iy])
        vv[ix,iy] = 0.5*(v[ix+1,iy]+v[ix,iy])
    end
    return
end


