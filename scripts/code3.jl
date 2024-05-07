# code3.m
# A very simple Navier-Stokes solver for a drop falling in a rectangular 
# box. A forward in time, centered in space discretization is used. 
# The density is advected by a front tracking scheme and surface tension
# and variable viscosity is included
using CairoMakie
# domain size and physical variables
Lx = 1.0; Ly = 1.0
gx = 0.0; gy = -100.0
rho1 = 1.0; rho2 = 2.0
m1 = 0.01; m2 = 0.05
sigma = 10
rro = rho1
unorth = 0.0; usouth = 0.0; veast = 0.0; vwest = 0.0; time = 0.0
rad = 0.15; xc = 0.5; yc = 0.7  # Initial drop size and location

# Numerical variables
nx = 32; ny = 32;
dt = 0.00125
nstep = 300; maxit = 200
maxError = 0.001; beta = 1.2

# Zero various arrays
u = zeros(nx+1, ny+2)
v = zeros(nx+2, ny+1)
p = zeros(nx+2, ny+2)
ut = zeros(nx+1, ny+2)
vt = zeros(nx+2, ny+1)
tmp1 = zeros(nx+2, ny+2)
uu = zeros(nx+1, ny+1)
vv = zeros(nx+1, ny+1)
tmp2 = zeros(nx+2, ny+2)
fx = zeros(nx+2, ny+2)
fy = zeros(nx+2, ny+2)
un = zeros(nx+1, ny+2)
vn = zeros(nx+2, ny+1)   # second order

# Set the grid 
dx = Lx/nx; dy = Ly/ny
x = [dx*(i-1.5) for i in 1:nx+2]
y = [dy*(j-1.5) for j in 1:ny+2]

# Set density and viscosity in the domain and the drop
r = fill(rho1, nx+2, ny+2)
m = fill(m1, nx+2, ny+2)
rn = zeros(nx+2, ny+2)
mn = zeros(nx+2, ny+2)   # second order
for i in 2:nx+1, j in 2:ny+1
  if (x[i]-xc)^2 + (y[j]-yc)^2 < rad^2
    r[i,j] = rho2
    m[i,j] = m2
  end
end
dt = min(dt,0.25 * min(dx, dy)^2 *min(rho1, rho2) / maximum(m))
println("dt = ", dt)
# SETUP THE FRONT
Nf = 100
xf = zeros(Nf+2)
yf = zeros(Nf+2)
xfn = zeros(Nf+2)
yfn = zeros(Nf+2)     # second order

uf = zeros(Nf+2)
vf = zeros(Nf+2)
tx = zeros(Nf+2)
ty = zeros(Nf+2)

for l in 1:Nf+2
    xf[l] = xc - rad * sin(2.0 * pi * (l - 1) / Nf)
    yf[l] = yc + rad * cos(2.0 * pi * (l - 1) / Nf)
end

# -------------------- START TIME LOOP ------------------------
for is in 1:nstep
    global u, v, p, r, ut, vt, tmp1, uu, vv, tmp2, fx, fy, un, vn, rn, m, mn, xf, yf, xfn, yfn, tx, ty, uf, vf, Nf, time
    un = copy(u)
    vn = copy(v)
    rn = copy(r)
    mn = copy(m)
    xfn = copy(xf)
    yfn = copy(yf)   # second order
    for substep in 1:2    # second order

        #------------------ FIND SURFACE TENSION --------------
        fx = zeros(nx+2, ny+2)
        fy = zeros(nx+2, ny+2)  # Set fx & fy to zero
        tx = zeros(size(xf))
        ty = zeros(size(yf))
        uf = zeros(size(xf))
        vf = zeros(size(yf))
        
        for l in 1:Nf+1
            ds = sqrt((xf[l+1] - xf[l])^2 + (yf[l+1] - yf[l])^2)
            tx[l] = (xf[l+1] - xf[l]) / ds
            ty[l] = (yf[l+1] - yf[l]) / ds  # Tangent vectors
        end
        tx[Nf+2] = tx[2]
        ty[Nf+2] = ty[2]
        for l in 2:Nf+1     # Distribute to the fixed grid
            nfx = sigma * (tx[l] - tx[l-1])
            nfy = sigma * (ty[l] - ty[l-1])

            ip = Int(floor(Int, xf[l] / dx)) + 1
            jp = Int(floor(Int, (yf[l] + 0.5 * dy) / dy)) + 1
            ax = xf[l] / dx - ip + 1
            ay = (yf[l] + 0.5 * dy) / dy - jp + 1
            fx[ip, jp] = fx[ip, jp] + (1.0 - ax) * (1.0 - ay) * nfx / dx / dy
            fx[ip+1, jp] = fx[ip+1, jp] + ax * (1.0 - ay) * nfx / dx / dy
            fx[ip, jp+1] = fx[ip, jp+1] + (1.0 - ax) * ay * nfx / dx / dy
            fx[ip+1, jp+1] = fx[ip+1, jp+1] + ax * ay * nfx / dx / dy

            ip = Int(floor(Int, (xf[l] + 0.5 * dx) / dx)) + 1
            jp = Int(floor(Int, yf[l] / dy)) + 1
            ax = (xf[l] + 0.5 * dx) / dx - ip + 1
            ay = yf[l] / dy - jp + 1
            fy[ip, jp] = fy[ip, jp] + (1.0 - ax) * (1.0 - ay) * nfy / dx / dy
            fy[ip+1, jp] = fy[ip+1, jp] + ax * (1.0 - ay) * nfy / dx / dy
            fy[ip, jp+1] = fy[ip, jp+1] + (1.0 - ax) * ay * nfy / dx / dy
            fy[ip+1, jp+1] = fy[ip+1, jp+1] + ax * ay * nfy / dx / dy
        end
        #println("fy = ", fy)

        # tangential velocity at boundaries
        u[1:nx+1, 1] .= 2 * usouth .- u[1:nx+1, 2]
        u[1:nx+1, ny+2] .= 2 * unorth .- u[1:nx+1, ny+1]
        v[1, 1:ny+1] .= 2 * vwest .- v[2, 1:ny+1]
        v[nx+2, 1:ny+1] .= 2 * veast .- v[nx+1, 1:ny+1]

        # temporary u-velocity - advection
        for i in 2:nx
            for j in 2:ny+1
                ut[i, j] = u[i, j] + dt * (-0.25 * 
                (((u[i+1, j] + u[i, j])^2 - (u[i, j] + u[i-1, j])^2) / dx 
                + ((u[i, j+1] + u[i, j]) * (v[i+1, j] + v[i, j]) 
                - (u[i, j] + u[i, j-1]) * (v[i+1, j-1] + v[i, j-1])) / dy) 
                + fx[i, j] / (0.5 * (r[i+1, j] + r[i, j])) 
                - (1.0 - rro / (0.5 * (r[i+1, j] + r[i, j]))) * gx)
            end
        end
        # temporary v-velocity - advection
        for i in 2:nx+1
            for j in 2:ny
                vt[i, j] = v[i, j] + dt * (-0.25 * 
                (((u[i, j+1] + u[i, j]) * (v[i+1, j] + v[i, j]) 
                - (u[i-1, j+1] + u[i-1, j]) * (v[i, j] + v[i-1, j])) / dx 
                + ((v[i, j+1] + v[i, j])^2 - (v[i, j] + v[i, j-1])^2) / dy) 
                + fy[i, j] / (0.5 * (r[i, j+1] + r[i, j])) 
                - (1.0 - rro / (0.5 * (r[i, j+1] + r[i, j]))) * gy)
            end
        end
        # println("vt = ", vt)
        # exit()
        # temporary u-velocity - diffusion
        for i in 2:nx
            for j in 2:ny+1
                ut[i, j] = ut[i, j] + dt * ((1. / dx) * 2. 
                * (m[i+1, j] * (1. / dx) * (u[i+1, j] - u[i, j]) 
                - m[i, j] * (1. / dx) * (u[i, j] - u[i-1, j])) 
                + (1. / dy) * (0.25 * (m[i, j] + m[i+1, j] + m[i+1, j+1] + m[i, j+1]) * ((1. / dy) * (u[i, j+1] - u[i, j]) 
                + (1. / dx) * (v[i+1, j] - v[i, j])) - 
                0.25 * (m[i, j] + m[i+1, j] + m[i+1, j-1] + m[i, j-1]) * ((1. / dy) * (u[i, j] - u[i, j-1])
                + (1. / dx) * (v[i+1, j-1] - v[i, j-1]))) / (0.5 * (r[i+1, j] + r[i, j])))
            end
        end
        # temporary v-velocity - diffusion
        for i=2:nx+1 
            for j=2:ny
                vt[i,j] = vt[i,j] + dt * (
                (1.0/dx) * (0.25 * (m[i,j] + m[i+1,j] + m[i+1,j+1] + m[i,j+1]) * 
                ((1.0/dy) * (u[i,j+1] - u[i,j]) + (1.0/dx) * (v[i+1,j] - v[i,j])) 
                - 0.25 * (m[i,j] + m[i,j+1] + m[i-1,j+1] + m[i-1,j]) * 
                ((1.0/dy) * (u[i-1,j+1] - u[i-1,j]) + (1.0/dx) * (v[i,j] - v[i-1,j]))) 
                + (1.0/dy) * 2.0 * (m[i,j+1] * (1.0/dy) * (v[i,j+1] - v[i,j]) 
                - m[i,j] * (1.0/dy) * (v[i,j] - v[i,j-1])) ) / (0.5 * (r[i,j+1] + r[i,j]))
            end
        end
        
        rt = copy(r)
        lrg = 1000
        rt[1:nx+2, 1] .= lrg
        rt[1:nx+2, ny+2] .= lrg
        rt[1, 1:ny+2] .= lrg
        rt[nx+2, 1:ny+2] .= lrg
        
        for i=2:nx+1
            for j=2:ny+1
                tmp1[i,j] = (0.5/dt) * ((ut[i,j] - ut[i-1,j]) / dx + (vt[i,j] - vt[i,j-1]) / dy)
                tmp2[i,j] = 1.0 / ((1.0/dx) * (1.0 / (dx * (rt[i+1,j] + rt[i,j])) + 
                            1.0 / (dx * (rt[i-1,j] + rt[i,j]))) 
                            + (1.0/dy) * (1.0 / (dy * (rt[i,j+1] + rt[i,j]))
                            + 1.0 / (dy * (rt[i,j-1] + rt[i,j]))))
            end
        end
        #println("tmp2 = ", tmp2)
        #exit()

        # solve for pressure
        for it=1:maxit
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
            #display(p)
            #println("tmp1 = ", tmp1)
            #println("tmp2 = ", tmp2)
            #exit()
            if maximum(abs.(oldArray .- p)) < maxError
                println("Converged at iteration ", it)
                break
            end
        end
        #display(p)  
        #exit()  
        # correct the u-velocity field
        for i=2:nx
            for j=2:ny+1
                u[i,j] = ut[i,j] - dt * (2.0/dx) * (p[i+1,j] - p[i,j]) / (r[i+1,j] + r[i,j])
            end
        end
        #display(u)
        # correct the v-velocity field
        for i=2:nx+1
            for j=2:ny
                v[i,j] = vt[i,j] - dt * (2.0/dy) * (p[i,j+1] - p[i,j]) / (r[i,j+1] + r[i,j])
            end
        end
        #display(v)
        #exit()
        # advect the front
        for l=2:Nf+1
            ip = Int(floor(xf[l] / dx)) + 1
            jp = Int(floor((yf[l] + 0.5 * dy) / dy)) + 1
            ax = xf[l] / dx - ip + 1
            ay = (yf[l] + 0.5 * dy) / dy - jp + 1

            #=
            uf[l] = (1.0 - ax)*(1.0 - ay)*u[ip,jp] 
                    + ax*(1.0 - ay)*u[ip+1,jp] 
                    + (1.0 - ax)*ay*u[ip,jp+1]+ax*ay*u[ip+1,jp+1]
            =#
            v1 = (1.0-ax)*(1.0-ay)*u[ip,jp]
            v2 = ax*(1.0-ay)*u[ip+1,jp]
            v3 = (1.0-ax)*ay*u[ip,jp+1]
            v4 = ax*ay*u[ip+1,jp+1]
            uf[l] = v1+v2+v3+v4

            ip = Int(floor((xf[l] + 0.5 * dx) / dx)) + 1
            jp = Int(floor(yf[l] / dy)) + 1
            ax = (xf[l] + 0.5 * dx) / dx - ip + 1
            ay = yf[l] / dy - jp + 1
            #=
            vf[l] = (1.0 - ax) * (1.0 - ay) * v[ip,jp] 
                    + ax * (1.0 - ay) * v[ip+1,jp] 
                    + (1.0 - ax) * ay * v[ip,jp+1] + ax * ay * v[ip+1,jp+1]
            =#
            v1 = (1.0-ax)*(1.0-ay)*v[ip,jp]
            v2 = ax*(1.0-ay)*v[ip+1,jp]
            v3 = (1.0-ax)*ay*v[ip,jp+1]
            v4 = ax*ay*v[ip+1,jp+1]
            vf[l] = v1+v2+v3+v4
        end
        #display(uf)
        #display(vf)
        #exit()
        # move the front
        for i=2:Nf+1
            xf[i] = xf[i] + dt * uf[i]
            yf[i] = yf[i] + dt * vf[i]
            if (xf[i] < 0.0 || xf[i] > Lx || yf[i] < 0.0 || yf[i] > Ly)
                println("Front out of bounds at l = ", i, " x = ", xf[i], " y = ", yf[i])
            end
        end
        
        xf[1] = xf[Nf+1]
        yf[1] = yf[Nf+1]
        xf[Nf+2] = xf[2]
        yf[Nf+2] = yf[2]
        
        #------------ distribute gradient --------------
        fx = zeros(nx+2, ny+2)
        fy = zeros(nx+2, ny+2)  # Set fx & fy to zero
        for l=2:Nf+1
            nfx = -0.5 * (yf[l+1] - yf[l-1]) * (rho2 - rho1)
            nfy = 0.5 * (xf[l+1] - xf[l-1]) * (rho2 - rho1)  # Normal vector

            ip = Int(floor(xf[l] / dx)) + 1
            jp = Int(floor((yf[l] + 0.5 * dy) / dy)) + 1
            ax = xf[l] / dx - ip + 1
            ay = (yf[l] + 0.5 * dy) / dy - jp + 1
            fx[ip,jp] += (1.0 - ax) * (1.0 - ay) * nfx / dx / dy
            fx[ip+1,jp] += ax * (1.0 - ay) * nfx / dx / dy
            fx[ip,jp+1] += (1.0 - ax) * ay * nfx / dx / dy
            fx[ip+1,jp+1] += ax * ay * nfx / dx / dy

            ip = Int(floor((xf[l] + 0.5 * dx) / dx)) + 1
            jp = Int(floor(yf[l] / dy)) + 1
            ax = (xf[l] + 0.5 * dx) / dx - ip + 1
            ay = yf[l] / dy - jp + 1
            fy[ip,jp] += (1.0 - ax) * (1.0 - ay) * nfy / dx / dy
            fy[ip+1,jp] += ax * (1.0 - ay) * nfy / dx / dy
            fy[ip,jp+1] += (1.0 - ax) * ay * nfy / dx / dy
            fy[ip+1,jp+1] += ax * ay * nfy / dx / dy
        end

        #------------ construct the density --------------
        for iter=1:maxit
            oldArray = copy(r)
            for i=2:nx+1
                for j=2:ny+1
                    r[i,j] = (1.0 - beta) * r[i,j] + 
                            beta * (0.25 * (r[i+1,j] + r[i-1,j] + r[i,j+1] + r[i,j-1] 
                            + dx * fx[i-1,j] - dx * fx[i,j] + dy * fy[i,j-1] - dy * fy[i,j]))
                end
            end
            if maximum(abs.(oldArray .- r)) < maxError
                break
            end
        end

        #------------ update the viscosity --------------
        m = zeros(nx+2, ny+2) .+ m1
        for i=2:nx+1
            for j=2:ny+1
                m[i,j] = m1 + (m2 - m1) * (r[i,j] - rho1) / (rho2 - rho1)
            end
        end 

    end # end the time iteration---second order   

    u = 0.5 * (u + un)
    v = 0.5 * (v + vn)
    r = 0.5 * (r + rn)
    m = 0.5 * (m + mn)  # second order
    xf = 0.5 * (xf + xfn)
    yf = 0.5 * (yf + yfn)   # second order

    # convert to vectors so we can resize if necessary
    xf = vec(xf)
    yf = vec(yf)
    #------------ Add points to the front ------------
    xfold = copy(xf)
    yfold = copy(yf)
    j = 1
    for l=2:Nf+1
        ds = sqrt(((xfold[l] - xf[j]) / dx)^2 + ((yfold[l] - yf[j]) / dy)^2)
        if ds > 0.5
            j += 1
            if j >= size(xf,1)
                println("Too many points added to the front, increase Nf")
                resize!(xf, 2 * Nf)
                resize!(yf, 2 * Nf)
            end
            xf[j] = 0.5 * (xfold[l] + xf[j-1])
            yf[j] = 0.5 * (yfold[l] + yf[j-1])
            j += 1
            if j >= size(xf,1)
                println("Too many points added to the front, increase Nf")
                resize!(xf, 2 * Nf)
                resize!(yf, 2 * Nf)
            end
            xf[j] = xfold[l]
            yf[j] = yfold[l]
        elseif ds < 0.25
            # DO NOTHING!
        else
            j += 1
            if j >= size(xf,1)
                println("Too many points added to the front, increase Nf")
                resize!(xf, 2 * Nf)
                resize!(yf, 2 * Nf)
            end
            xf[j] = xfold[l]
            yf[j] = yfold[l]
        end    
    end
    Nf = j - 1
    xf[1] = xf[Nf+1]
    yf[1] = yf[Nf+1]
    xf[Nf+2] = xf[2]
    yf[Nf+2] = yf[2]

    # plot the results
    time += dt
    uu = 0.5 * (u[1:nx+1, 2:ny+2] + u[1:nx+1, 1:ny+1])
    vv = 0.5 * (v[2:nx+2, 1:ny+1] + v[1:nx+1, 1:ny+1])
    xh = [dx * (i - 1) for i=1:nx+1]
    yh = [dy * (j - 1) for j=1:ny+1]

    if is % 10 == 0
        println("time = ", time)
        println("max(u) = ", maximum(u))
        println("max(v) = ", maximum(v))
        s = 1
        fig = Figure()
        ax = Axis(fig[1, 1])
        CairoMakie.contour!(ax, Array(x), Array(y), Array(r), levels=20)
        CairoMakie.arrows!(ax, xh[1:s:end], yh[1:s:end], Array(uu)[1:s:end,1:s:end], Array(vv)[1:s:end,1:s:end], 
            arrowsize=0.03, lengthscale=0.02, color=:red)
        limits!(ax, 0, Lx, 0, Ly)
        save("out_vis/step_$is.png", fig)
    end

end
