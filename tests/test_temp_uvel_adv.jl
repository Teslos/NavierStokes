using CUDA
using Test
using BenchmarkTools

# GPU kernel (as provided)
function temp_uvel_adv!(ut, u, v, r, dt, dx, dy, fx, gx, rro)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    if (i > 1 && i <= size(ut,1)-1 && j > 1 && j <= size(ut,2)-1)
            ut[i, j] = u[i, j] + dt * (-0.25 * 
            (((u[i+1, j] + u[i, j])^2 - (u[i, j] + u[i-1, j])^2) / dx 
            + ((u[i, j+1] + u[i, j]) * (v[i+1, j] + v[i, j]) 
            - (u[i, j] + u[i, j-1]) * (v[i+1, j-1] + v[i, j-1])) / dy) 
            + fx[i, j] / (0.5 * (r[i+1, j] + r[i, j])) 
            - (1.0 - rro / (0.5 * (r[i+1, j] + r[i, j]))) * gx)
    end
    return
end

# CPU version of the same function
function temp_uvel_adv_cpu!(ut, u, v, r, dt, dx, dy, fx, gx, rro)
    Threads.@threads for i in 2:size(ut,1)-1
        for j in 2:size(ut,2)-1
            ut[i, j] = u[i, j] + dt * (-0.25 * 
            (((u[i+1, j] + u[i, j])^2 - (u[i, j] + u[i-1, j])^2) / dx 
            + ((u[i, j+1] + u[i, j]) * (v[i+1, j] + v[i, j]) 
            - (u[i, j] + u[i, j-1]) * (v[i+1, j-1] + v[i, j-1])) / dy) 
            + fx[i, j] / (0.5 * (r[i+1, j] + r[i, j])) 
            - (1.0 - rro / (0.5 * (r[i+1, j] + r[i, j]))) * gx)
        end
    end
    return
end

# Function to launch the GPU kernel
function launch_temp_uvel_adv!(ut, u, v, r, dt, dx, dy, fx, gx, rro)
    nx, ny = size(ut)
    threads = (16, 16)
    blocks = (cld(nx, threads[1]), cld(ny, threads[2]))
    
    @cuda blocks=blocks threads=threads temp_uvel_adv!(ut, u, v, r, dt, dx, dy, fx, gx, rro)
    return
end

function test_temp_uvel_adv()
    # Test parameters
    nx, ny = 32, 32
    dt = 0.01
    dx, dy = 0.1, 0.1
    gx = 9.81
    rro = 1000.0  # Reference density
    
    # Initialize arrays
    u = ones(Float32, nx+1, ny+2)
    v = ones(Float32, nx+2, ny+1)
    r = ones(Float32, nx+2, ny+2) * rro  # Density
    fx = zeros(Float32, nx+2, ny+2)      # Force in x direction
    
    # Add some variation to density and force
    for i in 1:nx, j in 1:ny
        if (i-nx/2)^2 + (j-ny/2)^2 < (nx/4)^2
            r[i,j] = 1.1 * rro
            fx[i,j] = 0.1
        end
    end
    
    # CPU arrays
    ut_cpu = similar(u)
    
    # Run CPU version
    temp_uvel_adv_cpu!(ut_cpu, u, v, r, dt, dx, dy, fx, gx, rro)
    # check for NaN values
    if any(isnan, ut_cpu)
        println("NaN values found in CPU result")
    end
    # print the result
    println("CPU result:")
    println(ut_cpu)
    
    # Only run GPU tests if CUDA is available
    if CUDA.functional()
        # GPU arrays
        u_gpu = CuArray(u)
        v_gpu = CuArray(v)
        r_gpu = CuArray(r)
        fx_gpu = CuArray(fx)
        ut_gpu = similar(u_gpu)
        
        # Run GPU version
        launch_temp_uvel_adv!(ut_gpu, u_gpu, v_gpu, r_gpu, dt, dx, dy, fx_gpu, gx, rro)
        
        # Transfer back to CPU for comparison
        ut_from_gpu = Array(ut_gpu)
        # check for NaN values
        if any(isnan, ut_from_gpu)
            println("NaN values found in GPU result")
        end
        # check for NaN values
        if any(isnan, ut_cpu)
            println("NaN values found in CPU result")
        end
        # Test GPU result against CPU result
        @test isapprox(ut_from_gpu, ut_cpu, rtol=1e-5)
        
        # Benchmark
        println("CPU performance:")
        @btime temp_uvel_adv_cpu!($ut_cpu, $u, $v, $r, $dt, $dx, $dy, $fx, $gx, $rro)
        
        println("GPU performance:")
        @btime CUDA.@sync launch_temp_uvel_adv!($ut_gpu, $u_gpu, $v_gpu, $r_gpu, $dt, $dx, $dy, $fx_gpu, $gx, $rro)
    else
        println("CUDA not available, skipping GPU tests")
    end
end

# Run the test
test_temp_uvel_adv()