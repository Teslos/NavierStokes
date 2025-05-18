using CUDA
using Test
using BenchmarkTools

# CPU version (as provided)
function temp_vvel_adv!(vt, u, v, r, dt, dx, dy, fy, gy, rro)
    Threads.@threads for i in 2:size(vt,1)-1
        for j in 2:size(vt,2)-1
            vt[i, j] = v[i, j] + dt * (-0.25 * 
            (((u[i, j+1] + u[i, j]) * (v[i+1, j] + v[i, j]) 
            - (u[i-1, j+1] + u[i-1, j]) * (v[i, j] + v[i-1, j])) / dx 
            + ((v[i, j+1] + v[i, j])^2 - (v[i, j] + v[i, j-1])^2) / dy) 
            + fy[i, j] / (0.5 * (r[i, j+1] + r[i, j])) 
            - (1.0 - rro / (0.5 * (r[i, j+1] + r[i, j]))) * gy)
        end
    end
    return
end

# GPU kernel version
function temp_vvel_adv_kernel!(vt, u, v, r, dt, dx, dy, fy, gy, rro)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    
    if (i > 1 && i <= size(vt,1)-1 && j > 1 && j <= size(vt,2)-1)
        vt[i, j] = v[i, j] + dt * (-0.25 * 
        (((u[i, j+1] + u[i, j]) * (v[i+1, j] + v[i, j]) 
        - (u[i-1, j+1] + u[i-1, j]) * (v[i, j] + v[i-1, j])) / dx 
        + ((v[i, j+1] + v[i, j])^2 - (v[i, j] + v[i, j-1])^2) / dy) 
        + fy[i, j] / (0.5 * (r[i, j+1] + r[i, j])) 
        - (1.0 - rro / (0.5 * (r[i, j+1] + r[i, j]))) * gy)
    end
    return
end

# Function to launch the GPU kernel
function temp_vvel_adv_gpu!(vt, u, v, r, dt, dx, dy, fy, gy, rro)
    nx, ny = size(vt)
    threads = (16, 16)
    blocks = (cld(nx, threads[1]), cld(ny, threads[2]))
    
    @cuda blocks=blocks threads=threads temp_vvel_adv_kernel!(vt, u, v, r, dt, dx, dy, fy, gy, rro)
    return
end

function test_temp_vvel_adv()
    # Test parameters
    nx, ny = 128, 128
    dt = 0.01
    dx, dy = 0.1, 0.1
    gy = 9.81
    rro = 1000.0  # Reference density
    
    # Initialize arrays with realistic fluid simulation data
    u = zeros(Float32, nx, ny)
    v = zeros(Float32, nx, ny)
    r = ones(Float32, nx, ny) * rro  # Density
    fy = zeros(Float32, nx, ny)      # Force in y direction
    
    # Add a circular region with different density and velocity
    center_x, center_y = nx÷2, ny÷2
    radius = nx÷4
    
    for i in 1:nx, j in 1:ny
        dist = sqrt((i-center_x)^2 + (j-center_y)^2)
        if dist < radius
            # Inside the circle: higher density and some force
            r[i,j] = 1.1 * rro
            fy[i,j] = 0.1
            
            # Add a vortex-like velocity field
            angle = atan(j-center_y, i-center_x)
            u[i,j] = -0.5 * sin(angle) * (1.0 - dist/radius)
            v[i,j] = 0.5 * cos(angle) * (1.0 - dist/radius)
        else
            # Outside: small random perturbations
            u[i,j] = 0.05 * randn()
            v[i,j] = 0.05 * randn()
        end
    end
    
    # CPU arrays
    vt_cpu = similar(v)
    fill!(vt_cpu, 0.0)
    
    # Run CPU version
    temp_vvel_adv!(vt_cpu, u, v, r, dt, dx, dy, fy, gy, rro)
    
    # Only run GPU tests if CUDA is available
    if CUDA.functional()
        # GPU arrays
        u_gpu = CuArray(u)
        v_gpu = CuArray(v)
        r_gpu = CuArray(r)
        fy_gpu = CuArray(fy)
        vt_gpu = CuArray(zeros(Float32, nx, ny))
        
        # Run GPU version
        temp_vvel_adv_gpu!(vt_gpu, u_gpu, v_gpu, r_gpu, dt, dx, dy, fy_gpu, gy, rro)
        
        # Transfer back to CPU for comparison
        vt_from_gpu = Array(vt_gpu)
        
        # Test GPU result against CPU result
        @test isapprox(vt_from_gpu, vt_cpu, rtol=1e-5)
        
        # Benchmark
        println("CPU performance:")
        @btime temp_vvel_adv!($vt_cpu, $u, $v, $r, $dt, $dx, $dy, $fy, $gy, $rro)
        
        println("GPU performance:")
        @btime CUDA.@sync temp_vvel_adv_gpu!($vt_gpu, $u_gpu, $v_gpu, $r_gpu, $dt, $dx, $dy, $fy_gpu, $gy, $rro)
        
        # Calculate speedup
        cpu_time = @benchmark temp_vvel_adv!($vt_cpu, $u, $v, $r, $dt, $dx, $dy, $fy, $gy, $rro)
        gpu_time = @benchmark CUDA.@sync temp_vvel_adv_gpu!($vt_gpu, $u_gpu, $v_gpu, $r_gpu, $dt, $dx, $dy, $fy_gpu, $gy, $rro)
        
        speedup = median(cpu_time).time / median(gpu_time).time
        println("GPU speedup: $(speedup)x")
        
        # Visualize differences (optional)
        max_diff = maximum(abs.(vt_from_gpu - vt_cpu))
        println("Maximum absolute difference: $max_diff")
    else
        println("CUDA not available, skipping GPU tests")
    end
end

# Run the test
test_temp_vvel_adv()