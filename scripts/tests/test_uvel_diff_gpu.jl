using CUDA
using Test
using BenchmarkTools

# CPU version (as provided)
function temp_uvel_diff!(ut, u, v, r, dt, dx, dy, m)
    Threads.@threads for i in 2:size(ut,1)-1
        for j in 2:size(ut,2)-1
            ut[i, j] = ut[i, j] + dt * ((1. / dx) * 2.0 
            * (m[i+1, j] * (1. / dx) * (u[i+1, j] - u[i, j]) 
            - m[i, j] * (1. / dx) * (u[i, j] - u[i-1, j])) 
            + (1. / dy) * (0.25 * (m[i, j] + m[i+1, j] + m[i+1, j+1] + m[i, j+1]) * ((1. / dy) * (u[i, j+1] - u[i, j]) 
            + (1. / dx) * (v[i+1, j] - v[i, j])) - 
            0.25 * (m[i, j] + m[i+1, j] + m[i+1, j-1] + m[i, j-1]) * ((1. / dy) * (u[i, j] - u[i, j-1])
            + (1. / dx) * (v[i+1, j-1] - v[i, j-1]))) / (0.5 * (r[i+1, j] + r[i, j])))
        end
    end
    return
end

# GPU kernel version
function temp_uvel_diff_kernel!(ut, u, v, r, dt, dx, dy, m)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    
    if (i > 1 && i <= size(ut,1)-1 && j > 1 && j <= size(ut,2)-1)
        ut[i, j] = ut[i, j] + dt * ((1. / dx) * 2. 
        * (m[i+1, j] * (1. / dx) * (u[i+1, j] - u[i, j]) 
        - m[i, j] * (1. / dx) * (u[i, j] - u[i-1, j])) 
        + (1. / dy) * (0.25 * (m[i, j] + m[i+1, j] + m[i+1, j+1] + m[i, j+1]) * ((1. / dy) * (u[i, j+1] - u[i, j]) 
        + (1. / dx) * (v[i+1, j] - v[i, j])) - 
        0.25 * (m[i, j] + m[i+1, j] + m[i+1, j-1] + m[i, j-1]) * ((1. / dy) * (u[i, j] - u[i, j-1])
        + (1. / dx) * (v[i+1, j-1] - v[i, j-1]))) / (0.5 * (r[i+1, j] + r[i, j])))
    end
    return
end

# Function to launch the GPU kernel
function temp_uvel_diff_gpu!(ut, u, v, r, dt, dx, dy, m)
    nx, ny = size(ut)
    threads = (16, 16)
    blocks = (cld(nx, threads[1]), cld(ny, threads[2]))
    
    @cuda blocks=blocks threads=threads temp_uvel_diff_kernel!(ut, u, v, r, dt, dx, dy, m)
    return
end

function test_temp_uvel_diff()
    # Test parameters
    nx, ny = 128, 128
    dt = 0.01
    dx, dy = 0.1, 0.1
    
    # Initialize arrays with realistic fluid simulation data
    u = zeros(Float32, nx, ny)
    v = zeros(Float32, nx, ny)
    r = ones(Float32, nx, ny) * 1000.0  # Density
    m = ones(Float32, nx, ny) * 0.001   # Viscosity
    
    # Add a circular region with different properties
    center_x, center_y = nx÷2, ny÷2
    radius = nx÷4
    
    for i in 1:nx, j in 1:ny
        dist = sqrt((i-center_x)^2 + (j-center_y)^2)
        if dist < radius
            # Inside the circle: higher viscosity
            m[i,j] = 0.01
            
            # Add a shear layer velocity field
            u[i,j] = tanh((j - center_y) / 10.0)
            v[i,j] = 0.05 * sin(2π * (i - center_x) / radius)
        else
            # Outside: small random perturbations
            u[i,j] = 0.05 * randn()
            v[i,j] = 0.05 * randn()
        end
    end
    
    # Create initial ut arrays with some values (since the function adds to ut)
    ut_cpu = copy(u)
    
    # Run CPU version
    temp_uvel_diff!(ut_cpu, u, v, r, dt, dx, dy, m)
    
    # Only run GPU tests if CUDA is available
    if CUDA.functional()
        # GPU arrays
        u_gpu = CuArray(u)
        v_gpu = CuArray(v)
        r_gpu = CuArray(r)
        m_gpu = CuArray(m)
        ut_gpu = CuArray(copy(u))
        
        # Run GPU version
        temp_uvel_diff_gpu!(ut_gpu, u_gpu, v_gpu, r_gpu, dt, dx, dy, m_gpu)
        
        # Transfer back to CPU for comparison
        ut_from_gpu = Array(ut_gpu)
        
        # Test GPU result against CPU result
        @test isapprox(ut_from_gpu, ut_cpu, rtol=1e-5)
        
        # Benchmark
        println("CPU performance:")
        @btime temp_uvel_diff!($ut_cpu, $u, $v, $r, $dt, $dx, $dy, $m)
        
        println("GPU performance:")
        @btime CUDA.@sync temp_uvel_diff_gpu!($ut_gpu, $u_gpu, $v_gpu, $r_gpu, $dt, $dx, $dy, $m_gpu)
        
        # Calculate speedup
        cpu_time = @benchmark temp_uvel_diff!($ut_cpu, $u, $v, $r, $dt, $dx, $dy, $m)
        gpu_time = @benchmark CUDA.@sync temp_uvel_diff_gpu!($ut_gpu, $u_gpu, $v_gpu, $r_gpu, $dt, $dx, $dy, $m_gpu)
        
        speedup = median(cpu_time).time / median(gpu_time).time
        println("GPU speedup: $(speedup)x")
        
        # Visualize differences (optional)
        max_diff = maximum(abs.(ut_from_gpu - ut_cpu))
        println("Maximum absolute difference: $max_diff")
        
        # Test with larger grid for better performance comparison
        if nx < 512
            println("\nTesting with larger grid (512x512):")
            nx_large, ny_large = 512, 512
            
            # Initialize larger arrays
            u_large = zeros(Float32, nx_large, ny_large)
            v_large = zeros(Float32, nx_large, ny_large)
            r_large = ones(Float32, nx_large, ny_large) * 1000.0
            m_large = ones(Float32, nx_large, ny_large) * 0.001
            ut_large = copy(u_large)
            
            # Add pattern
            center_x, center_y = nx_large÷2, ny_large÷2
            radius = nx_large÷4
            
            for i in 1:nx_large, j in 1:ny_large
                dist = sqrt((i-center_x)^2 + (j-center_y)^2)
                if dist < radius
                    m_large[i,j] = 0.01
                    u_large[i,j] = tanh((j - center_y) / 10.0)
                    v_large[i,j] = 0.05 * sin(2π * (i - center_x) / radius)
                else
                    u_large[i,j] = 0.05 * randn()
                    v_large[i,j] = 0.05 * randn()
                end
            end
            
            # GPU arrays
            u_large_gpu = CuArray(u_large)
            v_large_gpu = CuArray(v_large)
            r_large_gpu = CuArray(r_large)
            m_large_gpu = CuArray(m_large)
            ut_large_gpu = CuArray(copy(u_large))
            
            # Benchmark larger grid
            println("CPU performance (512x512):")
            @btime temp_uvel_diff!($ut_large, $u_large, $v_large, $r_large, $dt, $dx, $dy, $m_large)
            
            println("GPU performance (512x512):")
            @btime CUDA.@sync temp_uvel_diff_gpu!($ut_large_gpu, $u_large_gpu, $v_large_gpu, $r_large_gpu, $dt, $dx, $dy, $m_large_gpu)
            
            # Calculate speedup for larger grid
            cpu_time_large = @benchmark temp_uvel_diff!($ut_large, $u_large, $v_large, $r_large, $dt, $dx, $dy, $m_large)
            gpu_time_large = @benchmark CUDA.@sync temp_uvel_diff_gpu!($ut_large_gpu, $u_large_gpu, $v_large_gpu, $r_large_gpu, $dt, $dx, $dy, $m_large_gpu)
            
            speedup_large = median(cpu_time_large).time / median(gpu_time_large).time
            println("GPU speedup (512x512): $(speedup_large)x")
        end
    else
        println("CUDA not available, skipping GPU tests")
    end
end

# Run the test
test_temp_uvel_diff()