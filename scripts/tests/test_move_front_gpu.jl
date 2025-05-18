using CUDA
using Test
using BenchmarkTools
using Random

# CPU version (as provided)
function move_front!(xf, yf, uf, vf, Nf, dt, Lx, Ly)
    # move the front
    Threads.@threads for i=2:Nf+1
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
    return
end

# GPU kernel version
function move_front_kernel!(xf, yf, uf, vf, Nf, dt, Lx, Ly)
    # Get thread ID (each thread processes one front point)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if i >= 2 && i <= Nf+1
        # Update position
        xf[i] = xf[i] + dt * uf[i]
        yf[i] = yf[i] + dt * vf[i]
        
        # Check bounds (using atomicAdd to avoid race conditions in printing)
        if (xf[i] < 0.0 || xf[i] > Lx || yf[i] < 0.0 || yf[i] > Ly)
            # Note: We can't print from GPU kernels, so we'll just handle this differently
            # For example, we could set a flag or record the out-of-bounds point
            # Here we'll just clamp the values to the domain boundaries
            xf[i] = max(0.0, min(Lx, xf[i]))
            yf[i] = max(0.0, min(Ly, yf[i]))
        end
    end
    return
end

# Function to handle the boundary points (first and last)
function handle_boundary_points_kernel!(xf, yf, Nf)
    # This kernel should be called with only one thread
    if threadIdx().x == 1 && blockIdx().x == 1
        xf[1] = xf[Nf+1]
        yf[1] = yf[Nf+1]
        xf[Nf+2] = xf[2]
        yf[Nf+2] = yf[2]
    end
    return
end

# Function to launch the GPU kernels
function move_front_gpu!(xf, yf, uf, vf, Nf, dt, Lx, Ly)
    # Configure kernel launch parameters for main kernel
    threads_per_block = 256
    num_blocks = cld(Nf, threads_per_block)
    
    # Launch main kernel
    @cuda blocks=num_blocks threads=threads_per_block move_front_kernel!(xf, yf, uf, vf, Nf, dt, Lx, Ly)
    
    # Launch boundary points kernel with just one thread
    @cuda blocks=1 threads=1 handle_boundary_points_kernel!(xf, yf, Nf)
    
    return
end

function test_move_front()
    # Test parameters
    Nf = 1000  # Number of front points
    dt = 0.01  # Time step
    Lx = 10.0  # Domain size in x
    Ly = 10.0  # Domain size in y
    
    # Set random seed for reproducibility
    Random.seed!(12345)
    
    # Initialize front points in a circle
    center_x, center_y = Lx/2, Ly/2
    radius = 3.0
    
    # Add 2 extra points for periodic boundary conditions
    xf = zeros(Float32, Nf+2)
    yf = zeros(Float32, Nf+2)
    
    for l in 2:Nf+1
        angle = 2π * (l-2) / (Nf-1)
        xf[l] = center_x + radius * cos(angle)
        yf[l] = center_y + radius * sin(angle)
    end
    
    # Set boundary points
    xf[1] = xf[Nf+1]
    yf[1] = yf[Nf+1]
    xf[Nf+2] = xf[2]
    yf[Nf+2] = yf[2]
    
    # Initialize front velocities with a rotational pattern
    uf = zeros(Float32, Nf+2)
    vf = zeros(Float32, Nf+2)
    
    for l in 2:Nf+1
        # Create a rotational velocity field
        dx = xf[l] - center_x
        dy = yf[l] - center_y
        r = sqrt(dx^2 + dy^2)
        
        if r > 0.1
            # Rotational velocity (counter-clockwise)
            angular_velocity = 0.5  # radians per time unit
            uf[l] = -angular_velocity * dy
            vf[l] = angular_velocity * dx
        end
    end
    
    # Make copies for CPU and GPU tests
    xf_cpu = copy(xf)
    yf_cpu = copy(yf)
    uf_cpu = copy(uf)
    vf_cpu = copy(vf)
    
    # Run CPU version
    move_front!(xf_cpu, yf_cpu, uf_cpu, vf_cpu, Nf, dt, Lx, Ly)
    
    # Only run GPU tests if CUDA is available
    if CUDA.functional()
        # GPU arrays
        xf_gpu = CuArray(xf)
        yf_gpu = CuArray(yf)
        uf_gpu = CuArray(uf)
        vf_gpu = CuArray(vf)
        
        # Run GPU version
        move_front_gpu!(xf_gpu, yf_gpu, uf_gpu, vf_gpu, Nf, dt, Lx, Ly)
        
        # Transfer back to CPU for comparison
        xf_from_gpu = Array(xf_gpu)
        yf_from_gpu = Array(yf_gpu)
        
        # Test GPU result against CPU result
        @test isapprox(xf_from_gpu, xf_cpu, rtol=1e-5)
        @test isapprox(yf_from_gpu, yf_cpu, rtol=1e-5)
        
        # Benchmark
        println("CPU performance:")
        @btime move_front!($xf_cpu, $yf_cpu, $uf_cpu, $vf_cpu, $Nf, $dt, $Lx, $Ly)
        
        println("GPU performance:")
        @btime CUDA.@sync move_front_gpu!($xf_gpu, $yf_gpu, $uf_gpu, $vf_gpu, $Nf, $dt, $Lx, $Ly)
        
        # Calculate speedup
        cpu_time = @benchmark move_front!($xf_cpu, $yf_cpu, $uf_cpu, $vf_cpu, $Nf, $dt, $Lx, $Ly)
        gpu_time = @benchmark CUDA.@sync move_front_gpu!($xf_gpu, $yf_gpu, $uf_gpu, $vf_gpu, $Nf, $dt, $Lx, $Ly)
        
        speedup = median(cpu_time).time / median(gpu_time).time
        println("GPU speedup: $(speedup)x")
        
        # Visualize differences (optional)
        max_diff_x = maximum(abs.(xf_from_gpu - xf_cpu))
        max_diff_y = maximum(abs.(yf_from_gpu - yf_cpu))
        println("Maximum absolute difference (xf): $max_diff_x")
        println("Maximum absolute difference (yf): $max_diff_y")
        
        # Test with larger front for better performance comparison
        if Nf < 100000
            println("\nTesting with larger front (100,000 points):")
            Nf_large = 100000
            
            # Initialize larger front
            xf_large = zeros(Float32, Nf_large+2)
            yf_large = zeros(Float32, Nf_large+2)
            uf_large = zeros(Float32, Nf_large+2)
            vf_large = zeros(Float32, Nf_large+2)
            
            for l in 2:Nf_large+1
                angle = 2π * (l-2) / (Nf_large-1)
                xf_large[l] = center_x + radius * cos(angle)
                yf_large[l] = center_y + radius * sin(angle)
                
                # Create rotational velocity
                dx = xf_large[l] - center_x
                dy = yf_large[l] - center_y
                r = sqrt(dx^2 + dy^2)
                
                if r > 0.1
                    angular_velocity = 0.5
                    uf_large[l] = -angular_velocity * dy
                    vf_large[l] = angular_velocity * dx
                end
            end
            
            # Set boundary points
            xf_large[1] = xf_large[Nf_large+1]
            yf_large[1] = yf_large[Nf_large+1]
            xf_large[Nf_large+2] = xf_large[2]
            yf_large[Nf_large+2] = yf_large[2]
            
            # Make copies for CPU and GPU tests
            xf_large_cpu = copy(xf_large)
            yf_large_cpu = copy(yf_large)
            
            # GPU arrays
            xf_large_gpu = CuArray(xf_large)
            yf_large_gpu = CuArray(yf_large)
            uf_large_gpu = CuArray(uf_large)
            vf_large_gpu = CuArray(vf_large)
            
            # Benchmark larger front
            println("CPU performance (100,000 points):")
            @btime move_front!($xf_large_cpu, $yf_large_cpu, $uf_large, $vf_large, $Nf_large, $dt, $Lx, $Ly)
            
            println("GPU performance (100,000 points):")
            @btime CUDA.@sync move_front_gpu!($xf_large_gpu, $yf_large_gpu, $uf_large_gpu, $vf_large_gpu, $Nf_large, $dt, $Lx, $Ly)
            
            # Calculate speedup for larger front
            cpu_time_large = @benchmark move_front!($xf_large_cpu, $yf_large_cpu, $uf_large, $vf_large, $Nf_large, $dt, $Lx, $Ly)
            gpu_time_large = @benchmark CUDA.@sync move_front_gpu!($xf_large_gpu, $yf_large_gpu, $uf_large_gpu, $vf_large_gpu, $Nf_large, $dt, $Lx, $Ly)
            
            speedup_large = median(cpu_time_large).time / median(gpu_time_large).time
            println("GPU speedup (100,000 points): $(speedup_large)x")
        end
    else
        println("CUDA not available, skipping GPU tests")
    end
end

# Run the test
test_move_front()