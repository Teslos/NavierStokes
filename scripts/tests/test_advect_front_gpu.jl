using CUDA
using Test
using BenchmarkTools
using Random

# CPU version (as provided)
function advect_front!(uf, vf, xf, yf, u, v, dx, dy, Nf)
    # advect the front
    for l=2:Nf+1
        ip = Int(floor(xf[l] / dx)) + 1
        jp = Int(floor((yf[l] + 0.5 * dy) / dy)) + 1
        ax = xf[l] / dx - ip + 1
        ay = (yf[l] + 0.5 * dy) / dy - jp + 1

        v1 = (1.0-ax)*(1.0-ay)*u[ip,jp]
        v2 = ax*(1.0-ay)*u[ip+1,jp]
        v3 = (1.0-ax)*ay*u[ip,jp+1]
        v4 = ax*ay*u[ip+1,jp+1]
        uf[l] = v1+v2+v3+v4

        ip = Int(floor((xf[l] + 0.5 * dx) / dx)) + 1
        jp = Int(floor(yf[l] / dy)) + 1
        ax = (xf[l] + 0.5 * dx) / dx - ip + 1
        ay = yf[l] / dy - jp + 1
        
        v1 = (1.0-ax)*(1.0-ay)*v[ip,jp]
        v2 = ax*(1.0-ay)*v[ip+1,jp]
        v3 = (1.0-ax)*ay*v[ip,jp+1]
        v4 = ax*ay*v[ip+1,jp+1]
        vf[l] = v1+v2+v3+v4
    end
    return
end

# GPU kernel version
function advect_front_kernel!(uf, vf, xf, yf, u, v, dx, dy, Nf)
    # Get thread ID (each thread processes one front point)
    l = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if l >= 2 && l <= Nf+1
        # For u velocity
        ip = Int(floor(xf[l] / dx)) + 1
        jp = Int(floor((yf[l] + 0.5 * dy) / dy)) + 1
        ax = xf[l] / dx - ip + 1
        ay = (yf[l] + 0.5 * dy) / dy - jp + 1

        v1 = (1.0-ax)*(1.0-ay)*u[ip,jp]
        v2 = ax*(1.0-ay)*u[ip+1,jp]
        v3 = (1.0-ax)*ay*u[ip,jp+1]
        v4 = ax*ay*u[ip+1,jp+1]
        uf[l] = v1+v2+v3+v4

        # For v velocity
        ip = Int(floor((xf[l] + 0.5 * dx) / dx)) + 1
        jp = Int(floor(yf[l] / dy)) + 1
        ax = (xf[l] + 0.5 * dx) / dx - ip + 1
        ay = yf[l] / dy - jp + 1
        
        v1 = (1.0-ax)*(1.0-ay)*v[ip,jp]
        v2 = ax*(1.0-ay)*v[ip+1,jp]
        v3 = (1.0-ax)*ay*v[ip,jp+1]
        v4 = ax*ay*v[ip+1,jp+1]
        vf[l] = v1+v2+v3+v4
    end
    return
end

# Function to launch the GPU kernel
function advect_front_gpu!(uf, vf, xf, yf, u, v, dx, dy, Nf)
    # Configure kernel launch parameters
    threads_per_block = 256
    num_blocks = cld(Nf, threads_per_block)
    
    @cuda blocks=num_blocks threads=threads_per_block advect_front_kernel!(uf, vf, xf, yf, u, v, dx, dy, Nf)
    return
end

function test_advect_front()
    # Test parameters
    nx, ny = 128, 128
    dx, dy = 0.1, 0.1
    Nf = 1000  # Number of front points
    
    # Set random seed for reproducibility
    Random.seed!(12345)
    
    # Initialize velocity fields with a vortex pattern
    u = zeros(Float32, nx, ny)
    v = zeros(Float32, nx, ny)
    
    center_x, center_y = nx÷2, ny÷2
    for i in 1:nx, j in 1:ny
        # Distance from center
        dx_from_center = (i - center_x) * dx
        dy_from_center = (j - center_y) * dy
        r = sqrt(dx_from_center^2 + dy_from_center^2)
        
        if r > 0.1
            # Vortex pattern
            theta = atan(dy_from_center, dx_from_center)
            strength = 1.0 * exp(-r/2.0)
            u[i,j] = -strength * sin(theta)
            v[i,j] = strength * cos(theta)
        end
    end
    
    # Initialize front points in a circle
    xf = zeros(Float32, Nf+1)
    yf = zeros(Float32, Nf+1)
    
    radius = 3.0
    for l in 2:Nf+1
        angle = 2π * (l-2) / (Nf-1)
        xf[l] = center_x * dx + radius * cos(angle)
        yf[l] = center_y * dy + radius * sin(angle)
    end
    
    # Initialize front velocities
    uf_cpu = zeros(Float32, Nf+1)
    vf_cpu = zeros(Float32, Nf+1)
    
    # Run CPU version
    advect_front!(uf_cpu, vf_cpu, xf, yf, u, v, dx, dy, Nf)
    
    # Only run GPU tests if CUDA is available
    if CUDA.functional()
        # GPU arrays
        uf_gpu = zeros(Float32, Nf+1)
        vf_gpu = zeros(Float32, Nf+1)
        
        xf_gpu = CuArray(xf)
        yf_gpu = CuArray(yf)
        u_gpu = CuArray(u)
        v_gpu = CuArray(v)
        uf_gpu_d = CuArray(uf_gpu)
        vf_gpu_d = CuArray(vf_gpu)
        
        # Run GPU version
        advect_front_gpu!(uf_gpu_d, vf_gpu_d, xf_gpu, yf_gpu, u_gpu, v_gpu, dx, dy, Nf)
        
        # Transfer back to CPU for comparison
        uf_gpu = Array(uf_gpu_d)
        vf_gpu = Array(vf_gpu_d)
        
        # Test GPU result against CPU result
        @test isapprox(uf_gpu, uf_cpu, rtol=1e-5)
        @test isapprox(vf_gpu, vf_cpu, rtol=1e-5)
        
        # Benchmark
        println("CPU performance:")
        @btime advect_front!($uf_cpu, $vf_cpu, $xf, $yf, $u, $v, $dx, $dy, $Nf)
        
        println("GPU performance:")
        @btime CUDA.@sync advect_front_gpu!($uf_gpu_d, $vf_gpu_d, $xf_gpu, $yf_gpu, $u_gpu, $v_gpu, $dx, $dy, $Nf)
        
        # Calculate speedup
        cpu_time = @benchmark advect_front!($uf_cpu, $vf_cpu, $xf, $yf, $u, $v, $dx, $dy, $Nf)
        gpu_time = @benchmark CUDA.@sync advect_front_gpu!($uf_gpu_d, $vf_gpu_d, $xf_gpu, $yf_gpu, $u_gpu, $v_gpu, $dx, $dy, $Nf)
        
        speedup = median(cpu_time).time / median(gpu_time).time
        println("GPU speedup: $(speedup)x")
        
        # Visualize differences (optional)
        max_diff_u = maximum(abs.(uf_gpu - uf_cpu))
        max_diff_v = maximum(abs.(vf_gpu - vf_cpu))
        println("Maximum absolute difference (uf): $max_diff_u")
        println("Maximum absolute difference (vf): $max_diff_v")
        
        # Test with larger front for better performance comparison
        if Nf < 10000
            println("\nTesting with larger front (10000 points):")
            Nf_large = 10000
            
            # Initialize larger front
            xf_large = zeros(Float32, Nf_large+1)
            yf_large = zeros(Float32, Nf_large+1)
            
            for l in 2:Nf_large+1
                angle = 2π * (l-2) / (Nf_large-1)
                xf_large[l] = center_x * dx + radius * cos(angle)
                yf_large[l] = center_y * dy + radius * sin(angle)
            end
            
            # CPU arrays
            uf_large_cpu = zeros(Float32, Nf_large+1)
            vf_large_cpu = zeros(Float32, Nf_large+1)
            
            # GPU arrays
            xf_large_gpu = CuArray(xf_large)
            yf_large_gpu = CuArray(yf_large)
            uf_large_gpu = CuArray(zeros(Float32, Nf_large+1))
            vf_large_gpu = CuArray(zeros(Float32, Nf_large+1))
            
            # Benchmark larger front
            println("CPU performance (10000 points):")
            @btime advect_front!($uf_large_cpu, $vf_large_cpu, $xf_large, $yf_large, $u, $v, $dx, $dy, $Nf_large)
            
            println("GPU performance (10000 points):")
            @btime CUDA.@sync advect_front_gpu!($uf_large_gpu, $vf_large_gpu, $xf_large_gpu, $yf_large_gpu, $u_gpu, $v_gpu, $dx, $dy, $Nf_large)
            
            # Calculate speedup for larger front
            cpu_time_large = @benchmark advect_front!($uf_large_cpu, $vf_large_cpu, $xf_large, $yf_large, $u, $v, $dx, $dy, $Nf_large)
            gpu_time_large = @benchmark CUDA.@sync advect_front_gpu!($uf_large_gpu, $vf_large_gpu, $xf_large_gpu, $yf_large_gpu, $u_gpu, $v_gpu, $dx, $dy, $Nf_large)
            
            speedup_large = median(cpu_time_large).time / median(gpu_time_large).time
            println("GPU speedup (10000 points): $(speedup_large)x")
        end
    else
        println("CUDA not available, skipping GPU tests")
    end
end

# Run the test
test_advect_front()