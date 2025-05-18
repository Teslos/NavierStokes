using CUDA
using Test
using BenchmarkTools

# CPU version (as provided)
function compute_sources!(tmp1, tmp2, rt, ut, vt, dt, dx, dy, nx, ny)
    Threads.@threads for i=2:size(tmp1,1)-1
        for j=2:size(tmp1,2)-1
            tmp1[i,j] = (0.5/dt) * ((ut[i,j] - ut[i-1,j]) / dx + (vt[i,j] - vt[i,j-1]) / dy)
            tmp2[i,j] = 1.0 / ((1.0/dx) * (1.0 / (dx * (rt[i+1,j] + rt[i,j])) + 
                        1.0 / (dx * (rt[i-1,j] + rt[i,j]))) 
                        + (1.0/dy) * (1.0 / (dy * (rt[i,j+1] + rt[i,j]))
                        + 1.0 / (dy * (rt[i,j-1] + rt[i,j]))))
        end
    end
    return
end

# GPU kernel version
function compute_sources_kernel!(tmp1, tmp2, rt, ut, vt, dt, dx, dy, nx, ny)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    
    if (i > 1 && i <= size(tmp1,1)-1 && j > 1 && j <= size(tmp1,2)-1)
        tmp1[i,j] = (0.5/dt) * ((ut[i,j] - ut[i-1,j]) / dx + (vt[i,j] - vt[i,j-1]) / dy)
        tmp2[i,j] = 1.0 / ((1.0/dx) * (1.0 / (dx * (rt[i+1,j] + rt[i,j])) + 
                    1.0 / (dx * (rt[i-1,j] + rt[i,j]))) 
                    + (1.0/dy) * (1.0 / (dy * (rt[i,j+1] + rt[i,j]))
                    + 1.0 / (dy * (rt[i,j-1] + rt[i,j]))))
    end
    return
end

# Function to launch the GPU kernel
function compute_sources_gpu!(tmp1, tmp2, rt, ut, vt, dt, dx, dy, nx, ny)
    threads = (16, 16)
    blocks = (cld(nx, threads[1]), cld(ny, threads[2]))
    
    @cuda blocks=blocks threads=threads compute_sources_kernel!(tmp1, tmp2, rt, ut, vt, dt, dx, dy, nx, ny)
    return
end

function test_compute_sources()
    # Test parameters
    nx, ny = 128, 128
    dt = 0.01
    dx, dy = 0.1, 0.1
    
    # Initialize arrays with realistic fluid simulation data
    rt = ones(Float32, nx, ny) * 1000.0  # Density
    ut = zeros(Float32, nx, ny)          # x-velocity
    vt = zeros(Float32, nx, ny)          # y-velocity
    tmp1 = zeros(Float32, nx, ny)        # Divergence
    tmp2 = zeros(Float32, nx, ny)        # Pressure coefficient
    
    # Add a circular region with different properties
    center_x, center_y = nx÷2, ny÷2
    radius = nx÷4
    
    for i in 1:nx, j in 1:ny
        dist = sqrt((i-center_x)^2 + (j-center_y)^2)
        if dist < radius
            # Inside the circle: higher density
            rt[i,j] = 1100.0
            
            # Add a divergent velocity field (source/sink)
            angle = atan(j-center_y, i-center_x)
            r_norm = dist / radius
            
            if r_norm < 0.5
                # Source in the inner region
                ut[i,j] = 0.5 * cos(angle) * (0.5 - r_norm)
                vt[i,j] = 0.5 * sin(angle) * (0.5 - r_norm)
            else
                # Sink in the outer region
                ut[i,j] = -0.5 * cos(angle) * (r_norm - 0.5)
                vt[i,j] = -0.5 * sin(angle) * (r_norm - 0.5)
            end
        else
            # Outside: small random perturbations
            ut[i,j] = 0.05 * randn()
            vt[i,j] = 0.05 * randn()
        end
    end
    
    # CPU arrays
    tmp1_cpu = zeros(Float32, nx, ny)
    tmp2_cpu = zeros(Float32, nx, ny)
    
    # Run CPU version
    compute_sources!(tmp1_cpu, tmp2_cpu, rt, ut, vt, dt, dx, dy, nx, ny)
    
    # Only run GPU tests if CUDA is available
    if CUDA.functional()
        # GPU arrays
        rt_gpu = CuArray(rt)
        ut_gpu = CuArray(ut)
        vt_gpu = CuArray(vt)
        tmp1_gpu = CuArray(zeros(Float32, nx, ny))
        tmp2_gpu = CuArray(zeros(Float32, nx, ny))
        
        # Run GPU version
        compute_sources_gpu!(tmp1_gpu, tmp2_gpu, rt_gpu, ut_gpu, vt_gpu, dt, dx, dy, nx, ny)
        
        # Transfer back to CPU for comparison
        tmp1_from_gpu = Array(tmp1_gpu)
        tmp2_from_gpu = Array(tmp2_gpu)
        
        # Test GPU result against CPU result
        @test isapprox(tmp1_from_gpu, tmp1_cpu, rtol=1e-5)
        @test isapprox(tmp2_from_gpu, tmp2_cpu, rtol=1e-5)
        
        # Benchmark
        println("CPU performance:")
        @btime compute_sources!($tmp1_cpu, $tmp2_cpu, $rt, $ut, $vt, $dt, $dx, $dy, $nx, $ny)
        
        println("GPU performance:")
        @btime CUDA.@sync compute_sources_gpu!($tmp1_gpu, $tmp2_gpu, $rt_gpu, $ut_gpu, $vt_gpu, $dt, $dx, $dy, $nx, $ny)
        
        # Calculate speedup
        cpu_time = @benchmark compute_sources!($tmp1_cpu, $tmp2_cpu, $rt, $ut, $vt, $dt, $dx, $dy, $nx, $ny)
        gpu_time = @benchmark CUDA.@sync compute_sources_gpu!($tmp1_gpu, $tmp2_gpu, $rt_gpu, $ut_gpu, $vt_gpu, $dt, $dx, $dy, $nx, $ny)
        
        speedup = median(cpu_time).time / median(gpu_time).time
        println("GPU speedup: $(speedup)x")
        
        # Visualize differences (optional)
        max_diff1 = maximum(abs.(tmp1_from_gpu - tmp1_cpu))
        max_diff2 = maximum(abs.(tmp2_from_gpu - tmp2_cpu))
        println("Maximum absolute difference (tmp1): $max_diff1")
        println("Maximum absolute difference (tmp2): $max_diff2")
        
        # Test with larger grid for better performance comparison
        if nx < 512
            println("\nTesting with larger grid (512x512):")
            nx_large, ny_large = 512, 512
            
            # Initialize larger arrays
            rt_large = ones(Float32, nx_large, ny_large) * 1000.0
            ut_large = zeros(Float32, nx_large, ny_large)
            vt_large = zeros(Float32, nx_large, ny_large)
            tmp1_large = zeros(Float32, nx_large, ny_large)
            tmp2_large = zeros(Float32, nx_large, ny_large)
            
            # Add pattern
            center_x, center_y = nx_large÷2, ny_large÷2
            radius = nx_large÷4
            
            for i in 1:nx_large, j in 1:ny_large
                dist = sqrt((i-center_x)^2 + (j-center_y)^2)
                if dist < radius
                    rt_large[i,j] = 1100.0
                    
                    angle = atan(j-center_y, i-center_x)
                    r_norm = dist / radius
                    
                    if r_norm < 0.5
                        ut_large[i,j] = 0.5 * cos(angle) * (0.5 - r_norm)
                        vt_large[i,j] = 0.5 * sin(angle) * (0.5 - r_norm)
                    else
                        ut_large[i,j] = -0.5 * cos(angle) * (r_norm - 0.5)
                        vt_large[i,j] = -0.5 * sin(angle) * (r_norm - 0.5)
                    end
                else
                    ut_large[i,j] = 0.05 * randn()
                    vt_large[i,j] = 0.05 * randn()
                end
            end
            
            # GPU arrays
            rt_large_gpu = CuArray(rt_large)
            ut_large_gpu = CuArray(ut_large)
            vt_large_gpu = CuArray(vt_large)
            tmp1_large_gpu = CuArray(zeros(Float32, nx_large, ny_large))
            tmp2_large_gpu = CuArray(zeros(Float32, nx_large, ny_large))
            
            # Benchmark larger grid
            println("CPU performance (512x512):")
            @btime compute_sources!($tmp1_large, $tmp2_large, $rt_large, $ut_large, $vt_large, $dt, $dx, $dy, $nx_large, $ny_large)
            
            println("GPU performance (512x512):")
            @btime CUDA.@sync compute_sources_gpu!($tmp1_large_gpu, $tmp2_large_gpu, $rt_large_gpu, $ut_large_gpu, $vt_large_gpu, $dt, $dx, $dy, $nx_large, $ny_large)
            
            # Calculate speedup for larger grid
            cpu_time_large = @benchmark compute_sources!($tmp1_large, $tmp2_large, $rt_large, $ut_large, $vt_large, $dt, $dx, $dy, $nx_large, $ny_large)
            gpu_time_large = @benchmark CUDA.@sync compute_sources_gpu!($tmp1_large_gpu, $tmp2_large_gpu, $rt_large_gpu, $ut_large_gpu, $vt_large_gpu, $dt, $dx, $dy, $nx_large, $ny_large)
            
            speedup_large = median(cpu_time_large).time / median(gpu_time_large).time
            println("GPU speedup (512x512): $(speedup_large)x")
        end
    else
        println("CUDA not available, skipping GPU tests")
    end
end

# Run the test
test_compute_sources()