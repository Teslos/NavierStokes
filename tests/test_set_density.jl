using CUDA
using Test
using BenchmarkTools

# CPU version
function set_density_viscosity!(r, m, xc, yc, x, y, rad, rho2, m2)
    Threads.@threads for i=2:size(r,1)-1
        for j=2:size(r,2)-1
            if (x[i]-xc)^2 + (y[j]-yc)^2 < rad^2
                r[i,j] = rho2
                m[i,j] = m2
            end
        end
    end
    return
end

# GPU version
function set_density_viscosity_gpu!(r, m, xc, yc, x, y, rad, rho2, m2)
    function kernel!(r, m, xc, yc, x, y, rad, rho2, m2)
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + 1
        j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + 1
        
        if 2 <= i <= size(r, 1) - 1 && 2 <= j <= size(r, 2) - 1
            if (x[i] - xc)^2 + (y[j] - yc)^2 < rad^2
                r[i, j] = rho2
                m[i, j] = m2
            end
        end
        return
    end
    
    nx, ny = size(r)
    threads = (16, 16)
    blocks = (cld(nx, threads[1]), cld(ny, threads[2]))
    
    @cuda blocks=blocks threads=threads kernel!(r, m, xc, yc, x, y, rad, rho2, m2)
    return
end

function test_density_viscosity()
    # Test parameters
    nx, ny = 256, 256
    xc, yc = 0.5, 0.5
    rad = 0.2
    rho2 = 2.0
    m2 = 0.1
    
    # Create grid
    x = range(0, 1, length=nx)
    y = range(0, 1, length=ny)
    
    # CPU arrays
    r_cpu = ones(Float32, nx, ny)
    m_cpu = ones(Float32, nx, ny) * 0.01
    
    # Reference arrays for validation
    r_ref = copy(r_cpu)
    m_ref = copy(m_cpu)
    
    # Run CPU version
    set_density_viscosity!(r_cpu, m_cpu, xc, yc, x, y, rad, rho2, m2)
    
    # Create reference result manually for validation
    for i in 2:nx-1
        for j in 2:ny-1
            if (x[i]-xc)^2 + (y[j]-yc)^2 < rad^2
                r_ref[i,j] = rho2
                m_ref[i,j] = m2
            end
        end
    end
    
    # Test CPU result
    @test r_cpu ≈ r_ref
    @test m_cpu ≈ m_ref
    
    # Only run GPU tests if CUDA is available
    if CUDA.functional()
        # GPU arrays
        r_gpu = CuArray(ones(Float32, nx, ny))
        m_gpu = CuArray(ones(Float32, nx, ny) * 0.01)
        x_gpu = CuArray(collect(x))
        y_gpu = CuArray(collect(y))
        
        # Run GPU version
        set_density_viscosity_gpu!(r_gpu, m_gpu, xc, yc, x_gpu, y_gpu, rad, rho2, m2)
        
        # Transfer back to CPU for comparison
        r_from_gpu = Array(r_gpu)
        m_from_gpu = Array(m_gpu)
        
        # Test GPU result
        @test r_from_gpu ≈ r_ref
        @test m_from_gpu ≈ m_ref
        
        # Benchmark
        println("CPU performance:")
        @btime set_density_viscosity!($r_cpu, $m_cpu, $xc, $yc, $x, $y, $rad, $rho2, $m2)
        
        println("GPU performance:")
        @btime CUDA.@sync set_density_viscosity_gpu!($r_gpu, $m_gpu, $xc, $yc, $x_gpu, $y_gpu, $rad, $rho2, $m2)
    else
        println("CUDA not available, skipping GPU tests")
    end
end

# Run the test


test_density_viscosity()