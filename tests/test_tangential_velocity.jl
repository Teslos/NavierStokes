using CUDA
using Test
using BenchmarkTools

# CPU version
function set_tangential!(u, usouth, unorth, v, vwest, veast, nx, ny)
    # tangential velocity at boundaries
    u[1:nx+1, 1] .= 2 * usouth .- u[1:nx+1, 2]
    u[1:nx+1, ny+2] .= 2 * unorth .- u[1:nx+1, ny+1]
    v[1, 1:ny+1] .= 2 * vwest .- v[2, 1:ny+1]
    v[nx+2, 1:ny+1] .= 2 * veast .- v[nx+1, 1:ny+1]
    return
end

# GPU version
function set_tangential_gpu!(u, usouth, unorth, v, vwest, veast, nx, ny)
    # South boundary
    function kernel_south!(u, usouth, nx)
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        if i <= nx+1
            u[i, 1] = 2 * usouth - u[i, 2]
        end
        return
    end
    
    # North boundary
    function kernel_north!(u, unorth, nx, ny)
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        if i <= nx+1
            u[i, ny+2] = 2 * unorth - u[i, ny+1]
        end
        return
    end
    
    # West boundary
    function kernel_west!(v, vwest, ny)
        j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        if j <= ny+1
            v[1, j] = 2 * vwest - v[2, j]
        end
        return
    end
    
    # East boundary
    function kernel_east!(v, veast, nx, ny)
        j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        if j <= ny+1
            v[nx+2, j] = 2 * veast - v[nx+1, j]
        end
        return
    end
    
    # Launch kernels
    threads = 256
    blocks_x = cld(nx+1, threads)
    blocks_y = cld(ny+1, threads)
    
    @cuda blocks=blocks_x threads=threads kernel_south!(u, usouth, nx)
    @cuda blocks=blocks_x threads=threads kernel_north!(u, unorth, nx, ny)
    @cuda blocks=blocks_y threads=threads kernel_west!(v, vwest, ny)
    @cuda blocks=blocks_y threads=threads kernel_east!(v, veast, nx, ny)
    
    return
end

function test_set_tangential()
    # Test parameters
    nx, ny = 128, 128
    usouth, unorth = 0.1, -0.1
    vwest, veast = 0.2, -0.2
    
    # CPU arrays
    u_cpu = rand(Float32, nx+1, ny+2)
    v_cpu = rand(Float32, nx+2, ny+1)
    
    # Reference arrays for validation
    u_ref = copy(u_cpu)
    v_ref = copy(v_cpu)
    
    # Run CPU version
    set_tangential!(u_cpu, usouth, unorth, v_cpu, vwest, veast, nx, ny)
    
    # Create reference result manually for validation
    u_ref[1:nx+1, 1] .= 2 * usouth .- u_ref[1:nx+1, 2]
    u_ref[1:nx+1, ny+2] .= 2 * unorth .- u_ref[1:nx+1, ny+1]
    v_ref[1, 1:ny+1] .= 2 * vwest .- v_ref[2, 1:ny+1]
    v_ref[nx+2, 1:ny+1] .= 2 * veast .- v_ref[nx+1, 1:ny+1]
    
    # Test CPU result
    @test u_cpu ≈ u_ref
    @test v_cpu ≈ v_ref
    
    # Only run GPU tests if CUDA is available
    if CUDA.functional()
        # Create new arrays for GPU test
        u_orig = rand(Float32, nx+1, ny+2)
        v_orig = rand(Float32, nx+2, ny+1)
        
        # GPU arrays
        u_gpu = CuArray(copy(u_orig))
        v_gpu = CuArray(copy(v_orig))
        
        # CPU arrays for reference
        u_cpu_ref = copy(u_orig)
        v_cpu_ref = copy(v_orig)
        
        # Run GPU version
        set_tangential_gpu!(u_gpu, usouth, unorth, v_gpu, vwest, veast, nx, ny)
        
        # Run CPU version on reference
        set_tangential!(u_cpu_ref, usouth, unorth, v_cpu_ref, vwest, veast, nx, ny)
        
        # Transfer back to CPU for comparison
        u_from_gpu = Array(u_gpu)
        v_from_gpu = Array(v_gpu)
        
        # Test GPU result
        @test u_from_gpu ≈ u_cpu_ref
        @test v_from_gpu ≈ v_cpu_ref
        
        # Benchmark
        println("CPU performance:")
        @btime set_tangential!($u_cpu, $usouth, $unorth, $v_cpu, $vwest, $veast, $nx, $ny)
        
        println("GPU performance:")
        @btime CUDA.@sync set_tangential_gpu!($u_gpu, $usouth, $unorth, $v_gpu, $vwest, $veast, $nx, $ny)
    else
        println("CUDA not available, skipping GPU tests")
    end
end

# Run the test
test_set_tangential()