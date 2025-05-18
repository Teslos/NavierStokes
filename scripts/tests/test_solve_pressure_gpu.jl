using CUDA
using Test
using BenchmarkTools
using CUDA

# Red kernel (updates cells where i+j is even)
function solve_pressure_kernel_red!(p, tmp1, tmp2, rt, beta, dx, dy)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    
    if (i > 1 && i <= size(p,1)-1 && j > 1 && j <= size(p,2)-1)
        # Only update red cells (i+j is even)
        if (i + j) % 2 == 0
            p[i,j] = (1.0-beta)*p[i,j] + beta*tmp2[i,j]*(
            (1.0/dx)*(p[i+1,j]/(dx*(rt[i+1,j]+rt[i,j])) +
            p[i-1,j]/(dx*(rt[i-1,j]+rt[i,j]))) +
            (1.0/dy)*(p[i,j+1]/(dy*(rt[i,j+1]+rt[i,j])) +
            p[i,j-1]/(dy*(rt[i,j-1]+rt[i,j]))) - tmp1[i,j])
        end
    end
    return
end

# Black kernel (updates cells where i+j is odd)
function solve_pressure_kernel_black!(p, tmp1, tmp2, rt, beta, dx, dy)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    
    if (i > 1 && i <= size(p,1)-1 && j > 1 && j <= size(p,2)-1)
        # Only update black cells (i+j is odd)
        if (i + j) % 2 == 1
            p[i,j] = (1.0-beta)*p[i,j] + beta*tmp2[i,j]*(
            (1.0/dx)*(p[i+1,j]/(dx*(rt[i+1,j]+rt[i,j])) +
            p[i-1,j]/(dx*(rt[i-1,j]+rt[i,j]))) +
            (1.0/dy)*(p[i,j+1]/(dy*(rt[i,j+1]+rt[i,j])) +
            p[i,j-1]/(dy*(rt[i,j-1]+rt[i,j]))) - tmp1[i,j])
        end
    end
    return
end

# Function to launch the GPU kernels for a single Red-Black SOR iteration
function solve_pressure_redblack_gpu!(p, tmp1, tmp2, rt, beta, dx, dy)
    nx, ny = size(p)
    threads = (16, 16)
    blocks = (cld(nx, threads[1]), cld(ny, threads[2]))
    
    # First update red cells
    @cuda blocks=blocks threads=threads solve_pressure_kernel_red!(p, tmp1, tmp2, rt, beta, dx, dy)
    
    # Synchronize to ensure all red updates are complete
    CUDA.synchronize()
    
    # Then update black cells
    @cuda blocks=blocks threads=threads solve_pressure_kernel_black!(p, tmp1, tmp2, rt, beta, dx, dy)
    
    # Synchronize again
    CUDA.synchronize()
    
    return
end

# Function to perform full pressure solver iteration using Red-Black SOR
function full_pressure_solver_redblack_gpu!(p, tmp1, tmp2, rt, beta, dx, dy, maxit, maxError)
    error = 1.0
    iter = 0
    
    # GPU arrays
    p_gpu = CuArray(p)
    tmp1_gpu = CuArray(tmp1)
    tmp2_gpu = CuArray(tmp2)
    rt_gpu = CuArray(rt)
    
    while error > maxError && iter < maxit
        # Store old pressure for error calculation
        p_old_gpu = copy(p_gpu)
        
        # Single iteration of Red-Black SOR
        solve_pressure_redblack_gpu!(p_gpu, tmp1_gpu, tmp2_gpu, rt_gpu, beta, dx, dy)
        
        # Calculate error
        p_diff = maximum(abs.(p_gpu - p_old_gpu))
        p_max = maximum(abs.(p_gpu))
        error = p_diff / p_max  # Convert to scalar
        iter += 1
        
        # Optional: print progress every 100 iterations
        if iter % 100 == 0
            println("Iteration $iter, error = $error")
        end
    end
    
    # Transfer result back to CPU
    p .= Array(p_gpu)
    
    return iter, error
end

# CPU version for comparison
function solve_pressure!(p, tmp1, tmp2, rt, beta, dx, dy)
    # solve for pressure
    Threads.@threads for i=2:size(p,1)-1
        for j=2:size(p,2)-1
            p[i,j] = (1.0-beta)*p[i,j] + beta*tmp2[i,j]*(
            (1.0/dx)*(p[i+1,j]/(dx*(rt[i+1,j]+rt[i,j])) +
            p[i-1,j]/(dx*(rt[i-1,j]+rt[i,j]))) +
            (1.0/dy)*(p[i,j+1]/(dy*(rt[i,j+1]+rt[i,j])) +
            p[i,j-1]/(dy*(rt[i,j-1]+rt[i,j]))) - tmp1[i,j])
        end
    end
    return
end

# Function to perform full pressure solver iteration on CPU
function full_pressure_solver_cpu!(p, tmp1, tmp2, rt, beta, dx, dy, maxit, maxError)
    error = 1.0
    iter = 0
    
    while error > maxError && iter < maxit
        # Store old pressure for error calculation
        p_old = copy(p)
        
        # Single iteration of pressure solver
        solve_pressure!(p, tmp1, tmp2, rt, beta, dx, dy)
        
        # Calculate error
        error = maximum(abs.(p - p_old)) / maximum(abs.(p))
        iter += 1
        
        # Optional: print progress every 100 iterations
        if iter % 100 == 0
            println("Iteration $iter, error = $error")
        end
    end
    
    return iter, error
end

# Example usage
function test_pressure_solver()
    # Test parameters
    nx, ny = 256, 256
    dx, dy = 0.1, 0.1
    beta = 1.5      # SOR relaxation parameter
    maxit = 5000    # Maximum iterations
    maxError = 1e-5 # Convergence criterion
    
    # Initialize arrays
    rt = ones(Float32, nx, ny) * 1000.0  # Density
    p = zeros(Float32, nx, ny)           # Pressure
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
            
            # Initialize divergence field (source/sink pattern)
            r_norm = dist / radius
            if r_norm < 0.5
                tmp1[i,j] = 10.0 * (0.5 - r_norm)  # Source
            else
                tmp1[i,j] = -10.0 * (r_norm - 0.5) # Sink
            end
        end
    end
    
    # Initialize pressure coefficients
    for i in 2:nx-1, j in 2:ny-1
        tmp2[i,j] = 1.0 / ((1.0/dx) * (1.0 / (dx * (rt[i+1,j] + rt[i,j])) + 
                    1.0 / (dx * (rt[i-1,j] + rt[i,j]))) 
                    + (1.0/dy) * (1.0 / (dy * (rt[i,j+1] + rt[i,j]))
                    + 1.0 / (dy * (rt[i,j-1] + rt[i,j]))))
    end
    
    # Run CPU solver
    p_cpu = copy(p)
    println("Running CPU solver...")
    cpu_iters, cpu_error = @time full_pressure_solver_cpu!(p_cpu, tmp1, tmp2, rt, beta, dx, dy, maxit, maxError)
    println("CPU solver: $cpu_iters iterations, final error: $cpu_error")
    
    # Run GPU Red-Black SOR solver
    p_gpu = copy(p)
    println("\nRunning GPU Red-Black SOR solver...")
    gpu_iters, gpu_error = @time full_pressure_solver_redblack_gpu!(p_gpu, tmp1, tmp2, rt, beta, dx, dy, maxit, maxError)
    println("GPU solver: $gpu_iters iterations, final error: $gpu_error")
    
    # Compare results
    max_diff = maximum(abs.(p_gpu - p_cpu))
    rel_diff = max_diff / maximum(abs.(p_cpu))
    println("\nMaximum absolute difference: $max_diff")
    println("Relative difference: $rel_diff")
    
    return p_cpu, p_gpu, cpu_iters, gpu_iters
end
# Run the test
p_cpu, p_gpu, cpu_iters, gpu_iters = test_pressure_solver()