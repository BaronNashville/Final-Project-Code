using RadiiPolynomial, DifferentialEquations

function generate_manifold_data(a::Sequence, range::Matrix{Float64}, num_points::Vector{Int})
    data = zeros(4, (num_points[1]*num_points[2]))

    for i in 0:num_points[1]-1
        for j in 0:num_points[2]-1
            θ₁ = range[1,1] + i/(num_points[1]-1) * (range[1,2] - range[1,1])
            θ₂ = range[2,1] + j/(num_points[2]-1) * (range[2,2] - range[2,1])

            data[:,1 + i*num_points[2] + j] = a(θ₁, θ₂)
        end
    end

    return data
end

function distance_from_fixed_points(data)
    min_distance = Inf
    min_distance_location = (data[:,1])

    reflection = [
        0 0 1 0
        0 0 0 -1
        1 0 0 0
        0 -1 0 0
    ]

    for i in 1:length(data[1,:])
        x = data[:,i]

        reflected_x = reflection * x

        if 0.5*norm(x-reflected_x) < min_distance
            min_distance = 0.5*norm(x-reflected_x)
            min_distance_location = x
        end
        
    end

    return min_distance, min_distance_location
end

function logistic_prime(x::Float64, r::Float64)
    return 1+r - 2*r*x;
end

function logistic(x::Float64, r::Float64)
    return (1+r)*x - r*x^2;
end

function Df(equilibrium::Vector{Float64}, σ::Float64, r::Float64)
    return [
        0       1       0       0
        σ^2    0       -σ^2*logistic_prime(equilibrium[3], r)     0
        0       0       0       1
        -σ^2*logistic_prime(equilibrium[1], r)     0       σ^2        0
    ]
end

function integrate_boundary(a::Sequence, σ::Float64, r::Float64, range::Matrix{Float64}, num_points::Vector{Int}, T::Float64)
    points = zeros(4,1)

    tspan = (0.0,-T)

    f(x,p,t) = [
        x[2]
        σ*(x[1] - logistic(x[3], r))
        x[4]
        σ*(x[3] - logistic(x[1], r))
    ]

    
    for i in [0,num_points[1]-1]
        for j in 0:num_points[2]-1
            θ₁ = range[1,1] + i/(num_points[1]-1) * (range[1,2] - range[1,1])
            θ₂ = range[2,1] + j/(num_points[2]-1) * (range[2,2] - range[2,1])

            x₀ = a(θ₁, θ₂)

            prob = ODEProblem(f, x₀, tspan)
            sol = solve(prob, Tsit5(), reltol = 1e-6, saveat = 0.001)

            points = [points;;mapreduce(permutedims, vcat, sol.u)']
        end
    end

    for i in 0:num_points[1]-1
        for j in [0,num_points[2]-1]
            θ₁ = range[1,1] + i/(num_points[1]-1) * (range[1,2] - range[1,1])
            θ₂ = range[2,1] + j/(num_points[2]-1) * (range[2,2] - range[2,1])

            x₀ = a(θ₁, θ₂)

            prob = ODEProblem(f, x₀, tspan)
            sol = solve(prob, Tsit5(), reltol = 1e-6, saveat = 0.001)

            points = [points;;mapreduce(permutedims, vcat, sol.u)']
        end
    end    

    return points[:,2:end]
end

function integrate_point(x₀::Vector{Float64}, σ::Float64, r::Float64, Tspan::Tuple{Float64, Float64})
    f(x,p,t) = [
        x[2]
        σ*(x[1] - logistic(x[3], r))
        x[4]
        σ*(x[3] - logistic(x[1], r))
    ]

    prob = ODEProblem(f, x₀, Tspan)
    sol = solve(prob, Tsit5(), reltol = 1e-14, saveat = 0.001)

    return mapreduce(permutedims, vcat, sol.u)', sol.t
end

function integrate_conjugacy_point(θ₀::Vector{Float64}, a::Sequence, λ₁::Float64, λ₂::Float64, Tspan::Tuple{Float64, Float64})
    f(θ,p,t) = [
        λ₁*θ[1]
        λ₂*θ[2]
    ]

    prob = ODEProblem(f, θ₀, Tspan)
    sol = solve(prob, Tsit5(), reltol = 1e-14, saveat = 0.001)

    θ_sol = mapreduce(permutedims, vcat, sol.u)'

    x_sol = zeros(4, length(θ_sol[1,:]))

    for i in 1:length(θ_sol[1,:])
        x_sol[:,i] = a(θ_sol[1,i], θ_sol[2,i])
    end
    #println(x_sol)

    return x_sol, sol.t
end

function reflection_data(data)
    new_data = copy(data)

    reflection = [
        0 0 1 0
        0 0 0 -1
        1 0 0 0
        0 -1 0 0
    ]

    for i in 1:length(data[1,:])
        new_data[:,i] = reflection*data[:,i]
    end

    return new_data
end

function get_theta(a::Sequence, range::Matrix{Float64}, num_points::Vector{Int}, x::Vector{Float64})
    θ = [0,0]
    min_distance = Inf

    for i in [0,num_points[1]-1]
        for j in 0:num_points[2]-1
            θ₁ = range[1,1] + i/(num_points[1]-1) * (range[1,2] - range[1,1])
            θ₂ = range[2,1] + j/(num_points[2]-1) * (range[2,2] - range[2,1])

            x₀ = a(θ₁, θ₂)

            if norm(x₀-x) < min_distance
                min_distance = norm(x₀-x)
                θ = [θ₁, θ₂]
            end
        end
    end

    for i in 0:num_points[1]-1
        for j in [0,num_points[2]-1]
            θ₁ = range[1,1] + i/(num_points[1]-1) * (range[1,2] - range[1,1])
            θ₂ = range[2,1] + j/(num_points[2]-1) * (range[2,2] - range[2,1])

            x₀ = a(θ₁, θ₂)

            if norm(x₀-x) < min_distance
                min_distance = norm(x₀-x)
                θ = [θ₁, θ₂]
            end
        end
    end    

    return θ
end

function g(X, a::Sequence, σ::Float64, r::Float64)
    θ₁ = X[1]
    θ₂ = X[2]
    L = X[3]

    f(x,p,t) = [
        x[2]
        σ*(x[1] - logistic(x[3], r))
        x[4]
        σ*(x[3] - logistic(x[1], r))
    ]

    prob = ODEProblem(f, a(θ₁, θ₂), (L,0))
    sol = solve(prob, Tsit5(), reltol = 1e-14, save_everystep = false)

    return mapreduce(permutedims, vcat, sol.u)'[:,end]
end

function w(X, a::Sequence, σ::Float64, r::Float64)
    θ₁ = X[1]
    θ₂ = X[2]
    L = X[3]

    return [
        g(X, a, σ, r)[1] - g(X, a, σ, r)[3]
        g(X, a, σ, r)[2] + g(X, a, σ, r)[4]
        θ₁^2 + θ₂^2 - 0.95
    ]
end

function Dw(X, a::Sequence, σ::Float64, r::Float64)
    h = 0.000001
    A = zeros(3,3)
    Id = [
        1 0 0
        0 1 0
        0 0 1
    ]

    for i in 1:3
        A[:,i] = 1/(2*h) * (w(X+h*Id[:,i], a, σ, r)-w(X-h*Id[:,i], a, σ, r))
    end

    return A
end

function candidate_finder(X, a::Sequence, σ::Float64, r::Float64)
    θ₁ = X[1]
    θ₂ = X[2]
    L = X[3]

    X, = newton(X -> (w(X, a, σ, r), Dw(X, a, σ, r)), X)

    return X[1:2],X[3]
end