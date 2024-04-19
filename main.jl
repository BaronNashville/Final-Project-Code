include("./functions.jl")
include("./helpers.jl")

using RadiiPolynomial, GLMakie

# Setting up the equation parameters
σ::Float64 = 10;
r::Float64 = 2.2;

# Setting up the numerics parameters
N = [50,30];  # Number of Taylor coefficients

manifold_range::Matrix{Float64} = [-1 1; -1 1];  # How long in time we grow the manifolds
manifold_num_points::Vector{Int64} = [1000,1000]; # How many points in time are we using

vector_length::Vector{Float64} = [4, 4];    # Length of eigenvectors for parameterization method

if r <= 2 || r >= sqrt(5)
    error("Chosen r does not satisfy the requirements")
end

# Two cycle of the logistic map where n₋ < n₊
n₋ = (r+2 - sqrt(r^2 -4))/(2*r);
n₊ = (r+2 + sqrt(r^2 -4))/(2*r);

println("n₋ = " * string(n₋)  * "\nn₊ = " * string(n₊));

# Evaluating the derivative of the logistic map at n₋ and n₊
n₋_prime = logistic_prime(n₋, r);
n₊_prime = logistic_prime(n₊, r);    

println("n₋_prime = " * string(n₋_prime) * "\nn₊_prime = " * string(n₊_prime));
println();


#Unstable manifold
equilibrium₁ = [n₋; 0; n₊; 0];

A₁ = Df(equilibrium₁, σ, r);

λ₁ = sqrt(σ^2 * (1 - sqrt(n₋_prime*n₊_prime)));
ξ₁ = [(-sqrt(Complex(n₊_prime))/(sqrt(Complex(n₋_prime))*λ₁)).re; -sqrt(n₊_prime/n₋_prime); 1/λ₁; 1];
ξ₁ = (vector_length[1] / norm(ξ₁)) * ξ₁;

println("λ₁ = " * string(λ₁) * "\nξ₁ = " * string(ξ₁));
println("||A₁*ξ₁ - λ₁*ξ₁|| = " * string(norm(A₁*ξ₁ - λ₁*ξ₁)));
println();

λ₂ = sqrt(σ^2 * (1 + sqrt(n₋_prime*n₊_prime)));
ξ₂ = [(sqrt(Complex(n₊_prime))/(sqrt(Complex(n₋_prime))*λ₂)).re; sqrt(n₊_prime/n₋_prime); 1/λ₂; 1];
ξ₂ = (vector_length[2] / norm(ξ₂)) * ξ₂;

println("λ₂ = " * string(λ₂) * "\nξ₂ = " * string(ξ₂));
println("||A₁*ξ₂ - λ₂*ξ₂|| = " * string(norm(A₁*ξ₂ - λ₂*ξ₂)));
println();

# Stable manifold
equilibrium₂ = [n₊; 0; n₋; 0];

A₂ = Df(equilibrium₂, σ, r);

λ₃ = -sqrt(σ^2*(1-sqrt(n₋_prime*n₊_prime)));
ξ₃ = [(-sqrt(Complex(n₋_prime))/(sqrt(complex(n₊_prime))*λ₃)).re; -sqrt(n₋_prime/n₊_prime); 1/λ₃; 1];
ξ₃ = (vector_length[1] / norm(ξ₃)) * ξ₃;

println("λ₃ = " * string(λ₃) * "\nξ₃ = " * string(ξ₃));
println("||A₂*ξ₃ - λ₃*ξ₃|| = " * string(norm(A₂*ξ₃ - λ₃*ξ₃)));
println();

λ₄ = -sqrt(σ^2*(1+sqrt(n₋_prime*n₊_prime)));
ξ₄ = [(sqrt(Complex(n₋_prime))/(sqrt(Complex(n₊_prime))*λ₄)).re; sqrt(n₋_prime/n₊_prime); 1/λ₄; 1];
ξ₄ = (vector_length[2] / norm(ξ₄)) * ξ₄;

println("λ₄ = " * string(λ₄) * "\nξ₄ = " * string(ξ₄));
println("||A₂*ξ₄ - λ₄*ξ₄|| = " * string(norm(A₂*ξ₄ - λ₄*ξ₄)));
println();

S = Taylor(N[1]) ⊗ Taylor(N[2]); # 2-index Taylor sequence space
a = zeros(S^4);

a, = newton!((F, DF, a) -> (F!(F, a, N, σ, r, equilibrium₂, λ₃, λ₄, ξ₃, ξ₄), DF!(DF, a, N, σ, r, equilibrium₂, λ₃, λ₄, ξ₃, ξ₄)), a)

a₁ = component(a,1)
a₂ = component(a,2)
a₃ = component(a,3)
a₄ = component(a,4)

println("a₁(N₁,0) = " * string(a₁[(N[1],0)]) * ", a₁(0,N₂) = " * string(a₁[(0,N[2])]))
println("a₂(N₁,0) = " * string(a₂[(N[1],0)]) * ", a₂(0,N₂) = " * string(a₂[(0,N[2])]))
println("a₃(N₁,0) = " * string(a₃[(N[1],0)]) * ", a₃(0,N₂) = " * string(a₃[(0,N[2])]))
println("a₄(N₁,0) = " * string(a₄[(N[1],0)]) * ", a₄(0,N₂) = " * string(a₄[(0,N[2])]))

manifold_data = generate_manifold_data(a, manifold_range, manifold_num_points)
reflected_manifold_data = reflection_data(manifold_data)

distance, location = distance_from_fixed_points(manifold_data)
println("Minkowksi distance between manifold and fixed points of R = " * string(distance) * " attained at point x = " * string(location))


θ = get_theta(a, manifold_range, manifold_num_points, location)

θ,L = candidate_finder([θ₁,θ₂,0], a, σ, r)

println("θ = " * string(θ) * ", L = " * string(L))


# Integrating possible intersection point
connecting_orbit_data, connecting_orbit_time = integrate_point(a(θ[1],θ[2]), σ, r, (L,0.0))
reflected_connecting_orbit_data = reflection_data(connecting_orbit_data)

# Applying the conjugacy to get values inside the manifold
manifold_orbit_data, manifold_orbit_time = integrate_conjugacy_point(θ, a, λ₃, λ₄, (L,2.0))
reflected_manifold_orbit_data = reflection_data(manifold_orbit_data)


# Plotting

# Using GLMakie
fig = Figure()
ODEax = Axis3(fig[1,1],
    title = L"Manifolds and conntecting orbit for $σ = 10$ and $r = 2.2$",
    titlesize = 20,
    xlabel = L"$x_1$",
    xlabelsize = 20,
    ylabel = L"$x_3$",
    ylabelsize = 20,
    zlabel = L"$x_2$",
    zlabelsize = 20   
)

# Stable manifold
stable = GLMakie.surface!(ODEax,
    reshape(manifold_data[1,:], manifold_num_points[2], manifold_num_points[1]),
    reshape(manifold_data[3,:], manifold_num_points[2], manifold_num_points[1]),
    reshape(manifold_data[2,:], manifold_num_points[2], manifold_num_points[1]),
    color = reshape(manifold_data[4,:], manifold_num_points[2], manifold_num_points[1]),
    colorrange = (-4,4),
    transparency = true,
)

# Unstable manifold
unstable = GLMakie.surface!(ODEax,
    reshape(reflected_manifold_data[1,:], manifold_num_points[2], manifold_num_points[1]),
    reshape(reflected_manifold_data[3,:], manifold_num_points[2], manifold_num_points[1]),
    reshape(reflected_manifold_data[2,:], manifold_num_points[2], manifold_num_points[1]),
    color = reshape(reflected_manifold_data[4,:], manifold_num_points[2], manifold_num_points[1]),
    colorrange = (-4,4),
    transparency = true,
)


# Connecting orbit
orbit = GLMakie.lines!(ODEax,
    [reverse(reflected_manifold_orbit_data[1,1:end-1]);reflected_connecting_orbit_data[1,1:end-1];reverse(connecting_orbit_data[1,1:end-1]); manifold_orbit_data[1,:]],
    [reverse(reflected_manifold_orbit_data[3,1:end-1]);reflected_connecting_orbit_data[3,1:end-1];reverse(connecting_orbit_data[3,1:end-1]); manifold_orbit_data[3,:]],
    [reverse(reflected_manifold_orbit_data[2,1:end-1]);reflected_connecting_orbit_data[2,1:end-1];reverse(connecting_orbit_data[2,1:end-1]); manifold_orbit_data[2,:]],
    color = [reverse(reflected_manifold_orbit_data[4,1:end-1]);reflected_connecting_orbit_data[4,1:end-1];reverse(connecting_orbit_data[4,1:end-1]); manifold_orbit_data[4,:]],
    colorrange = (-4,4),
    label = "Connecting Orbit"
)


fp1 = GLMakie.scatter!(ODEax,
    n₊,
    n₋,
    0,
    color = :red,
    markersize = 15,
    label = "Fixed Points"
)

fp2 = GLMakie.scatter!(ODEax,
    n₋,
    n₊,
    0,
    color = :red,
    markersize = 15
)

Colorbar(fig[1,2], limits = (-4,4), label = L"x_4", labelsize = 20)
axislegend()


IDEax = Axis(fig[1,3],
    title = L"Two-cycle of IDE for $σ = 10$ and $r = 2.2$",
    titlesize = 20,
    xlabel = L"$t$",
    xlabelsize = 20,
    ylabel = L"$y$",
    ylabelsize = 20,
    limits = (nothing, (0, 1.5))  
)

GLMakie.lines!(IDEax,
    [-reverse(manifold_orbit_time[1:end-1]);-connecting_orbit_time[1:end-1]; reverse(connecting_orbit_time[1:end-1]);manifold_orbit_time],
    [reverse(reflected_manifold_orbit_data[1,1:end-1]);reflected_connecting_orbit_data[1,1:end-1]; reverse(connecting_orbit_data[1,1:end-1]);manifold_orbit_data[1,:]],
    color = :red,
    label = L"N"
)

GLMakie.lines!(IDEax,
    [-reverse(manifold_orbit_time[1:end-1]);-connecting_orbit_time[1:end-1]; reverse(connecting_orbit_time[1:end-1]);manifold_orbit_time],
    [reverse(reflected_manifold_orbit_data[3,1:end-1]);reflected_connecting_orbit_data[3,1:end-1]; reverse(connecting_orbit_data[3,1:end-1]);manifold_orbit_data[3,:]],
    color = :blue,
    label = L"M"
)

GLMakie.lines!(IDEax,
    [-reverse(manifold_orbit_time[1:end-1]);-connecting_orbit_time[1:end-1]; reverse(connecting_orbit_time)[1:end-1];manifold_orbit_time],
    n₋*ones(length([-reverse(manifold_orbit_time[1:end-1]);-connecting_orbit_time[1:end-1]; reverse(connecting_orbit_time)[1:end-1];manifold_orbit_time])),
    color = :black,
    label = L"n_-"
)

GLMakie.lines!(IDEax,
    [-reverse(manifold_orbit_time[1:end-1]);-connecting_orbit_time[1:end-1]; reverse(connecting_orbit_time)[1:end-1];manifold_orbit_time],
    n₊*ones(length([-reverse(manifold_orbit_time[1:end-1]);-connecting_orbit_time[1:end-1]; reverse(connecting_orbit_time)[1:end-1];manifold_orbit_time])),
    color = :black,
    label = L"n_+"
)

axislegend()



display(fig)