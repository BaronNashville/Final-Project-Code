include("./helpers.jl")

using RadiiPolynomial, DifferentialEquations

function F!(F::Sequence, a::Sequence, N::Vector{Int}, σ::Float64, r::Float64, equilibrium::Vector{Float64}, λ₁::Float64, λ₂::Float64, ξ₁::Vector{Float64}, ξ₂::Vector{Float64})
    #Extracting what space we are in and initializaing Φ
    s = space(component(a,1));
    Φ = zeros(s^4);

    #Setting all components of F to be 0
    F .= 0;

    #Extracting the components
    F₁ = component(F, 1);
    F₂ = component(F, 2);
    F₃ = component(F, 3);
    F₄ = component(F, 4);

    a₁ = component(a, 1);
    a₂ = component(a, 2);
    a₃ = component(a, 3);
    a₄ = component(a, 4);

    Φ₁ = component(Φ, 1);
    Φ₂ = component(Φ, 2);
    Φ₃ = component(Φ, 3);
    Φ₄ = component(Φ, 4);

    #Computing Φ for higher order terms
    Φ!(Φ, a, σ, r);

    #Computing the sequence operator D = α₁λ₁ + α₂λ₂
    D₁ = zeros(s,s);
    D₂ = zeros(s,s);

    for i in 0:N[1]
        for j in 0:N[2]
            D₁[(i,j),(i,j)] = i;
            D₂[(i,j),(i,j)] = j;
        end
    end

    D = λ₁*D₁ + λ₂*D₂;

    #Setting the appropriate values for the higher order terms
    F₁[:] = (D * a₁ - Φ₁)[:];
    F₂[:] = (D * a₂ - Φ₂)[:];
    F₃[:] = (D * a₃ - Φ₃)[:];
    F₄[:] = (D * a₄ - Φ₄)[:];

    #Setting the appropriate values for the intial conditions
    F₁[(0,0)] = a₁[(0,0)] - equilibrium[1];
    F₂[(0,0)] = a₂[(0,0)] - equilibrium[2];
    F₃[(0,0)] = a₃[(0,0)] - equilibrium[3];
    F₄[(0,0)] = a₄[(0,0)] - equilibrium[4];

    F₁[(1,0)] = a₁[(1,0)] - ξ₁[1];
    F₂[(1,0)] = a₂[(1,0)] - ξ₁[2];
    F₃[(1,0)] = a₃[(1,0)] - ξ₁[3];
    F₄[(1,0)] = a₄[(1,0)] - ξ₁[4];

    F₁[(0,1)] = a₁[(0,1)] - ξ₂[1];
    F₂[(0,1)] = a₂[(0,1)] - ξ₂[2];
    F₃[(0,1)] = a₃[(0,1)] - ξ₂[3];
    F₄[(0,1)] = a₄[(0,1)] - ξ₂[4];
end

function Φ!(Φ::Sequence, a::Sequence, σ::Float64, r::Float64)
    #Setting all componends of Φ to be 0
    Φ .= 0;

    #Extracting the components
    Φ₁ = component(Φ, 1);
    Φ₂ = component(Φ, 2);
    Φ₃ = component(Φ, 3);
    Φ₄ = component(Φ, 4);

    a₁ = component(a, 1);
    a₂ = component(a, 2);
    a₃ = component(a, 3);
    a₄ = component(a, 4);

    #Setting the appropriate values for the higher order terms
    Φ₁[:] = a₂[:];
    Φ₂[:] = project(σ^2*(a₁ - (1+r)*a₃ + r*(a₃*a₃)), space(Φ₂))[:];
    Φ₃[:] = a₄[:];
    Φ₄[:] = project(σ^2*(a₃ - (1+r)*a₁ + r*(a₁*a₁)), space(Φ₄))[:];
end

function DΦ!(DΦ::LinearOperator, a::Sequence, σ::Float64, r::Float64)
    # Initialize DΦ to be zero, then fill in the correct blocks
    DΦ .= 0

    # Extract the space of our sequence
    s = space(component(a,1))

    a₁ = component(a, 1);
    a₃ = component(a, 3);

    component(DΦ, 1, 2).coefficients[:] = project(I, s, s).coefficients[:]

    component(DΦ, 2, 1).coefficients[:] = project(σ^2*I, s, s).coefficients[:]
    component(DΦ, 2, 3).coefficients[:] = project(-σ^2*(1+r)*I, s, s).coefficients[:] + project(σ^2* 2*r*Multiplication(a₃), s, s).coefficients[:]

    component(DΦ, 3, 4).coefficients[:] = project(I, s, s).coefficients[:]

    component(DΦ, 4, 1).coefficients[:] = project(-σ^2*(1+r)*I, s, s).coefficients[:] + project(σ^2* 2*r*Multiplication(a₁), s, s).coefficients[:]
    component(DΦ, 4, 3).coefficients[:] = project(σ^2*I, s, s).coefficients[:]
end

function DF!(DF::LinearOperator, a::Sequence, N::Vector{Int}, σ::Float64, r::Float64, equilibrium::Vector{Float64}, λ₁::Float64, λ₂::Float64, ξ₁::Vector{Float64}, ξ₂::Vector{Float64})
    DF .= 0

    s = space(component(a,1))

    D₁ = zeros(s,s)
    D₂ = zeros(s,s)

    for i in 0:N[1]
        for j in 0:N[2]
            D₁[(i,j),(i,j)] = i 
            D₂[(i,j),(i,j)] = j
        end
    end

    D = λ₁*D₁ + λ₂*D₂

    DΦ = zeros(s^4, s^4)
    DΦ!(DΦ, a, σ, r)

    # Setting the higher order terms
    component(DF,1,1).coefficients[:] = D.coefficients[:] - component(DΦ,1,1).coefficients[:]
    component(DF,2,1).coefficients[:] = -component(DΦ,2,1).coefficients[:]
    component(DF,3,1).coefficients[:] = -component(DΦ,3,1).coefficients[:]
    component(DF,4,1).coefficients[:] = -component(DΦ,4,1).coefficients[:]

    component(DF,1,2).coefficients[:] = -component(DΦ,1,2).coefficients[:]
    component(DF,2,2).coefficients[:] = D.coefficients[:] - component(DΦ,2,2).coefficients[:]
    component(DF,3,2).coefficients[:] = -component(DΦ,3,2).coefficients[:]
    component(DF,4,2).coefficients[:] = -component(DΦ,4,2).coefficients[:]

    component(DF,1,3).coefficients[:] = -component(DΦ,1,3).coefficients[:]
    component(DF,2,3).coefficients[:] = -component(DΦ,2,3).coefficients[:]
    component(DF,3,3).coefficients[:] = D.coefficients[:] - component(DΦ,3,3).coefficients[:]
    component(DF,4,3).coefficients[:] = -component(DΦ,4,3).coefficients[:]

    component(DF,1,4).coefficients[:] = -component(DΦ,1,4).coefficients[:]
    component(DF,2,4).coefficients[:] = -component(DΦ,2,4).coefficients[:]
    component(DF,3,4).coefficients[:] = -component(DΦ,3,4).coefficients[:]
    component(DF,4,4).coefficients[:] = D.coefficients[:] - component(DΦ,4,4).coefficients[:]

    # Setting the lower order terms to zero
    for i in 0:1
        for j in 0:1-i
            component(DF,1,1)[(i,j),:] .= 0
            component(DF,2,1)[(i,j),:] .= 0
            component(DF,3,1)[(i,j),:] .= 0
            component(DF,4,1)[(i,j),:] .= 0

            component(DF,1,2)[(i,j),:] .= 0
            component(DF,2,2)[(i,j),:] .= 0
            component(DF,3,2)[(i,j),:] .= 0
            component(DF,4,2)[(i,j),:] .= 0

            component(DF,1,3)[(i,j),:] .= 0
            component(DF,2,3)[(i,j),:] .= 0
            component(DF,3,3)[(i,j),:] .= 0
            component(DF,4,3)[(i,j),:] .= 0

            component(DF,1,4)[(i,j),:] .= 0
            component(DF,2,4)[(i,j),:] .= 0
            component(DF,3,4)[(i,j),:] .= 0
            component(DF,4,4)[(i,j),:] .= 0
        end
    end

    # Putting the correct lower order terms
    component(DF,1,1)[(0,0),(0,0)] = 1;
    component(DF,2,2)[(0,0),(0,0)] = 1;
    component(DF,3,3)[(0,0),(0,0)] = 1;
    component(DF,4,4)[(0,0),(0,0)] = 1;

    component(DF,1,1)[(1,0),(1,0)] = 1;
    component(DF,2,2)[(1,0),(1,0)] = 1;
    component(DF,3,3)[(1,0),(1,0)] = 1;
    component(DF,4,4)[(1,0),(1,0)] = 1;

    component(DF,1,1)[(0,1),(0,1)] = 1;
    component(DF,2,2)[(0,1),(0,1)] = 1;
    component(DF,3,3)[(0,1),(0,1)] = 1;
    component(DF,4,4)[(0,1),(0,1)] = 1;
end

function DF_approx!(DF_approx::LinearOperator, a::Sequence, N::Vector{Int}, σ::Float64, r::Float64, equilibrium::Vector{Float64}, λ₁::Float64, λ₂::Float64, ξ₁::Vector{Float64}, ξ₂::Vector{Float64})
    DF_approx .= 0;
    h = 0.000001

    a₁ = component(a,1)
    a₂ = component(a,2)
    a₃ = component(a,3)
    a₄ = component(a,4)

    new_a = zeros(space(a))

    new_a₁ = component(new_a,1)
    new_a₂ = component(new_a,2)
    new_a₃ = component(new_a,3)
    new_a₄ = component(new_a,4)

    F = zeros(space(a))
    Fₕ = zeros(space(a))

    for i in 0:N[1]
        for j in 0:N[2]
            new_a₁[:] = a₁[:]
            new_a₂[:] = a₂[:]
            new_a₃[:] = a₃[:]
            new_a₄[:] = a₄[:]

            # First component
            new_a₁[(i,j)] = new_a₁[(i,j)] + h

            F!(F, a, N, σ, r, equilibrium, λ₁, λ₂, ξ₁, ξ₂)
            F!(Fₕ, new_a, N, σ, r, equilibrium, λ₁, λ₂, ξ₁, ξ₂)

            approx = 1/h * (Fₕ-F)

            component(DF_approx, :, 1)[:,(i,j)] = approx[:]

            new_a₁[(i,j)] = new_a₁[(i,j)] - h

            # Second component
            new_a₂[(i,j)] = new_a₂[(i,j)] + h

            F!(F, a, N, σ, r, equilibrium, λ₁, λ₂, ξ₁, ξ₂)
            F!(Fₕ, new_a, N, σ, r, equilibrium, λ₁, λ₂, ξ₁, ξ₂)

            approx = 1/h * (Fₕ-F)

            component(DF_approx, :, 2)[:,(i,j)] = approx[:]

            new_a₂[(i,j)] = new_a₂[(i,j)] - h

            # Third component
            new_a₃[(i,j)] = new_a₃[(i,j)] + h

            F!(F, a, N, σ, r, equilibrium, λ₁, λ₂, ξ₁, ξ₂)
            F!(Fₕ, new_a, N, σ, r, equilibrium, λ₁, λ₂, ξ₁, ξ₂)

            approx = 1/h * (Fₕ-F)

            component(DF_approx, :, 3)[:,(i,j)] = approx[:]

            new_a₃[(i,j)] = new_a₃[(i,j)] - h

            # Fourth component
            new_a₄[(i,j)] = new_a₄[(i,j)] + h

            F!(F, a, N, σ, r, equilibrium, λ₁, λ₂, ξ₁, ξ₂)
            F!(Fₕ, new_a, N, σ, r, equilibrium, λ₁, λ₂, ξ₁, ξ₂)

            approx = 1/h * (Fₕ-F)

            component(DF_approx, :, 4)[:,(i,j)] = approx[:]

            new_a₄[(i,j)] = new_a₄[(i,j)] - h
        end
    end        
end