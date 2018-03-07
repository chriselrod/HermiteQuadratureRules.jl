__precompile__()

module HermiteQuadratureRules

using Reduce, Compat, Base.Cartesian, Distributions, SpecialFunctions


export  generate_weights,
        eval_on_nodes,
        herm_design


include("rules_grad_free.jl")
include("rules.jl")
include("distribution_derivatives.jl")

function __init__()
    setprecision(512)
    Reduce.Rational(false)
    nothing
end

@generated function normal_moment_general(::Val{N}) where N
    expr = rcall(:(int(x^$N * exp(-x^2/2) / y, x, -infinity, infinity)))
    quote
        y = sqrt(big(2)π)
        $expr
    end
end
function normal_moment_general(N::Int)
    y = big(2)π
    rcall(:(int(x^N * exp(-x^2/2) / y, x, -infinity, infinity)))
end
@generated function normal_moment(::Val{N}) where N
    N%2 == 1 ? 0.0 : normal_moment_general(N)
end
@generated function poly_integral_tuple(::Val{N}) where N
    ntuple( i -> normal_moment_general(i) , Val(N))
end

function eval_poly(x::T, β) where T
    @boundscheck length(β) > 1
    @inbounds out = β[1]
    @inbounds for i ∈ 2:length(β)
        out = fma(out, x, β[i])
    end
    out
end




"""
Order is function val, and then derivatives.
"""
function herm_design(nodes, ::Type{T} = BigFloat) where T
    n = length(nodes)
    X = Matrix{T}(2n, 2n)
    buffer = T.(nodes)
    ∂buffer = similar(buffer)
    for iₙ ∈ 1:n
        X[2iₙ-1,1] = one(T)
        X[2iₙ,1] = zero(T)
    end
    for iₙ ∈ 1:n
        X[2iₙ-1,2] = buffer[iₙ]
        X[2iₙ,2] = one(T)
    end
    for j ∈ 2:2n-1
        ∂buffer .= buffer .* j
        buffer .*= nodes
        for iₙ ∈ 1:n
            X[2iₙ-1,j+1] = buffer[iₙ]
            X[2iₙ,j+1] = ∂buffer[iₙ]
        end
    end
    X
end
function herm_design_sym(nodes::AbstractVector{T}) where T
    n = length(nodes)
    X = Matrix{T}(4n+2, 4n+2)
    buffer = T.(nodes)
    ∂buffer = similar(buffer)
    for iₙ ∈ 1:2n+1
        X[2iₙ-1,1] = one(T)
        X[2iₙ,1] = zero(T)
    end
    for iₙ ∈ 1:n
        X[2iₙ-1,2] = -buffer[n-iₙ+1]
        X[2iₙ,2] = one(T)
    end
    X[2n+1,2] = zero(T)
    X[2n+2,2] = one(T)
    for iₙ ∈ 1:n
        X[2iₙ+1+2n,2] = buffer[iₙ]
        X[2iₙ+2+2n,2] = one(T)
    end
    for j ∈ 2:4n+1
        ∂buffer .= buffer .* j
        buffer .*= nodes
        for iₙ ∈ 1:n
            X[2iₙ-1,j+1] = (-1)^j* buffer[n-iₙ+1]
            X[2iₙ,j+1]  = (-1)^(j-1)*∂buffer[n-iₙ+1]
        end
        X[2n+1,j+1] = zero(T)
        X[2n+2,j+1] = one(T)
        for iₙ ∈ 1:n
            X[2iₙ+1+2n,j+1] = buffer[iₙ]
            X[2iₙ+2+2n,j+1] = ∂buffer[iₙ]
        end
    end
    X
end
function herm_design_sym_reversed_order(nodes::AbstractVector{T}) where T
    n = length(nodes)
    N = 4n+2
    X = Matrix{T}(N,N)
    buffer = Array{T}(uninitialized, n)
    ∂buffer = Array{T}(uninitialized, n)
    for j ∈ 1:N-2
        exponent = N - j
        @. ∂buffer = exponent * nodes ^ (exponent - 1)
        @. buffer = nodes ^ exponent
        for iₙ ∈ 1:n
            X[2iₙ-1,j] = (-1)^exponent * buffer[n-iₙ+1]
            X[2iₙ,j]  = (-1)^(exponent-1) * ∂buffer[n-iₙ+1]
        end
        X[2n+1,j] = zero(T)
        X[2n+2,j] = one(T)
        for iₙ ∈ 1:n
            X[2iₙ+1+2n,j] = buffer[iₙ]
            X[2iₙ+2+2n,j] = ∂buffer[iₙ]
        end
    end
    @. buffer = T(nodes)
    for iₙ ∈ 1:n
        X[2iₙ-1,N-1] = -buffer[n-iₙ+1]
        X[2iₙ,N-1] = one(T)
    end
    X[2n+1,N-1] = zero(T)
    X[2n+2,N-1] = one(T)
    for iₙ ∈ 1:n
        X[2iₙ+1+2n,N-1] = buffer[iₙ]
        X[2iₙ+2+2n,N-1] = one(T)
    end
    for iₙ ∈ 1:2n+1
        X[2iₙ-1,N] = one(T)
        X[2iₙ,N] = zero(T)
    end
    X
end

@generated function weight_mat(Xⁱ::AbstractMatrix{T}, ::Val{N}) where {N,T}
    quote
        weights = Matrix{BigFloat}($N, size(Xⁱ,2))
        for j ∈ 1:size(Xⁱ,2)
            @nexprs $N i -> begin
                weights[i,j] = normal_moment_general(Val{2(i-1)}()) * Xⁱ[2(i-1)+1,j]
            end
        end
        weights 
    end
end
@generated function weight_vec(Xⁱ::AbstractMatrix{T}, ::Val{N}) where {N,T}
    quote
        weights = fill(zero(T), size(Xⁱ,2))
        for j ∈ 1:size(Xⁱ,2)
            @nexprs $N i -> begin
                weights[j] += normal_moment_general(Val{2(i-1)}()) * Xⁱ[2(i-1)+1,j]
            end
        end
        weights 
    end
end
function weight_vec(Xⁱ::AbstractMatrix{T}) where T#Dynamic dispatch
    weight_vec(Xⁱ, Val{size(Xⁱ,1) ÷ 2}())::Vector{T}
end

function generate_weights(nodes)
    X = herm_design(nodes)
    weight_vec(inv(X))
end

function generate_weights_sym(nodes)
    X = herm_design_sym(nodes)
    weight_vec(inv(X))
end

function eval_on_nodes(f, nodes::AbstractVector{T}) where T
    out = Matrix{T}(2, length(nodes))
    for i ∈ eachindex(nodes)
        fn = f(nodes[i] + eps()*im )
        out[1,i] = imag(fn) / eps()
        out[2,i] = real(fn)
    end
    out
end
function eval_on_nodes(f::Distributions.UnivariateDistribution, nodes::AbstractVector{T}) where T
    out = Matrix{T}(2, length(nodes))
    for i ∈ eachindex(nodes)
        out[:,i] .= fg(f, nodes[i])
    end
    out
end
function eval_weighted_nodes(f::Distributions.UnivariateDistribution, nodes::AbstractVector{T}) where T
    out = Matrix{T}(2, length(nodes))
    for i ∈ eachindex(nodes)
        out[:,i] .= fgw(f, nodes[i])
    end
    out
end


using Optim, ForwardDiff
initial_n = BigFloat.(sort(rand(4)));
node_weight_magnitude(nodes) = sum(x -> (x^8-1//(4length(nodes)+2) ), generate_weights_sym(nodes))
td = TwiceDifferentiable(node_weight_magnitude, initial_n; autodiff = :forward)

opt = optimize(td, initial_n, Newton())
end # module
