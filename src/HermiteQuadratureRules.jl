__precompile__()

module HermiteQuadratureRules

using Reduce, Compat, Base.Cartesian, Distributions, SpecialFunctions


export  generate_weights,
        eval_on_nodes,
        herm_design


include("distribution_derivatives.jl")

function __init__()
    setprecision(512)
    Reduce.Rational(false)
    nothing
end

@generated function normal_moment_general(::Val{N}) where N
    rcall(:(int(x^$N * exp(-x^2/2), x, -infinity, infinity)))
end
function normal_moment_general(N::Int)
    rcall(:(int(x^N * exp(-x^2/2), x, -infinity, infinity)))
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
Order is derivatives, and then function val.
"""
function herm_design(nodes, ::Type{T} = BigFloat) where T
    n = length(nodes)
    X = Matrix{T}(2n, 2n)
    buffer = T.(nodes)
    ∂buffer = similar(buffer)
    for iₙ ∈ 1:n
        X[2iₙ-1,1] = zero(T)
        X[2iₙ,1] = one(T)
    end
    for iₙ ∈ 1:n
        X[2iₙ-1,2] = one(T)
        X[2iₙ,2] = buffer[iₙ]
    end
    for j ∈ 2:2n-1
        ∂buffer .= buffer .* j
        buffer .*= nodes
        for iₙ ∈ 1:n
            X[2iₙ-1,j+1] = ∂buffer[iₙ]
            X[2iₙ,j+1] = buffer[iₙ]
        end
    end
    X
end

@generated function weight_mat(Xⁱ::AbstractMatrix, ::Val{N}) where N
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
@generated function weight_vec(Xⁱ::AbstractMatrix, ::Val{N}) where N
    quote
        weights = fill(zero(BigFloat), size(Xⁱ,2))
        for j ∈ 1:size(Xⁱ,2)
            @nexprs $N i -> begin
                weights[j] += normal_moment_general(Val{2(i-1)}()) * Xⁱ[2(i-1)+1,j]
            end
        end
        weights 
    end
end
function weight_vec(Xⁱ::AbstractMatrix)#Dynamic dispatch
    weight_vec(Xⁱ, Val{size(Xⁱ,1) ÷ 2}())::Vector{BigFloat}
end

function generate_weights(nodes)
    X = herm_design(nodes)
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


end # module
