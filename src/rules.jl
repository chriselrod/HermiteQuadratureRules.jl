const hermite_nodes = (
    [0.0],
    [-1.7320508075688772, 0.0, 1.7320508075688772],
    [-4.184956017672732, -2.861279576057058, -1.7320508075688772, -0.7410953499945409, 0.0, 0.7410953499945409, 1.7320508075688772, 2.861279576057058, 4.184956017672732],
    [-6.36339449433637, -5.187016039913656, -4.184956017672732, -3.2053337944991944, -2.861279576057058, -2.5960831150492023, -1.7320508075688772, -1.230423634027306, -0.7410953499945409, 0.0, 0.7410953499945409, 1.230423634027306, 1.7320508075688772, 2.5960831150492023, 2.861279576057058, 3.2053337944991944, 4.184956017672732, 5.187016039913656, 6.36339449433637]
)

const hermite_nodes_diffs = (
    [0.0],
    [-1.7320508075688772, 1.7320508075688772],
    [-4.184956017672732, -2.861279576057058, -0.7410953499945409, 0.7410953499945409, 2.861279576057058, 4.184956017672732],
    [-6.36339449433637, -5.187016039913656, -3.2053337944991944, -2.5960831150492023, -1.230423634027306, 1.230423634027306, 2.5960831150492023, 3.2053337944991944, 5.187016039913656, 6.36339449433637]
)

using Base.Cartesian, Reduce
Reduce.Rational(false)


@generated function max_degree_poly(::Val{N}, l::Int, T = Int) where N
    quote
        j_0 = l
        s_0 = 0
        out = Vector{NTuple{N,T}}(0)
        @nloops $N i p -> begin
            0:j_{$N-p}
        end p -> begin
            s_{$N-p+1} = s_{$N-p} + i_p
            j_{$N-p+1} = l - s_{$N-p+1}
        end begin
            push!(out, (@ntuple $N j -> T(i_{j})) )
        end
        out
    end
end
function tupGridd(x::NTuple{N,T}) where {N,T}
    Expr(:call, :*, ntuple(i -> gen_diff(x[i], i), Val{N}())... )
end
gen_diff(v, i) = v > 0 ? :( $(Symbol('`' + i, "_", v)) - $(Symbol('`' + i, "_", v-1)) ) : Symbol('`' + i, "_", v)


function process_weights(x::Vector{NTuple{N,Int}}) where N
    expr = Expr(:call, :+, tupGridd(x[1]), tupGridd(x[2]) )
    for i in 3:length(x)
        push!(expr.args, tupGridd(x[i]) )
    end
    expr
end


p1 = max_degree_poly(Val(2), 2)
s1 = process_weights(p1)
rcall(s1, :expand)

function weights(::Val{N}, l::Int) where N
    p1 = max_degree_poly(Val{N}(), l)
    s1 = process_weights(p1)
    rcall(s1, :expand)::Expr
end

rules126 = readdlm("/home/celrod/Documents/notebooks/hermite_rules/rule_1_2_6_qr_0.csv", ',');
nodes126 = rules126[1,:]
weights126 = rules126[3,:];
using HermiteQuadratureRules
X = herm_design(nodes126);