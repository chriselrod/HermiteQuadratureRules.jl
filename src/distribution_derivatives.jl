



function fg(d::Distributions.Logistic, x)
    scaled = (x - d.μ)/2d.θ
    core = sech( scaled )^2 / 4d.θ
    core * -tanh(scaled) / d.θ, core
end
function fgw(d::Distributions.Logistic, x)
    scaled = (x - d.μ)/2d.θ
    core = sech( scaled )^2 / 4d.θ
    weighted_core = core*exp(x^2/2)
    x*weighted_core - weighted_core * tanh(scaled) / d.θ, weighted_core
end
function fw(d::Distributions.Logistic, x)
    scaled = (x - d.μ)/2d.θ
    sech( scaled )^2*exp(x^2/2) / 4d.θ
end