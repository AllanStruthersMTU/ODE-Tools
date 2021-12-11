using Gridap
using Plots
using Gridap: ∇
  using Gridap: Δ
using LinearAlgebra
using Gridap: mean

# Define the manufactured solution
u(x) = 1000*(x[1]^4*(1-x[1])^4*x[2]^4*(1-x[2])^4*x[3]^4*(1-x[3])^4)

# Mesh generation
function oneproblem(domain::Tuple, n::Int, order::Int, β::Int, u::Function, D::Int)
  ∇u(x) = ∇(u)(x)
  Δu(x) = Δ(u)(x)
  f(x) = Δ(Δu)(x)

  partition = repeat([n], D)
  model = simplexify(CartesianDiscreteModel(domain, partition))

  # FESpaces
  V = TestFESpace(model, ReferenceFE(lagrangian, Float64, order), conformity=:L2)
  U = TrialFESpace(V)
  V² = MultiFieldFESpace([V,V])

  # Triangulations
  degree = 2*order
  Ω = Triangulation(model)
  Γ = BoundaryTriangulation(model)
  Λ = SkeletonTriangulation(model)

  # Measure
  dΩ = Measure(Ω, degree)
  dΓ = Measure(Γ, degree)
  dΛ = Measure(Λ, degree)

  # Normal Vector
  n_Γ = get_normal_vector(Γ)
  n_Λ = get_normal_vector(Λ)

  # Weak form components
  # β is the "strength" of the penalty term
  # 1/n is indicative of mesh size
  # αₖ ∈ ℝ.
  h = (1/n)^β
  αₖ = 5
  B_Γ(w, q) = ∫( - (∇(w)⋅n_Γ)*q )dΓ
  B_Λ(w, q) = ∫( - mean(∇(w))⊙jump(q*n_Λ) )dΛ  + ∫( - mean(∇(q))⊙jump(w*n_Λ) )dΛ
  B_Ω(w, q) = ∫( ∇(w)⊙ ∇(q) )dΩ
  J_Γ(w, q) = ∫( (αₖ/h)*w*q )dΓ
  J_Λ(w, q) = ∫( (αₖ/h)*jump(w*n_Λ)⊙jump(q*n_Λ) )dΛ
  # Bilinear forms
  M(w, q) = ∫(w*q)dΩ
  B(w, q) = B_Ω(w, q) + B_Λ(w, q) + B_Γ(w, q)
  J(w, q) = J_Γ(w, q) + J_Λ(w, q)
  L₁(w) = ∫( (∇u)⋅(w*n_Γ) - u*(∇(w)⋅n_Γ))dΓ
  L₂(q) = -1*∫( f*q )dΩ - (αₖ/h)*∫( u*q )dΓ


  # The mixed DG problem.
  a((u,v), (w,q)) = B(w, u) + M(v, w) + B(v, q) - J(u, q)
  l((w,q)) = L₂(q) + L₁(w)
  op = AffineFEOperator(a, l, V², V²)
  uh, vh = solve(op)

  # L2 error
  e = u-uh
  l2(u) = sqrt(sum(∫(u⊙u)*dΩ ))
  l2e = l2(e)

  uh, vh, l2e, cond(get_matrix(op), Inf)
end


N = [2,4,6,8]
l2errs = zeros(Float64, length(N))
cond_nos = zeros(Float64, length(N))
# penalty_strength = 3
# p=plot()
# for poly_order = 1:5
#   print("\nPolynomial order = "*string(poly_order)*"\n")
#   for n ∈ 1:length(N)
#     ..,..,l2err = oneproblem((0,1,0,1), N[n], poly_order, penalty_strength, u)
#     l2errs[n] = l2err
#     @show l2err
#   end
#   plot!(p, log10.(1 ./N), log10.(l2errs), label="k="*string(poly_order))
#   order = log.(l2errs[2:end]./l2errs[1:end-1])./log.(N[1:end-1]./N[2:end]);
#   @show order
# end

# p1 = plot()
# poly_order = 1
# for penalty_strength ∈ [1,3]
#   print("\nPenalty Strength = "*string(penalty_strength)*"\n")
#   for n ∈ 1:length(N)
#     ..,..,l2err = oneproblem((0,1,0,1), N[n], poly_order, penalty_strength, u)
#     l2errs[n] = l2err
#     @show l2err
#   end
#   plot!(p1, log10.(1 ./N), log10.(l2errs), label="i="*string(penalty_strength))
#   order = log.(l2errs[2:end]./l2errs[1:end-1])./log.(N[1:end-1]./N[2:end]);
#   @show order
# end


# xlabel!(p, "\${\\log(h)}\$")
# ylabel!(p, "\${\\log(||u - u_h||)}\$")
# xlabel!(p1, "\${\\log(h)}\$")
# ylabel!(p1, "\${\\log(||u - u_h||)}\$")

D=3
domain = Tuple(repeat([0,1],D))
for p ∈ 1:2
  print("\nPolynomial order "*string(p)*"\n")
  for n ∈ 1:length(N)
    ..,..,l2err, cond_no = oneproblem(domain, N[n], p, 1, u, D)
    l2errs[n] = l2err
    cond_nos[n] = cond_no
    @show l2err
  end
  global ooc = log.(l2errs[2:end]./l2errs[1:end-1])./log.(N[1:end-1]./N[2:end]);
  @show ooc
  print("\n")
end