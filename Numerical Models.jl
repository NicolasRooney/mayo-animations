# ============================================================================
#  Mayonnaise — Advanced Modelling (WI4204)
#    §1  Imports + package setup                          (single block)
#    §2  Material model: MayoParams, G* helpers
#    §3  Solvers
#         §3.1  solve_final_2d   – 2D conservative FV         (Section 6.1)
#         §3.2  solve_cyl_torsion – 3D axisymmetric torsion   (Section 6.2)
#         §3.3  bulk_kron         – Kronecker assembly         (report §6.1)
#         §3.4  solve_mol         – MethodOfLines pipeline     (report §5.2)
#    §4  Analytical references
#         §4.1  1D closed form (Eq. 130)
#         §4.2  Lumped dynamic moduli (Chapters 3–4)
#    §5  Verification suite (against §4.1 and the static-affine analytic)
#    §6  Frequency sweeps and report figures
#    §7  Animations
#    §8  Driver
# ============================================================================


# ─── §1  Imports ────────────────────────────────────────────────────────────
import Pkg
Pkg.activate(".")
Pkg.add(["CairoMakie", "ModelingToolkit", "MethodOfLines",
         "DomainSets", "NonlinearSolve"])
Pkg.instantiate()
Pkg.precompile()
using SparseArrays, LinearAlgebra, Printf, Base64
using CairoMakie
using ModelingToolkit, MethodOfLines, DomainSets, NonlinearSolve


# ─── §2  Material model ────────────────────────────────────────────────────
#
# A single MayoParams struct describes both the matrix (strong gel) and a
# soft circular inclusion (weak gel).  Default values follow report Table 2:
# the matrix is MayoB and the inclusion is MayoA1.  Geometry is reused by
# §3.2 with the convention  L → R (radius),  H → h (gap height).

mutable struct MayoParams
    rho::Float64           # density [kg/m^3]
    L::Float64             # x-extent (or radius R in §3.2)  [m]
    H::Float64             # y-extent (or gap height h)      [m]
    # Matrix
    G_strong::Float64      # plateau modulus [Pa]
    alpha_strong::Float64  # fractional order in (0,1)
    eta_strong::Float64    # generalised viscosity [Pa s^alpha]
    # Inclusion
    G_weak::Float64
    alpha_weak::Float64
    eta_weak::Float64
end

# Default instance: MayoB matrix with a soft MayoA1 inclusion.
const mp = MayoParams(1000.0, 0.05, 0.01,
                      2500.0, 0.45, 120.0,    # strong (MayoB)
                       255.0, 0.40,  10.0)    # weak   (MayoA1)

# Inclusion geometry (single source of truth — edit here if you want to
# change the test case for both the solver and the animation overlay).
const INCL_CX     = mp.L / 2
const INCL_CY     = mp.H / 2
const INCL_RADIUS = 0.005   # 5 mm

"""
    get_Gstar_hetero(x, y, w, p) -> ComplexF64

Heterogeneous fractional KV complex modulus.  A circular inclusion of
radius `INCL_RADIUS` centred at `(INCL_CX, INCL_CY)` carries the weak-gel
fields of `p`; everything else carries the strong-gel fields.
"""
function get_Gstar_hetero(x, y, w, p::MayoParams)
    dist = sqrt((x - INCL_CX)^2 + (y - INCL_CY)^2)
    if dist < INCL_RADIUS
        return p.G_weak   + p.eta_weak   * (im * w)^p.alpha_weak
    else
        return p.G_strong + p.eta_strong * (im * w)^p.alpha_strong
    end
end

"""
    Gstar_uniform(p, w; model=:fkv, eta_kv=150.0) -> ComplexF64

Spatially uniform complex modulus, parameterised by the strong-gel fields.
Used by the constant-coefficient solvers (`bulk_kron`, `solve_mol`) and
by the analytical comparisons in §4.

`eta_kv` is the classical KV viscosity in Pa·s used only when `model=:kv`
(the fractional `eta_strong` has units Pa·s^α and is not dimensionally
the same quantity).
"""
function Gstar_uniform(p::MayoParams, w::Real;
                       model::Symbol = :fkv, eta_kv::Real = 150.0)
    model === :elastic && return complex(p.G_strong, 0.0)
    model === :kv      && return complex(p.G_strong, w * eta_kv)
    model === :fkv     && return p.G_strong + p.eta_strong * (im * w)^p.alpha_strong
    error("model must be :elastic, :kv or :fkv")
end


# ─── §3.1  solve_final_2d ──────────────────────────────────────────────────
"""
    solve_final_2d(p, w; Nx=80, Ny=40, tau0=50.0,
                   Gstar_field = (x,y,w) -> get_Gstar_hetero(x,y,w,p))

Conservative finite-volume solver for the harmonic 2D shear BVP
(report Eq. 165):

    ∇·(G*(x,y;ω) ∇U₀) + ρω² U₀ = 0          on (0,L) × (0,H)
    U₀(x,0) = 0                              bottom (Dirichlet)
    G*(x,H;ω) ∂_y U₀(x,H) = τ₀              top (Neumann, stress-controlled)
    ∂_x U₀(0,y) = ∂_x U₀(L,y) = 0            sides (traction-free)

Five-point stencil with arithmetic face-averaged moduli — works across
sharp jumps in G*(x,y).  Returns  `(U_complex_2D, xs, ys)`.

The optional `Gstar_field` lets the caller substitute any complex modulus
field — used in §6 to sweep elastic/KV/FKV models through the same
solver code path.
"""
function solve_final_2d(p::MayoParams, w::Real;
                        Nx::Int = 80, Ny::Int = 40, tau0::Real = 50.0,
                        Gstar_field = (x, y, w) -> get_Gstar_hetero(x, y, w, p))
    hx, hy   = p.L / Nx, p.H / Ny
    nx, ny   = Nx + 1, Ny + 1
    xs       = range(0.0, p.L; length = nx)
    ys       = range(0.0, p.H; length = ny)
    N        = nx * ny
    idx(i, j) = (j - 1) * nx + i

    Gmat = Matrix{ComplexF64}(undef, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        Gmat[i, j] = Gstar_field(xs[i], ys[j], w)
    end
    face(a, b) = 0.5 * (a + b)

    II = Int[]; JJ = Int[]; VV = ComplexF64[]
    b  = zeros(ComplexF64, N)
    addentry!(r, c, v) = (push!(II, r); push!(JJ, c); push!(VV, v))

    @inbounds for j in 1:ny, i in 1:nx
        r = idx(i, j)

        # Bottom: Dirichlet U₀ = 0
        if j == 1
            addentry!(r, r, 1.0 + 0im)
            continue
        end

        Gc = Gmat[i, j]
        # Face-averaged moduli; sides mirror so that ∂_x U₀ = 0
        GE = i < nx ? face(Gc, Gmat[i + 1, j]) : face(Gc, Gmat[i - 1, j])
        GW = i > 1  ? face(Gc, Gmat[i - 1, j]) : face(Gc, Gmat[i + 1, j])
        GS = face(Gc, Gmat[i, j - 1])
        GN = j < ny ? face(Gc, Gmat[i, j + 1]) : Gc

        iE = i < nx ? idx(i + 1, j) : idx(i - 1, j)
        iW = i > 1  ? idx(i - 1, j) : idx(i + 1, j)
        jS = idx(i, j - 1)

        addentry!(r, iE, GE / hx^2)
        addentry!(r, iW, GW / hx^2)
        addentry!(r, jS, GS / hy^2)

        diag = -(GE + GW) / hx^2 - (GN + GS) / hy^2 + p.rho * w^2

        if j < ny
            addentry!(r, idx(i, j + 1), GN / hy^2)
            addentry!(r, r, diag)
        else
            # Top: ghost-elimination of G* ∂_y U₀ = τ₀ at y = H
            #   U_{ny+1} = U_{ny-1} + 2 hy τ₀ / Gc
            addentry!(r, jS, GN / hy^2)
            addentry!(r, r,  diag)
            b[r] -= (GN / hy^2) * (2 * hy * tau0 / Gc)
        end
    end

    A = sparse(II, JJ, VV, N, N)
    return reshape(A \ b, nx, ny), xs, ys
end


# ─── §3.2  solve_cyl_torsion ───────────────────────────────────────────────
"""
    solve_cyl_torsion(p, w; NR=80, NZ=40, theta0=1e-3,
                      Gstar_field=nothing)

Steady-state harmonic torsion on the meridian section (0,R) × (0,h) of
the rheometer (Section 6.2 of the report):

    (1/r²) ∂_r[r² G* (∂_r U₀ − U₀/r)] + ∂_z(G* ∂_z U₀) + ρω² U₀ = 0
    U₀(r, 0) = 0                         bottom (fixed plate)   Eq. (168)
    U₀(r, h) = r θ₀                      top (rotating plate)   Eq. (169)
    G* (∂_r U₀ − U₀/r) = 0 at r=R        lateral traction-free  Eq. (172)
    U₀(0, z) = 0                         axis regularity        Eq. (173)

Note: Eq. (172) is the traction-free condition for purely azimuthal motion
(∂_r U₀ − U₀/r = 0), implemented here via ghost cells. The conservative
shear-flux r-stencil below coincides with the divergence operator of Eq. (178)
ONLY for spatially uniform G*. For a heterogeneous G*(r,z) it omits a
−∂_r(G*) U₀ / r term, so this solver is validated/used for homogeneous moduli
only (which is all the report's 3D results require).
...
"""
function solve_cyl_torsion(p::MayoParams, w::Real;
                           NR::Int = 80, NZ::Int = 40,
                           theta0::Real = 1e-3,
                           Gstar_field = nothing)
    R, h     = p.L, p.H
    hr, hz   = R / NR, h / NZ
    nr, nz   = NR + 1, NZ + 1
    rs       = range(0.0, R; length = nr)
    zs       = range(0.0, h; length = nz)
    N        = nr * nz
    idx(i, j) = (j - 1) * nr + i

    if Gstar_field === nothing
        Gunif = Gstar_uniform(p, w; model = :fkv)
        Gstar_field = (r, z, w) -> Gunif
    end

    Gmat = Matrix{ComplexF64}(undef, nr, nz)
    @inbounds for j in 1:nz, i in 1:nr
        Gmat[i, j] = Gstar_field(rs[i], zs[j], w)
    end
    face(a, b) = 0.5 * (a + b)

    II = Int[]; JJ = Int[]; VV = ComplexF64[]
    b  = zeros(ComplexF64, N)
    addentry!(r, c, v) = (push!(II, r); push!(JJ, c); push!(VV, v))

    @inbounds for j in 1:nz, i in 1:nr
        row = idx(i, j)

        # Dirichlet rows: axis, bottom, top.
        if i == 1
            addentry!(row, row, 1.0 + 0im); continue
        end
        if j == 1
            addentry!(row, row, 1.0 + 0im); continue
        end
        if j == nz
            addentry!(row, row, 1.0 + 0im)
            b[row] = rs[i] * theta0
            continue
        end

        Gc = Gmat[i, j]
        ri = rs[i]
        if i < nr
            GE = face(Gc, Gmat[i + 1, j]);  iE = idx(i + 1, j)
        else
            # Lateral r = R, traction-free: ghost  U_{nr+1} = U_{nr-1} + 2hr U_{nr,j}/R
            GE = Gc;                         iE = idx(i - 1, j)
        end
        GW = i > 1 ? face(Gc, Gmat[i - 1, j]) : Gc
        iW = i > 1 ? idx(i - 1, j) : idx(i + 1, j)
        GN = face(Gc, Gmat[i, j + 1]);       jN = idx(i, j + 1)
        GS = face(Gc, Gmat[i, j - 1]);       jS = idx(i, j - 1)

        # z-stencil (standard 1D conservative)
        addentry!(row, jN, GN / hz^2)
        addentry!(row, jS, GS / hz^2)
        diag_z = -(GN + GS) / hz^2

        # r-stencil (Fully conservative flux-difference)
        # Represents the discrete volume integral of (1/r^2) * ∂_r [ r^2 G* (∂_r U - U/r) ]
        rE = ri + hr / 2
        rW = ri - hr / 2

        cE = (GE / (ri^2 * hr)) * (rE^2 / hr - rE / 2)
        cW = (GW / (ri^2 * hr)) * (rW^2 / hr + rW / 2)

        diag_r = -(GE / (ri^2 * hr)) * (rE^2 / hr + rE / 2) -
                  (GW / (ri^2 * hr)) * (rW^2 / hr - rW / 2)

        addentry!(row, iE, cE)
        addentry!(row, iW, cW)

        # Lateral ghost contribution at r = R: U_{nr+1} = U_{nr-1} + 2hr U_{nr}/R
        # We simply add the folded cE contribution directly to the diagonal
        if i == nr
            diag_r += (2 * hr / R) * cE
        end

        addentry!(row, row, diag_r + diag_z + p.rho * w^2)
    end

    A = sparse(II, JJ, VV, N, N)
    return reshape(A \ b, nr, nz), rs, zs
end


# ─── §3.3  bulk_kron ───────────────────────────────────────────────────────
"""
    bulk_kron(p, w; Nx=60, Ny=60, model=:fkv)

Bulk Laplacian-plus-mass operator in Kronecker form for a uniform medium:

    A_bulk = I ⊗ D2x + D2y ⊗ I + k² I ,    k² = ρω² / G*

This is the assembly form used in the report's derivation of the 2D
discretisation (column-major indexing → A_xx = I⊗D2x, A_yy = D2y⊗I).
Returned matrix has no boundary conditions imposed; `solve_final_2d`
is the production solver.  Useful for the report figure showing the
algebraic structure and as a quick check of the bulk stencil.
"""
function bulk_kron(p::MayoParams, w::Real;
                   Nx::Int = 60, Ny::Int = 60, model::Symbol = :fkv)
    hx, hy = p.L / Nx, p.H / Ny
    nx, ny = Nx + 1, Ny + 1
    D2(n, h) = spdiagm(-1 => fill(1/h^2, n - 1),
                        0 => fill(-2/h^2, n),
                        1 => fill(1/h^2, n - 1))
    Ax, Ay = D2(nx, hx), D2(ny, hy)
    k2 = p.rho * w^2 / Gstar_uniform(p, w; model = model)
    return kron(sparse(I, ny, ny), Ax) + kron(Ay, sparse(I, nx, nx)) + k2 * I
end


# ─── §3.4  solve_mol ───────────────────────────────────────────────────────
"""
    solve_mol(p, w; N=40, model=:fkv, tau0=50.0)

Same 2D Helmholtz BVP, solved via the MethodOfLines.jl symbolic pipeline.
Split into real/imaginary parts: G* = a + ib,  U₀ = Ur + i Ui.
"""
function solve_mol(p::MayoParams, w::Real;
                   N::Int = 40, model::Symbol = :fkv, tau0::Real = 50.0)
    Gs = Gstar_uniform(p, w; model = model)
    a, b = real(Gs), imag(Gs)

    @parameters x y
    @variables Ur(..) Ui(..)
    Dx = Differential(x);   Dy = Differential(y)
    Dxx = Differential(x)^2; Dyy = Differential(y)^2
    rw2 = p.rho * w^2

    eqs = [a*(Dxx(Ur(x,y)) + Dyy(Ur(x,y))) - b*(Dxx(Ui(x,y)) + Dyy(Ui(x,y))) + rw2*Ur(x,y) ~ 0,
           b*(Dxx(Ur(x,y)) + Dyy(Ur(x,y))) + a*(Dxx(Ui(x,y)) + Dyy(Ui(x,y))) + rw2*Ui(x,y) ~ 0]

    # Decouple the top boundary condition to avoid MOL symbolic tearing errors
    denom = a^2 + b^2
    bcs = [Ur(x, 0.0)   ~ 0.0,
           Ui(x, 0.0)   ~ 0.0,
           Dy(Ur(x, p.H)) ~ a * tau0 / denom,
           Dy(Ui(x, p.H)) ~ -b * tau0 / denom,
           Dx(Ur(0.0, y)) ~ 0.0, Dx(Ui(0.0, y)) ~ 0.0,
           Dx(Ur(p.L, y)) ~ 0.0, Dx(Ui(p.L, y)) ~ 0.0]

    domains = [x ∈ Interval(0.0, p.L), y ∈ Interval(0.0, p.H)]
    @named pdesys = PDESystem(eqs, bcs, domains, [x, y], [Ur(x, y), Ui(x, y)])

    disc = MOLFiniteDifference([x => N, y => N], nothing)   # steady BVP
    prob = discretize(pdesys, disc)
    return NonlinearSolve.solve(prob, NewtonRaphson())
end


# ─── §4.1  1D analytic reference (Eq. 130) ─────────────────────────────────
"""
    analytic_U0_1d(p, w, y; tau0=50.0, model=:fkv)

Closed-form 1D Helmholtz solution along y (Eq. 130).  With x-uniform
forcing the 2D field is x-independent and every vertical slice of
`solve_final_2d` must equal this expression — used in §5.
"""
function analytic_U0_1d(p::MayoParams, w::Real, y;
                        tau0::Real = 50.0, model::Symbol = :fkv)
    Gs    = Gstar_uniform(p, w; model = model)
    kstar = w * sqrt(p.rho / Gs)
    return @. tau0 / (Gs * kstar * cos(kstar * p.H)) * sin(kstar * y)
end


# ─── §4.2  Lumped dynamic moduli (Chapters 3–4) ────────────────────────────
# These are the four constitutive laws compared in the report's dynamic
# moduli figure.  Free-standing parameters (eta_classical, tauM, ...) are
# illustrative — not tied to the fitted MayoParams values.

g_kv(p::MayoParams, w; eta_kv = 150.0)       = complex(p.G_strong, w * eta_kv)
g_fkv(p::MayoParams, w)                      = Gstar_uniform(p, w; model = :fkv)
g_maxwell(p::MayoParams, w; tauM = 0.1)      = p.G_strong * (im*w*tauM) / (1 + im*w*tauM)
function g_gen_maxwell(p::MayoParams, w;
                       Ge = p.G_strong,            # Added finite low-frequency elastic plateau
                       G1 = 0.5*p.G_strong, t1 = 0.05,
                       G2 = 0.5*p.G_strong, t2 = 0.5)
    # Includes Ge so that G' does not drop to zero at low frequencies
    Ge + G1*(im*w*t1)/(1+im*w*t1) + G2*(im*w*t2)/(1+im*w*t2)
end

"""
    Gstar_model(p, w; model=:fkv, alpha=p.alpha_strong, eta_kv=150.0) -> ComplexF64

Complex modulus for the three constitutive laws, with an explicit `alpha`
override so the fractional order can be swept independently of the struct.
"""
function Gstar_model(p::MayoParams, w::Real;
                     model::Symbol = :fkv, alpha::Real = p.alpha_strong,
                     eta_kv::Real = 150.0)
    model === :elastic && return complex(p.G_strong, 0.0)
    model === :kv      && return complex(p.G_strong, w * eta_kv)
    model === :fkv     && return p.G_strong + p.eta_strong * (im * w)^alpha
    error("model must be :elastic, :kv or :fkv")
end

"""
    tip_response_1d(p, w; model, alpha, eta_kv, tau0) -> ComplexF64

Closed-form tip amplitude U₀(L) of the 1D stress-driven shear bar of Section 5
(length p.L), from Eq. (152):

    U₀(L) = τ₀ / (G* k*) · tan(k* L),   k* = ω √(ρ / G*).
"""
function tip_response_1d(p::MayoParams, w::Real;
                         model::Symbol = :fkv, alpha::Real = p.alpha_strong,
                         eta_kv::Real = 150.0, tau0::Real = 50.0)
    Gs    = Gstar_model(p, w; model = model, alpha = alpha, eta_kv = eta_kv)
    kstar = w * sqrt(p.rho / Gs)
    return tau0 / (Gs * kstar) * tan(kstar * p.L)
end

"""
    plot_tip_frequency_response(p; file=...)

Reproduces report Figure 27: |U₀(L)| vs ω for the classical KV bar and two
fractional KV bars, with the undamped elastic resonance ω₁ = πc/(2L) marked.
The elastic curve itself is omitted (it diverges at ω₁); the point of the figure
is that dissipation turns that divergence into a finite, smoothed peak.
"""
function plot_tip_frequency_response(p::MayoParams = mp;
                                     ws = exp10.(range(-0.3, 2.4; length = 400)),
                                     file::String = "tip_frequency_response.png")
    kv  = [abs(tip_response_1d(p, w; model = :kv))               for w in ws]
    f45 = [abs(tip_response_1d(p, w; model = :fkv, alpha = 0.45)) for w in ws]
    f20 = [abs(tip_response_1d(p, w; model = :fkv, alpha = 0.20)) for w in ws]
    w1  = π * sqrt(p.G_strong / p.rho) / (2 * p.L)

    fig = Figure(size = (950, 560))
    ax  = Axis(fig[1, 1]; xscale = log10, yscale = log10,
               xlabel = "ω (rad/s)", ylabel = "|U₀(L)| (μm)",
               title  = "Tip displacement frequency response — resonance smoothing by fractional α")
    lines!(ax, ws, kv  .* 1e6; color = :dodgerblue, linewidth = 2,                       label = "Classical KV")
    lines!(ax, ws, f45 .* 1e6; color = :crimson,    linewidth = 2, linestyle = :dash,    label = "Fractional KV (α = 0.45)")
    lines!(ax, ws, f20 .* 1e6; color = :seagreen,   linewidth = 2, linestyle = :dashdot, label = "Fractional KV (α = 0.20)")
    vlines!(ax, [w1]; color = :black, linestyle = :dot,
            label = @sprintf("Elastic resonance ω₁ = %.0f rad/s", w1))
    axislegend(ax; position = :lt)
    save(file, fig); @info "wrote $file"
    @printf("Elastic resonance ω₁ ≈ %.1f rad/s\n", w1)
    return fig
end

# ─── §5  Verification suite ────────────────────────────────────────────────

"""
    verify_solve_final_2d(; w=10.0, tau0=50.0, Nx=60, Ny=60)

With a homogeneous `MayoParams` (strong ≡ weak), the 2D solver must be
x-uniform and match Eq. (130) on every slice.  Also runs a Ny refinement
study to confirm O(h²) convergence.
"""
function verify_solve_final_2d(; w::Real = 10.0, tau0::Real = 50.0,
                                Nx::Int = 60, Ny::Int = 60)
    p = MayoParams(1000.0, 0.05, 0.01,
                   2500.0, 0.45, 120.0,
                   2500.0, 0.45, 120.0)        # weak ≡ strong → uniform

    U, xs, ys = solve_final_2d(p, w; Nx = Nx, Ny = Ny, tau0 = tau0)
    Ua        = analytic_U0_1d(p, w, collect(ys); tau0 = tau0)

    x_unif  = maximum(abs.(U .- U[1:1, :])) / maximum(abs.(U))
    rel_err = maximum(maximum(abs.(U[i, :] .- Ua)) / maximum(abs.(Ua))
                      for i in 1:size(U, 1))

    @printf("[Test 1]  solve_final_2d  (constant G*,  ω = %.1f rad/s, Nx = Ny = %d)\n",
            w, Nx)
    @printf("          G*           = %s\n", string(Gstar_uniform(p, w)))
    @printf("          max |U|      = %.4f um\n", maximum(abs.(U)) * 1e6)
    @printf("          x-uniformity = %.2e   (machine eps expected)\n", x_unif)
    @printf("          max rel err  = %.2e   (O(h²) expected)\n",         rel_err)

    println("          grid refinement:")
    @printf("            %4s   %12s\n", "Ny", "rel err")
    errs = Float64[]; Nys = [20, 40, 80, 160]
    for Ny_ in Nys
        U_, _, ys_ = solve_final_2d(p, w; Nx = 20, Ny = Ny_, tau0 = tau0)
        Ua_  = analytic_U0_1d(p, w, collect(ys_); tau0 = tau0)
        mid  = size(U_, 1) ÷ 2 + 1
        err  = maximum(abs.(U_[mid, :] .- Ua_)) / maximum(abs.(Ua_))
        push!(errs, err)
        @printf("            %4d   %12.3e\n", Ny_, err)
    end
    rates = log2.(errs[1:end-1] ./ errs[2:end])
    @printf("          observed rates: %s\n", string(round.(rates, digits = 3)))
    return errs
end

"""
    verify_solve_cyl_torsion(; theta0=1e-3)

Static-limit and lateral-BC checks for the cylindrical torsion solver.
"""
function verify_solve_cyl_torsion(; theta0::Real = 1e-3)
    p = MayoParams(1000.0, 0.020, 0.001,
                   2500.0, 0.45, 120.0,
                   2500.0, 0.45, 120.0)        # uniform

    G0     = p.G_strong
    Gfield = (r, z, w) -> complex(G0, 0.0)     # real, constant ⇒ static
    w_stat = 1e-8

    println("[Test 2]  solve_cyl_torsion  (static limit, η_α = 0)")
    @printf("          %4s %4s   %12s\n", "NR", "NZ", "rel err")
    errs = Float64[]; hs = Float64[]
    for (NR, NZ) in [(20,10), (40,20), (80,40), (160,80)]
        U, rs, zs = solve_cyl_torsion(p, w_stat;
                                      NR = NR, NZ = NZ, theta0 = theta0,
                                      Gstar_field = Gfield)
        Uexact = [rs[i] * theta0 * zs[j] / p.H
                  for i in 1:length(rs), j in 1:length(zs)]
        rel = maximum(abs.(U .- Uexact)) / maximum(abs.(Uexact))
        push!(errs, rel); push!(hs, p.L / NR)
        @printf("          %4d %4d   %12.3e\n", NR, NZ, rel)
    end
    println("          (errors at machine epsilon: the affine solution is bilinear)")

    # Test 3: lateral traction-free BC at r = R for the fractional KV modulus
    w_dyn = 10.0
    Gdyn  = (r, z, w) -> Gstar_uniform(p, w; model = :fkv)
    NR, NZ = 200, 80
    U, rs, zs = solve_cyl_torsion(p, w_dyn;
                                  NR = NR, NZ = NZ, theta0 = theta0,
                                  Gstar_field = Gdyn)
    hr        = p.L / NR
    Uedge     = U[end, :]
    dUdr_R    = (U[end, :] .- U[end-1, :]) ./ hr
    residual  = dUdr_R .- Uedge ./ p.L
    rel_trac  = maximum(abs.(residual) ./ max.(abs.(Uedge ./ p.L), 1e-30))
    @printf("[Test 3]  lateral traction-free residual  (ω = %.1f rad/s)  = %.2e\n",
            w_dyn, rel_trac)
    return errs, rel_trac
end


# ─── §6  Frequency sweeps and report figures ───────────────────────────────

"""
    frequency_sweep_hetero(p; ws=...) -> (ws, amps_um)

Sweep the heterogeneous (inclusion) model over ω and record the top-edge
amplitude at x = L/2 in micrometres.
"""
function frequency_sweep_hetero(p::MayoParams;
                                ws = exp10.(range(1.5, 2.7; length = 100)))
    amps = Float64[]
    for w in ws
        U, _, _ = solve_final_2d(p, w; Nx = 60, Ny = 30)
        push!(amps, abs(U[end ÷ 2, end]) * 1e6)
    end
    return ws, amps
end

"""
    plot_frequency_sweep(p; file=...)

Frequency-sweep figure for the heterogeneous model.
"""
function plot_frequency_sweep(p::MayoParams = mp;
                              file::String = "frequency_sweep.png")
    ws_vals, amps = frequency_sweep_hetero(p)
    fig = Figure(size = (800, 500))
    ax  = Axis(fig[1, 1],
               title  = "Frequency response (heterogeneous fractional KV)",
               xlabel = "Frequency ω (rad/s)",
               ylabel = "Top-edge displacement (μm)",
               xscale = log10)
    lines!(  ax, ws_vals, amps; linewidth = 3, color = :crimson)
    scatter!(ax, ws_vals, amps; markersize = 8, color = :crimson)
    save(file, fig); @info "wrote $file"
    @printf("Resonance peak at ω ≈ %.1f rad/s\n", ws_vals[argmax(amps)])
    return fig
end

"""
    plot_dynamic_moduli(p; file=...)

Storage and loss moduli for the four constitutive laws of Chapters 3–4
(report figure illustrating the model hierarchy).
"""
function plot_dynamic_moduli(p::MayoParams = mp;
                             ws = exp10.(range(-1, 3; length = 100)),
                             file::String = "dynamic_moduli.png")
    fig = Figure(size = (1000, 800))
    ax1 = Axis(fig[1, 1]; title = "Storage modulus G'",
               xscale = log10, yscale = log10, ylabel = "Pa")
    ax2 = Axis(fig[2, 1]; title = "Loss modulus G''",
               xscale = log10, yscale = log10,
               xlabel = "ω (rad/s)", ylabel = "Pa")
    models = [("KV",            (w -> g_kv(p, w)),          :blue),
              ("Maxwell",       (w -> g_maxwell(p, w)),     :red),
              ("Gen. Maxwell",  (w -> g_gen_maxwell(p, w)), :green),
              ("Fractional KV", (w -> g_fkv(p, w)),         :orange)]
    for (name, func, clr) in models
        val = [func(w) for w in ws]
        lines!(ax1, ws, real.(val); label = name, color = clr, linewidth = 2.5)
        lines!(ax2, ws, imag.(val); label = name, color = clr, linewidth = 2.5)
    end
    axislegend(ax1; position = :lt)
    save(file, fig); @info "wrote $file"
    return fig
end

"""
    comparison_figure(p; file=...)

Two-panel report figure: (i) tip displacement vs frequency for elastic,
classical KV, fractional KV (resonance smoothing); (ii) heterogeneous
|U₀(x, y)| at ω = 10 rad/s with the inclusion outlined.
"""
function comparison_figure(p::MayoParams = mp;
                           file::String = "comparison.png")
    ws = exp10.(range(0, 2.7; length = 90))
    tip(field) = [abs(solve_final_2d(p, w; Nx = 70, Ny = 70,
                                     Gstar_field = field)[1][36, end])
                  for w in ws]
    el = tip((x, y, w) -> Gstar_uniform(p, w; model = :elastic))
    kv = tip((x, y, w) -> Gstar_uniform(p, w; model = :kv))
    fk = tip((x, y, w) -> Gstar_uniform(p, w; model = :fkv))
    w1 = π * sqrt(p.G_strong / p.rho) / (2 * p.H)

    fig = Figure(size = (1500, 460))
    ax1 = Axis(fig[1, 1]; xscale = log10, yscale = log10,
               xlabel = "ω (rad/s)", ylabel = "|U₀(top)| (μm)",
               title  = "Resonance smoothing")
    for (d, l) in ((el, "elastic"), (kv, "classical KV"), (fk, "fractional KV"))
        lines!(ax1, ws, d .* 1e6; label = l, linewidth = 2)
    end
    vlines!(ax1, [w1]; color = :black, linestyle = :dot)
    axislegend(ax1; position = :lb)

    U, xs, ys = solve_final_2d(p, 10.0; Nx = 90, Ny = 90)
    ax2 = Axis(fig[1, 2]; aspect = DataAspect(),
               xlabel = "x (mm)", ylabel = "y (mm)",
               title  = "Heterogeneous |U₀(x,y)|, ω = 10 rad/s")
    hm = heatmap!(ax2, xs .* 1e3, ys .* 1e3, abs.(U) .* 1e6; colormap = :magma)
    Colorbar(fig[1, 3], hm; label = "|U₀| (μm)")
    # inclusion outline
    θ = range(0, 2π; length = 120)
    lines!(ax2, INCL_CX*1e3 .+ INCL_RADIUS*1e3 .* cos.(θ),
                INCL_CY*1e3 .+ INCL_RADIUS*1e3 .* sin.(θ);
           color = :black, linestyle = :dash, linewidth = 1.6)
    save(file, fig); @info "wrote $file"
    return fig
end

"""
    plot_verification_figures(; file=...)

Companion figure to the §5 verification suite.
"""
function plot_verification_figures(; file::String = "verification.png")
    # Constant-G* MayoParams (heterogeneity off) for the 2D test
    p_uniform = MayoParams(1000.0, 0.05, 0.01,
                           2500.0, 0.45, 120.0,
                           2500.0, 0.45, 120.0)
    w, tau0 = 10.0, 50.0
    U, xs, ys = solve_final_2d(p_uniform, w; Nx = 60, Ny = 60, tau0 = tau0)
    Ua = analytic_U0_1d(p_uniform, w, collect(ys); tau0 = tau0)

    fig = Figure(size = (1300, 900))

    # Top-left: Test 1 mid-slice vs 1D analytic
    ax1 = Axis(fig[1, 1]; xlabel = "y (mm)", ylabel = "displacement (μm)",
               title = "Test 1: 2D mid-slice vs 1D analytic, ω = $(w) rad/s")
    i_mid = size(U, 1) ÷ 2 + 1
    lines!(   ax1, ys.*1e3, real.(U[i_mid, :]).*1e6; color = :dodgerblue, linewidth = 2, label = "Re U₀ (FV)")
    lines!(   ax1, ys.*1e3, imag.(U[i_mid, :]).*1e6; color = :orange,     linewidth = 2, label = "Im U₀ (FV)")
    lines!(   ax1, ys.*1e3, abs.( U[i_mid, :]).*1e6; color = :black,      linewidth = 2, label = "|U₀| (FV)")
    scatter!( ax1, ys.*1e3, real.(Ua).*1e6;          color = :dodgerblue, markersize = 7, marker = :circle, label = "Re U₀ (analytic)")
    scatter!( ax1, ys.*1e3, imag.(Ua).*1e6;          color = :orange,     markersize = 7, marker = :rect,   label = "Im U₀ (analytic)")
    axislegend(ax1; position = :lt)

    # Static-limit cylindrical solution
    p_cyl = MayoParams(1000.0, 0.020, 0.001,
                       2500.0, 0.45, 120.0,
                       2500.0, 0.45, 120.0)
    theta0 = 1e-3
    Gfield = (r, z, w) -> complex(p_cyl.G_strong, 0.0)
    U2, rs, zs = solve_cyl_torsion(p_cyl, 1e-8; NR = 80, NZ = 40,
                                   theta0 = theta0, Gstar_field = Gfield)
    Uex = [rs[i] * theta0 * zs[j] / p_cyl.H
           for i in 1:length(rs), j in 1:length(zs)]

    ax2 = Axis(fig[1, 2]; xlabel = "z (mm)", ylabel = "U₀ (μm)",
               title = "Test 2: static cylindrical solution vs R·θ₀·z/h")
    lines!(   ax2, zs.*1e3, real.(U2[end,     :]).*1e6; color = :dodgerblue, linewidth = 2, label = "solver, r=R")
    scatter!( ax2, zs.*1e3, real.(Uex[end,    :]).*1e6; color = :dodgerblue, markersize = 7, marker = :circle, label = "exact, r=R")
    lines!(   ax2, zs.*1e3, real.(U2[end÷2+1, :]).*1e6; color = :crimson,    linewidth = 2, label = "solver, r=R/2")
    scatter!( ax2, zs.*1e3, real.(Uex[end÷2+1,:]).*1e6; color = :crimson,    markersize = 7, marker = :rect,   label = "exact, r=R/2")
    axislegend(ax2; position = :lt)

    # Finite-omega cylindrical field
    Gdyn = (r, z, w) -> Gstar_uniform(p_cyl, w; model = :fkv)
    U3, rs3, zs3 = solve_cyl_torsion(p_cyl, 10.0; NR = 120, NZ = 60,
                                     theta0 = theta0, Gstar_field = Gdyn)
    gb  = fig[2, 1:2] = GridLayout()
    ax3 = Axis(gb[1, 1]; xlabel = "r (mm)", ylabel = "z (mm)",
               title = "|U₀(r,z)|  (fractional KV, ω = 10 rad/s)")
    hm  = heatmap!(ax3, rs3 .* 1e3, zs3 .* 1e3, abs.(U3) .* 1e6; colormap = :viridis)
    Colorbar(gb[1, 2], hm; label = "|U₀| (μm)")
    rowsize!(fig.layout, 2, Relative(0.42))

    save(file, fig); @info "wrote $file"
    return fig
end


# ─── §7  Animations ────────────────────────────────────────────────────────

"""
    animate_field(U, xs, ys, w; file=..., nframes=60, periods=1, incl=nothing,
                  title=..., cmap=:balance)

Heatmap animation of  u(x,y,t) = Re{U(x,y) e^{iωt}}  over `periods` forcing
periods.  `incl = (cx, cy, r)` overlays a dashed circle.
"""
function animate_field(U, xs, ys, w;
                       file::String = "shear.mp4", nframes::Int = 60,
                       periods::Int = 1, incl = nothing,
                       title::String = "2D visco-elastic shear",
                       cmap = :balance)
    amp = maximum(abs, U) * 1e6
    fig = Figure(size = (720, 680))
    ax  = Axis(fig[1, 1]; aspect = DataAspect(),
               xlabel = "x (mm)", ylabel = "y (mm)")
    ph  = Observable(0.0)
    fld = @lift(real.(U .* cis($ph)) .* 1e6)
    hm  = heatmap!(ax, xs.*1e3, ys.*1e3, fld;
                   colormap = cmap, colorrange = (-amp, amp))
    Colorbar(fig[1, 2], hm; label = "antiplane displacement u (μm)")
    if incl !== nothing
        cx, cy, r = incl
        θ = range(0, 2π; length = 120)
        lines!(ax, cx*1e3 .+ r*1e3 .* cos.(θ),
                   cy*1e3 .+ r*1e3 .* sin.(θ);
               color = :black, linestyle = :dash, linewidth = 1.6)
    end
    record(fig, file, range(0, 2π*periods; length = nframes); framerate = 18) do t
        ph[] = t
        ax.title = @sprintf("%s  (ω = %.0f rad/s, phase %.0f°)",
                            title, w, rad2deg(t))
    end
    @info "wrote $file"
    return file
end

"""
    animate_shear_3d(U, xs, ys, w; file=..., frames=60)

3D surface animation of the antiplane shear field.
"""
function animate_shear_3d(U, xs, ys, w;
                          file::String = "mayo_phys_wobble.mp4",
                          frames::Int = 60)
    fig = Figure(size = (800, 600))
    ax  = Axis3(fig[1, 1];
                title    = "Antiplane shear  (ω = $w rad/s)",
                xlabel   = "x (mm)", ylabel = "y (mm)",
                zlabel   = "u (μm)",
                azimuth  = 0.8π, elevation = 0.1π)
    ph  = Observable(0.0)
    fld = @lift(real.(U .* cis($ph)) .* 1e6)
    surface!(ax, xs.*1e3, ys.*1e3, fld;
             colormap = :cividis, shading = NoShading)
    amp = maximum(abs.(U) * 1e6) * 1.5
    zlims!(ax, -amp, amp)
    record(fig, file, range(0, 2π; length = frames); framerate = 20) do t
        ph[] = t
    end
    @info "wrote $file"
    return file
end

"""
    animate_cylindrical_twist_3d(U, rs, zs, w; file=..., frames=60)

Solid 3D Visualization: Maps the outer boundary of the sample into a
full 3D cylindrical shell. The shell is physically twisted by the
azimuthal displacement u(R,z,t) and colored by displacement magnitude.
"""
function animate_cylindrical_twist_3d(U, rs, zs, w;
                                      file::String = "cylindrical_twist_solid.mp4",
                                      frames::Int = 60,
                                      amp_scale::Real = 500.0) # Scales the visual warp
    R = rs[end]
    h = zs[end]

    fig = Figure(size = (800, 700))

    # 1. Lock the limits to exactly match the sample's bounding box in mm.
    # This completely stops the camera/frame from bouncing around.
    ax  = Axis3(fig[1, 1];
                title    = "Solid 3D Torsional Shear (ω = $w rad/s)",
                xlabel   = "x (mm)", ylabel = "y (mm)", zlabel = "z (mm)",
                azimuth  = 0.2π, elevation = 0.1π,
                aspect   = :data,
                limits   = ((-R*1e3, R*1e3), (-R*1e3, R*1e3), (0.0, h*1e3)))

    ph  = Observable(0.0)

    # 2. Extract the displacement exactly at the outer edge (r = R)
    U_edge = U[end, :]

    # Create a full 360-degree rotational grid
    thetas = range(0, 2π, length=60)

    # Time-dependent displacement at the edge
    u_edge_t = @lift(real.(U_edge .* cis($ph)))

    # 3. Map into a full 3D cylindrical shell, adding the twist to the angle
    X = @lift([R * cos(th + amp_scale * $u_edge_t[j] / R) for th in thetas, j in 1:length(zs)] .* 1e3)
    Y = @lift([R * sin(th + amp_scale * $u_edge_t[j] / R) for th in thetas, j in 1:length(zs)] .* 1e3)
    Z = [zs[j] for th in thetas, j in 1:length(zs)] .* 1e3

    # 4. Color by the physical displacement amplitude (in micrometers)
    C = @lift([$u_edge_t[j] for th in thetas, j in 1:length(zs)] .* 1e6)
    amp = maximum(abs.(U_edge)) * 1e6

    # Plot the solid 3D outer shell
    surface!(ax, X, Y, Z; color = C, colormap = :vik,
             colorrange = (-amp, amp), shading = NoShading)

    # Add a faint wireframe so the rotational twisting is visually obvious
    wireframe!(ax, X, Y, Z; color = (:black, 0.15), linewidth = 1)

    record(fig, file, range(0, 2π; length = frames); framerate = 20) do t
        ph[] = t
    end

    @info "wrote $file"
    return file
end

# ─── §8  Driver ────────────────────────────────────────────────────────────

function main()
    # ---- §5  Verification -------------------------------------------------
    println("="^70)
    verify_solve_final_2d()
    println("-"^70)
    verify_solve_cyl_torsion()
    println("="^70)

    # ---- §3.3 / §3.4  alternative discretisations sanity check ------------
    # bulk_kron returns the bulk operator only (no BCs) — verify size and
    # sparsity rather than solving with it directly.
    A_kron = bulk_kron(mp, 10.0; Nx = 40, Ny = 40, model = :fkv)
    @printf("bulk_kron      assembled  %d×%d  (nnz = %d)\n",
            size(A_kron, 1), size(A_kron, 2), nnz(A_kron))
    # solve_mol — alternative MOL pipeline (constant-coefficient FKV).
    # First call triggers ModelingToolkit compilation (can be slow);
    # comment out if you want a quick run.
    #println("solve_mol pipeline (MethodOfLines + NonlinearSolve)…")
    #sol_mol = solve_mol(mp, 10.0; N = 30, model = :fkv)
    #@printf("  NonlinearSolve retcode = %s\n", sol_mol.retcode)

    # ---- §6  Report figures -----------------------------------------------
    plot_dynamic_moduli(mp;       file = "dynamic_moduli.png")
    plot_frequency_sweep(mp;      file = "frequency_sweep.png")
    comparison_figure(mp;         file = "comparison.png")
    plot_verification_figures(;   file = "verification.png")

    # ---- §7  Animations ---------------------------------------------------
    # Headline: shear-wave scattering off the soft inclusion (wave regime).
    Uw, xw, yw = solve_final_2d(mp, 350.0; Nx = 120, Ny = 120)
    animate_field(Uw, xw, yw, 350.0;
                  file = "wave_scattering.mp4", nframes = 48,
                  incl = (INCL_CX, INCL_CY, INCL_RADIUS),
                  title = "Fractional KV shear wave", cmap = :balance)

    Uq, xq, yq = solve_final_2d(mp, 10.0; Nx = 90, Ny = 90)   # keep this grid's coords
    animate_field(Uq, xq, yq, 10.0;
                  file = "quasistatic_warp.mp4", nframes = 48,
                  incl = (INCL_CX, INCL_CY, INCL_RADIUS),
                  title = "Quasi-static (rheometric) regime", cmap = :vik)
    # 3D surface view for the slide deck.
    Uf, xf, yf = solve_final_2d(mp, 150.0)
    animate_shear_3d(Uf, xf, yf, 150.0; file = "mayo_phys_wobble.mp4")
    U_cyl, r_cyl, z_cyl = solve_cyl_torsion(mp, 10.0; NR=60, NZ=30)
    animate_cylindrical_twist_3d(U_cyl, r_cyl, z_cyl, 10.0; file="cylindrical_twist_solid.mp4")
    plot_tip_frequency_response(mp; file = "tip_frequency_response.png")
end

Base.invokelatest(main)
include("cyl_fractional_damping.jl")
verify_fractional_damping()
sweep_cyl_damping()
render_cyl_tall()