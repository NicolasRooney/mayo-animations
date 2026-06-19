# ============================================================================
#  cyl_fractional_damping.jl
#  Fractional Kelvin–Voigt (FKV) viscous damping for the cylindrical torsion
#  model of Numerical_Models.jl  (§3.2  solve_cyl_torsion).
#
#  This file is `include`d AFTER Numerical_Models.jl, so it reuses the existing
#  MayoParams, mp, Gstar_uniform, Gstar_model, solve_cyl_torsion and
#  animate_cylindrical_twist_3d directly — no solver is re-implemented.
# ----------------------------------------------------------------------------
#  WHERE THE DAMPING LIVES  (read this before the code)
#
#  Time-domain FKV shear constitutive law:
#       τ(t) = G γ(t) + η_α · D_t^α γ(t),      α ∈ (0,1),   [η_α] = Pa·s^α
#  with D_t^α the Caputo fractional derivative.  Azimuthal momentum balance for
#  the torsional displacement u(r,z,t) (engineering shears γ_{rφ}=∂_r u − u/r,
#  γ_{zφ}=∂_z u):
#       ρ ∂_tt u = (1/r²) ∂_r[ r² τ_{rφ} ] + ∂_z τ_{zφ},
#       τ_{rφ} = (G + η_α D_t^α) γ_{rφ},   τ_{zφ} = (G + η_α D_t^α) γ_{zφ}.
#
#  Harmonic ansatz  u = Re{ U₀(r,z) e^{iωt} }.  The Caputo derivative of e^{iωt}
#  carries the Fourier symbol  D_t^α → (iω)^α, so the bracket (G + η_α D_t^α)
#  becomes a single complex modulus G*(ω) and ∂_tt → −ω²:
#
#       (1/r²) ∂_r[ r² G* (∂_r U₀ − U₀/r) ] + ∂_z(G* ∂_z U₀) + ρω² U₀ = 0,
#
#       G*(ω) = G + η_α (iω)^α
#             = [ G + η_α ω^α cos(απ/2) ]  +  i [ η_α ω^α sin(απ/2) ]
#             =          G'(ω)             +  i          G''(ω).
#
#  ⇒ This is EXACTLY the operator already in solve_cyl_torsion.  The viscous
#    (dissipative) damping term is therefore NOT a new operator to bolt on — it
#    is the imaginary part  G''(ω) = η_α ω^α sin(απ/2) > 0  of the SAME complex
#    coefficient the solver already carries.  Setting η_α = 0 (elastic) makes G*
#    real and removes all damping; η_α > 0 introduces:
#       • a loss tangent   tanδ(ω) = G''/G' = η_α ω^α sin(απ/2)
#                                            / [G + η_α ω^α cos(απ/2)],
#       • a through-thickness phase lag (U₀ becomes complex), and
#       • strictly positive per-cycle dissipation ∝ ω·Im(G*)·∫|γ̂|² dV.
#
#  EXACT REFERENCE USED FOR VERIFICATION.  For spatially uniform G*, the ansatz
#  U₀(r,z) = r f(z) makes the radial shear γ_{rφ} = ∂_r U₀ − U₀/r ≡ 0
#  identically (verified symbolically for both the continuous operator AND the
#  discrete flux stencil, including the lateral r=R ghost fold), so the PDE
#  collapses to the 1-D Helmholtz problem
#       G* f''(z) + ρω² f(z) = 0,   f(0)=0,  f(h)=θ₀,
#  whose closed form is
#       f(z) = θ₀ sin(k* z)/sin(k* h),   k* = ω √(ρ/G*).
#  The discrete solver reproduces  U₀(r,z) = r θ₀ sin(k* z)/sin(k* h)  with the
#  radial direction resolved EXACTLY and only the O(Δz²) z-truncation remaining.
#  Because k* is complex, this profile is complex — its z-dependent phase IS the
#  fractional-KV damping.  In the static limit ω→0 it reduces to r θ₀ z/h.
# ============================================================================


"""
    torsion_profile(p, w, Gs, zs; theta0=1e-3) -> Vector{ComplexF64}

Through-thickness factor  f(z) = θ₀ sin(k* z)/sin(k* h),  k* = ω √(ρ/G*),
for a uniform complex modulus `Gs`.  Reduces to θ₀ z/h as ω→0.
"""
function torsion_profile(p::MayoParams, w::Real, Gs::Number, zs;
                         theta0::Real = 1e-3)
    kstar = w * sqrt(p.rho / Gs)        # complex √ (principal branch), Re(k*)>0
    kh    = kstar * p.H
    if abs(kh) < 1e-8                    # static / quasi-static limit: sin→arg
        return ComplexF64.(theta0 .* zs ./ p.H)
    end
    return ComplexF64.(theta0 .* sin.(kstar .* zs) ./ sin(kh))
end

"""
    torsion_analytic_2d(p, w, Gs, rs, zs; theta0=1e-3) -> Matrix{ComplexF64}

Separable analytic torsion field  U₀(r,z) = r · f(z)  (exact for uniform G*).
"""
function torsion_analytic_2d(p::MayoParams, w::Real, Gs::Number, rs, zs;
                             theta0::Real = 1e-3)
    f = torsion_profile(p, w, Gs, zs; theta0 = theta0)
    return [rs[i] * f[j] for i in 1:length(rs), j in 1:length(zs)]
end


# ─── Verification ───────────────────────────────────────────────────────────
"""
    verify_fractional_damping(; theta0=1e-3)

Rigorous checks that the fractional-KV damping is correctly carried by the
cylindrical torsion solver:

  A. solver → separable analytic  r θ₀ sin(k* z)/sin(k* h)  at O(Δz²);
  B. ω→0 static limit recovers the affine twist  r θ₀ z/h  (machine eps);
  C. damping signatures: closed-form vs numeric loss tangent, a strictly
     positive through-thickness phase lag and per-cycle dissipation for FKV,
     both exactly zero in the elastic (η_α=0) limit.
"""
function verify_fractional_damping(; theta0::Real = 1e-3)
    println("="^70)
    println("[FKV-cyl Test A]  solver vs separable analytic  r·θ₀·sin(k*z)/sin(k*h)")
    p  = MayoParams(1000.0, 0.020, 0.001,
                    2500.0, 0.45, 120.0,
                    2500.0, 0.45, 120.0)       # uniform MayoB strong gel
    w  = 2000.0
    Gs = Gstar_model(p, w; model = :fkv)
    Gfun = (r, z, ww) -> Gs                    # constant complex modulus field
    kh = w * sqrt(p.rho / Gs) * p.H
    @printf("          ω = %.0f rad/s,  G* = %.0f %+.0fi Pa,  k*h = %.4f %+.4fi\n",
            w, real(Gs), imag(Gs), real(kh), imag(kh))
    @printf("          %4s   %12s\n", "NZ", "rel err")
    errs = Float64[]
    for NZ in (10, 20, 40, 80)
        U, rs, zs = solve_cyl_torsion(p, w; NR = 20, NZ = NZ,
                                      theta0 = theta0, Gstar_field = Gfun)
        Uex = torsion_analytic_2d(p, w, Gs, rs, zs; theta0 = theta0)
        rel = maximum(abs.(U .- Uex)) / maximum(abs.(Uex))
        push!(errs, rel)
        @printf("          %4d   %12.3e\n", NZ, rel)
    end
    rates = log2.(errs[1:end-1] ./ errs[2:end])
    @printf("          observed z-rates: %s   (≈2 expected)\n",
            string(round.(rates, digits = 3)))

    # ---- B: static limit -----------------------------------------------------
    ws  = 1e-8
    Ges = Gstar_model(p, ws; model = :elastic)
    Us, rs, zs = solve_cyl_torsion(p, ws; NR = 40, NZ = 20,
                                   theta0 = theta0, Gstar_field = (r,z,ww)->Ges)
    Uaff = [rs[i] * theta0 * zs[j] / p.H for i in 1:length(rs), j in 1:length(zs)]
    rel_static = maximum(abs.(Us .- Uaff)) / maximum(abs.(Uaff))
    @printf("[FKV-cyl Test B]  static limit vs r·θ₀·z/h   rel err = %.2e  (≈machine eps)\n",
            rel_static)

    # ---- C: damping signatures (elastic vs fractional KV) --------------------
    println("[FKV-cyl Test C]  damping signatures  (elastic η=0  vs  fractional KV)")
    pc, wc = mp, 300.0
    Gel = Gstar_model(pc, wc; model = :elastic)
    Gfk = Gstar_model(pc, wc; model = :fkv)

    a   = pc.alpha_strong
    Gpp = pc.eta_strong * wc^a * sin(a*π/2)               # G'' closed form
    Gp  = pc.G_strong  + pc.eta_strong * wc^a * cos(a*π/2) # G'  closed form
    @printf("          loss tangent tanδ:  closed-form %.5f   from G* %.5f\n",
            Gpp / Gp, imag(Gfk) / real(Gfk))

    NR, NZ = 60, 40
    Uel, rsc, _ = solve_cyl_torsion(pc, wc; NR=NR, NZ=NZ, theta0=theta0,
                                    Gstar_field = (r,z,ww)->Gel)
    Ufk, _,  _  = solve_cyl_torsion(pc, wc; NR=NR, NZ=NZ, theta0=theta0,
                                    Gstar_field = (r,z,ww)->Gfk)

    # through-thickness phase lag relative to the driven plate U₀(R,h)=Rθ₀ (real)
    lag(U) = let e = U[end, :]
        rad2deg(maximum(abs.(angle.(e[2:end]) .- angle(e[end]))))
    end
    @printf("          through-thickness phase lag:  elastic %.4f°   FKV %.4f°\n",
            lag(Uel), lag(Ufk))

    # per-cycle dissipation  ∝  ω · Im(G*) · ∫ |∂_z U₀|² (2π r dr dz)  ≥ 0
    hz, hr = pc.H/NZ, pc.L/NR
    function dissip(U, G)
        dUz = (U[:, 2:end] .- U[:, 1:end-1]) ./ hz
        s = 0.0
        for j in 1:size(dUz, 2), i in 1:length(rsc)
            s += abs2(dUz[i, j]) * rsc[i]
        end
        0.5 * wc * imag(G) * s * (2π * hr * hz)
    end
    Del, Dfk = dissip(Uel, Gel), dissip(Ufk, Gfk)
    @printf("          per-cycle dissipation ∝:      elastic %.3e   FKV %.3e\n",
            Del, Dfk)

    ok = (rates[end] > 1.5) && (rel_static < 1e-8) &&
         (lag(Ufk)  > 1.0)  && (Dfk > 0) &&
         (lag(Uel) == 0.0)  && (Del == 0.0)
    println(ok ? "  ⇒ PASS: damping enters only via Im G*; the elastic limit is loss-free." :
                 "  ⇒ CHECK: review the diagnostics above.")
    println("="^70)
    return (errs = errs, rates = rates, rel_static = rel_static,
            tand = imag(Gfk)/real(Gfk), lag_fkv = lag(Ufk),
            dissip_fkv = Dfk, pass = ok)
end


# ─── Damping sweep + 3-D graphic ─────────────────────────────────────────────
"""
    sweep_cyl_damping(p=mp; w=300.0, NR=60, NZ=40, theta0=1e-3, make_mp4s=true)

Sweep the fractional-KV damping strength on the cylindrical torsion model and
produce, at frequency `w`:

  • cyl_damping_sweep.png  – |U₀(R,z)| and the through-thickness phase lag for
                             η_α ∈ {0 (elastic), ½η, η, 2η};
  • cyl_twist_elastic.mp4  – undamped reference (standing torsional oscillation);
  • cyl_twist_fkv.mp4      – fractional-KV damped twist: the 3-D torsion graphic
                             "like before, but with damping" — a lagging /
                             travelling spiral whose lag is set by Im G*.

Uses the existing solver and animate_cylindrical_twist_3d. The Gstar_field
lambdas take the solver's frequency argument `ww`, so each modulus is evaluated
consistently at the solve frequency.
"""
function sweep_cyl_damping(p::MayoParams = mp; w::Real = 300.0,
                           NR::Int = 60, NZ::Int = 40, theta0::Real = 1e-3,
                           make_mp4s::Bool = true)
    Gfkv(scale) = (r, z, ww) -> p.G_strong + scale*p.eta_strong*(im*ww)^p.alpha_strong
    cases = [("elastic (η=0)", (r,z,ww)->complex(p.G_strong, 0.0), :black),
             ("FKV  ½η",       Gfkv(0.5),                          :seagreen),
             ("FKV   η",       Gfkv(1.0),                          :crimson),
             ("FKV  2η",       Gfkv(2.0),                          :dodgerblue)]

    fig = Figure(size = (1150, 470))
    ax1 = Axis(fig[1, 1]; xlabel = "z (mm)", ylabel = "|U₀(R,z)| (μm)",
               title = @sprintf("Edge amplitude  (ω = %.0f rad/s)", w))
    ax2 = Axis(fig[1, 2]; xlabel = "z (mm)", ylabel = "phase lag vs plate (deg)",
               title = "Damping-induced through-thickness phase lag")
    for (label, Gfun, clr) in cases
        U, rs, zs = solve_cyl_torsion(p, w; NR=NR, NZ=NZ, theta0=theta0,
                                      Gstar_field = Gfun)
        e      = U[end, :]                                   # U₀(R,z)
        lagdeg = rad2deg.(angle.(e) .- angle(e[end]))        # ref = driven plate
        lines!(ax1, collect(zs).*1e3, abs.(e).*1e6; color=clr, linewidth=2.5, label=label)
        lines!(ax2, collect(zs).*1e3, lagdeg;       color=clr, linewidth=2.5, label=label)
    end
    axislegend(ax1; position = :lt)
    save("cyl_damping_sweep.png", fig); @info "wrote cyl_damping_sweep.png"

    if make_mp4s
        Uel, rsE, zsE = solve_cyl_torsion(p, w; NR=NR, NZ=NZ, theta0=theta0,
                          Gstar_field = (r,z,ww)->complex(p.G_strong, 0.0))
        animate_cylindrical_twist_3d(Uel, rsE, zsE, w; file = "cyl_twist_elastic.mp4")

        Ufk, rsF, zsF = solve_cyl_torsion(p, w; NR=NR, NZ=NZ, theta0=theta0,
                          Gstar_field = (r,z,ww)->Gstar_uniform(p, w; model = :fkv))
        animate_cylindrical_twist_3d(Ufk, rsF, zsF, w; file = "cyl_twist_fkv.mp4")
    end
    return ("cyl_damping_sweep.png", "cyl_twist_elastic.mp4", "cyl_twist_fkv.mp4")
end


# ─── Stretched-Z (taller) cylinder rendering ─────────────────────────────────
"""
    animate_cylindrical_twist_3d_tall(U, rs, zs, w; file="cyl_twist_tall.mp4",
                                      aspect_z=2.5, frames=60, amp_scale=500.0)

Identical to `animate_cylindrical_twist_3d` but exaggerates the vertical (z)
extent of the *view only*.  The Axis3 box aspect is set to (1, 1, `aspect_z`)
instead of `:data`, so the short physical sample (R ≫ h) renders as a tall
cylinder.  The z tick labels stay the true height in mm — only the drawn box is
stretched, so the dimensions are unchanged.  `aspect_z` is the z box length
relative to the x/y box (2.0 ≈ a cylinder ~2× taller than its diameter; raise
it for more stretch).

The mesh overlay is split so the static horizontal rings don't distract: the
constant-z rings are drawn faint (`ring_alpha`, set 0 to remove them) while the
spiraling generators that actually show the twist stay visible (`gen_alpha`,
`n_gen` of them).
"""
function animate_cylindrical_twist_3d_tall(U, rs, zs, w;
                                           file::String = "cyl_twist_tall.mp4",
                                           aspect_z::Real = 2.0,
                                           ring_alpha::Real = 0.05,
                                           gen_alpha::Real = 0.22,
                                           n_gen::Int = 24,
                                           frames::Int = 60,
                                           amp_scale::Real = 500.0)
    R = rs[end]
    h = zs[end]

    fig = Figure(size = (680, 820))            # taller canvas for the taller box

    ax  = Axis3(fig[1, 1];
                title    = "Torsional Shear — stretched z  (ω = $w rad/s)",
                xlabel   = "x (mm)", ylabel = "y (mm)", zlabel = "z (mm)",
                azimuth  = 0.2π, elevation = 0.1π,
                aspect   = (1.0, 1.0, Float64(aspect_z)),  # only change vs original
                limits   = ((-R*1e3, R*1e3), (-R*1e3, R*1e3), (0.0, h*1e3)))

    ph     = Observable(0.0)
    U_edge = U[end, :]
    thetas = range(0, 2π, length = 60)
    u_edge_t = @lift(real.(U_edge .* cis($ph)))

    X = @lift([R * cos(th + amp_scale * $u_edge_t[j] / R) for th in thetas, j in 1:length(zs)] .* 1e3)
    Y = @lift([R * sin(th + amp_scale * $u_edge_t[j] / R) for th in thetas, j in 1:length(zs)] .* 1e3)
    Z = [zs[j] for th in thetas, j in 1:length(zs)] .* 1e3

    C   = @lift([$u_edge_t[j] for th in thetas, j in 1:length(zs)] .* 1e6)
    amp = maximum(abs.(U_edge)) * 1e6

    surface!(ax, X, Y, Z; color = C, colormap = :vik,
             colorrange = (-amp, amp), shading = NoShading)

    # Cross-section rings (constant z).  Under a uniform per-height twist these
    # are invariant — the "non-moving horizontal bands" — so draw them static and
    # very faint just for a cross-section reference.  Lower `ring_alpha` to fade
    # them further, or set it to 0 to drop them entirely.
    ring_th = range(0, 2π, length = 80)
    if ring_alpha > 0
        for j in 1:length(zs)
            lines!(ax, R .* cos.(ring_th) .* 1e3, R .* sin.(ring_th) .* 1e3,
                   fill(zs[j] * 1e3, length(ring_th));
                   color = (:black, ring_alpha), linewidth = 1)
        end
    end

    # Generators (constant θ, varying z).  These spiral with the twist, so they
    # carry the motion — keep them visible.  `n_gen` sets how many.
    gi = unique(round.(Int, range(1, length(thetas); length = n_gen)))
    for k in gi
        gx = @lift([$X[k, j] for j in 1:length(zs)])
        gy = @lift([$Y[k, j] for j in 1:length(zs)])
        gz = [Z[k, j] for j in 1:length(zs)]
        lines!(ax, gx, gy, gz; color = (:black, gen_alpha), linewidth = 1)
    end

    record(fig, file, range(0, 2π; length = frames); framerate = 20) do t
        ph[] = t
    end
    @info "wrote $file"
    return file
end

"""
    render_cyl_tall(p=mp; w_torsion=10.0, w_damp=300.0, NR=60, NZ=40,
                    theta0=1e-3, aspect_z=2.5)

Supplemental set: re-renders the three cylinder animations with a stretched-z
view (same physics and dimensions as the originals, only a taller-looking box):

  • cylindrical_twist_solid_tall.mp4  – baseline torsion (mirrors main()'s run);
  • cyl_twist_elastic_tall.mp4        – elastic reference;
  • cyl_twist_fkv_tall.mp4            – fractional-KV damped.

Add `render_cyl_tall()` after `sweep_cyl_damping()` at the bottom of the script.
To emit MP4s directly instead of mp4s, change the `.mp4` extensions below to
`.mp4` (Makie picks the format from the extension).
"""
function render_cyl_tall(p::MayoParams = mp; w_torsion::Real = 10.0,
                         w_damp::Real = 300.0, NR::Int = 60, NZ::Int = 40,
                         theta0::Real = 1e-3, aspect_z::Real = 2.0)
    Ub, rsB, zsB = solve_cyl_torsion(p, w_torsion; NR = 60, NZ = 30, theta0 = theta0)
    animate_cylindrical_twist_3d_tall(Ub, rsB, zsB, w_torsion;
        file = "cylindrical_twist_solid_tall.mp4", aspect_z = aspect_z)

    Ue, rsE, zsE = solve_cyl_torsion(p, w_damp; NR = NR, NZ = NZ, theta0 = theta0,
        Gstar_field = (r, z, ww) -> complex(p.G_strong, 0.0))
    animate_cylindrical_twist_3d_tall(Ue, rsE, zsE, w_damp;
        file = "cyl_twist_elastic_tall.mp4", aspect_z = aspect_z)

    Uf, rsF, zsF = solve_cyl_torsion(p, w_damp; NR = NR, NZ = NZ, theta0 = theta0,
        Gstar_field = (r, z, ww) -> Gstar_uniform(p, w_damp; model = :fkv))
    animate_cylindrical_twist_3d_tall(Uf, rsF, zsF, w_damp;
        file = "cyl_twist_fkv_tall.mp4", aspect_z = aspect_z)

    return ("cylindrical_twist_solid_tall.mp4",
            "cyl_twist_elastic_tall.mp4", "cyl_twist_fkv_tall.mp4")
end
