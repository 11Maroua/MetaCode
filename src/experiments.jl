# ====================== LIVRABLE COMPLET ====================
# Métaheuristiques pour le Set Packing Problem
# EI1 : Glouton + Recherche Locale
# EI2 : GRASP vs Reactive GRASP
# EI3 : ACO vs GA vs GRASP
# ============================================================

using LinearAlgebra
using Printf
using JuMP, GLPK
using Statistics
using Plots
using Random

include("loadSPP.jl")
include("setSPP.jl")
include("Glouton.jl")
include("Exploration.jl")
include("Grasp.jl")


# ==============================================================================
# EI1 : GLOUTON + RECHERCHE LOCALE
# ==============================================================================


"""
Résolution exacte avec GLPK
"""
function resoudreSPP_exact(fname::String; timeout=600)
    C, A = loadSPP(fname)
    spp = setSPP(C, A)
    set_optimizer(spp, GLPK.Optimizer)
    set_silent(spp)
    set_time_limit_sec(spp, timeout)
    optimize!(spp)
    
    return (
        solution = value.(spp[:x]),
        objectif = objective_value(spp),
        temps = solve_time(spp),
        statut = termination_status(spp)
    )
end


"""
Expérimentation EI1 : 10 instances
"""
function experimentationSPP()
    println("\n" * "="^80)
    println("EXPÉRIMENTATION EI1 : 10 INSTANCES SPP")
    println("="^80)
    
    # Instances triées par taille croissante (m × n)
    instances = [
        "dat/didactic.dat",           # 7 × 9
        "dat/pb_100rnd0100.dat",      # 500 × 100
        "dat/pb_100rnd0300.dat",      # 500 × 100
        "dat/pb_200rnd0100.dat",      # 1000 × 200
        "dat/pb_200rnd0300.dat",      # 1000 × 200
        "dat/pb_500rnd1500.dat",      # 1500 × 500
        "dat/pb_500rnd0100.dat",      # 2500 × 500
        "dat/pb_500rnd0300.dat",      # 2500 × 500
        "dat/pb_1000rnd0100.dat",     # 5000 × 1000
        "dat/pb_2000rnd0100.dat"      # 10000 × 2000
    ]
    
    resultats = []
    
    for (idx, inst) in enumerate(instances)
        println("\n" * "="^80)
        println("[$idx/$(length(instances))] Instance: ", basename(inst))
        println("="^80)
        
        C, A = loadSPP(inst)
        m, n = size(A)
        println("Dimensions: m=$m, n=$n")
        
        # Résolution exacte avec GLPK (timeout 600s)
        z_opt = missing
        t_opt = missing
        statut_opt = "N/A"
        
        println("\n>>> Résolution exacte (GLPK) - timeout 600s...")
        # Résolution exacte SEULEMENT pour pb_100rnd0100.dat
        if basename(inst) == "pb_100rnd0100.dat"
            println("\n>>> Résolution exacte (GLPK) - timeout 600s...")
            try
                res = resoudreSPP_exact(inst, timeout=600)
                if res.statut == MOI.OPTIMAL
                    z_opt = res.objectif
                    t_opt = res.temps
                    statut_opt = "OPTIMAL"
                    println("✓ Z* = $z_opt | Temps = $(round(t_opt, digits=3))s")
                elseif res.statut == MOI.TIME_LIMIT
                    statut_opt = "TIMEOUT"
                    println(" Timeout 600s")
                else
                    statut_opt = "ERREUR"
                    println(" Statut: $(res.statut)")
                end
            catch e
                println("✗ Erreur: $e")
            end
        else
            println("\n>>> Résolution exacte : SKIPPED (seulement pour instance  pb_100rnd0100)")
        end
        
        println("\n>>> Construction gloutonne...")
        t_construct = @elapsed x_construct = construction_gloutonne(C, A, verbose=false)
        z_construct = dot(C, x_construct)
        gap_construct = ismissing(z_opt) ? missing : 100 * (z_opt - z_construct) / z_opt
        
        println("\n>>> Recherche locale...")
        t_local = @elapsed x_local = descente_profonde(x_construct, C, A, max_iter=1000, verbose=false)
        z_local = dot(C, x_local)
        t_total = t_construct + t_local
        gap_local = ismissing(z_opt) ? missing : 100 * (z_opt - z_local) / z_opt
        speedup = ismissing(t_opt) ? missing : t_opt / t_total
        
        if !ismissing(gap_local)
            println("Z_local = $z_local | Gap = $(round(gap_local, digits=2))% | Speedup = $(round(speedup, digits=1))x")
        else
            println("Z_local = $z_local | Temps = $(round(t_total, digits=4))s")
        end
        
        push!(resultats, (
            instance = basename(inst),
            m = m, n = n,
            z_opt = z_opt, t_opt = t_opt, statut_opt = statut_opt,
            z_construct = z_construct, t_construct = t_construct, gap_construct = gap_construct,
            z_local = z_local, t_total = t_total, gap_local = gap_local, speedup = speedup
        ))
    end
    
    afficher_resultats_EI1(resultats)
    sauvegarder_resultats_EI1(resultats)
    generer_latex_EI1(resultats)
    generer_graphiques_EI1(resultats)
    
    println("\n" * "="^80)
    println("  EXPÉRIMENTATION EI1 TERMINÉE")
    println("  Fichiers générés dans res/")
    println("="^80)
    
    return resultats
end


function afficher_resultats_EI1(resultats)
    println("\n" * "="^130)
    println("TABLEAU RÉCAPITULATIF EI1")
    println("="^130)
    
    @printf "%-20s %5s %6s | %10s %10s %8s | %10s %8s %8s | %10s %10s %8s %10s\n" "Instance" "m" "n" "Z*" "T*(s)" "Statut" "Z_const" "T_c(s)" "Gap%" "Z_local" "T_tot(s)" "Gap%" "Speedup"
    println("-"^130)
    
    for r in resultats
        @printf "%-20s %5d %6d | %10s %10s %8s | %10.0f %8.4f %8s | %10.0f %10.4f %8s %10s\n" r.instance r.m r.n (ismissing(r.z_opt) ? "---" : @sprintf("%.0f", r.z_opt)) (ismissing(r.t_opt) ? "---" : @sprintf("%.2f", r.t_opt)) r.statut_opt r.z_construct r.t_construct (ismissing(r.gap_construct) ? "---" : @sprintf("%.2f", r.gap_construct)) r.z_local r.t_total (ismissing(r.gap_local) ? "---" : @sprintf("%.2f", r.gap_local)) (ismissing(r.speedup) ? "---" : @sprintf("%.1f", r.speedup))
    end
    println("="^130)
end


function sauvegarder_resultats_EI1(resultats)
    mkpath("res")
    open("res/resultats_EI1.txt", "w") do f
        write(f, "RÉSULTATS EI1\n" * "="^80 * "\n\n")
        for r in resultats
            write(f, "Instance: $(r.instance)\n")
            write(f, "  m=$(r.m), n=$(r.n)\n")
            write(f, "  Z_opt=$(r.z_opt) ($(r.statut_opt))\n")
            write(f, "  Z_construct=$(r.z_construct), Gap=$(r.gap_construct)%\n")
            write(f, "  Z_local=$(r.z_local), Gap=$(r.gap_local)%\n\n")
        end
    end
    println(" res/resultats_EI1.txt généré")
end


function generer_latex_EI1(resultats)
    mkpath("res")
    open("res/tableau_EI1.tex", "w") do f
        write(f, "\\begin{table}[htbp]\n\\centering\n")
        write(f, "\\caption{Résultats EI1 : Glouton + Recherche Locale}\n")
        write(f, "\\label{tab:ei1}\n\\footnotesize\n")
        write(f, "\\begin{tabular}{|l|rr|rrr|rrr|rrrr|}\n\\hline\n")
        write(f, "\\textbf{Instance} & \\textbf{m} & \\textbf{n} & \\textbf{Z*} & \\textbf{T(s)} & \\textbf{Statut} & \\textbf{Z} & \\textbf{T(s)} & \\textbf{Gap\\%} & \\textbf{Z} & \\textbf{T(s)} & \\textbf{Gap\\%} & \\textbf{Speedup} \\\\\n\\hline\n")
        
        for r in resultats
            inst = replace(r.instance, "_" => "\\_")
            z_opt = ismissing(r.z_opt) ? "---" : @sprintf("%.0f", r.z_opt)
            t_opt = ismissing(r.t_opt) ? "---" : @sprintf("%.2f", r.t_opt)
            gap_c = ismissing(r.gap_construct) ? "---" : @sprintf("%.2f", r.gap_construct)
            gap_l = ismissing(r.gap_local) ? "---" : @sprintf("%.2f", r.gap_local)
            speedup = ismissing(r.speedup) ? "---" : @sprintf("%.1f", r.speedup)
            
            write(f, "$inst & $(r.m) & $(r.n) & $z_opt & $t_opt & $(r.statut_opt) & ")
            write(f, "$(Int(r.z_construct)) & $(round(r.t_construct, digits=4)) & $gap_c & ")
            write(f, "$(Int(r.z_local)) & $(round(r.t_total, digits=4)) & $gap_l & $speedup \\\\\n")
        end
        
        write(f, "\\hline\n\\end{tabular}\n\\end{table}\n")
    end
    println(" res/tableau_EI1.tex généré !")
end


function generer_graphiques_EI1(resultats)
    mkpath("res")
    
    instances = [r.instance for r in resultats]
    indices = 1:length(instances)
    
    # Graphique 1 : Évolution Construction → Local (COURBES)
    z_construct_vals = [r.z_construct for r in resultats]
    z_local_vals = [r.z_local for r in resultats]
    
    p1 = plot(indices, z_construct_vals, 
              label="Construction Gloutonne", 
              linewidth=3, color=:red, marker=:circle, markersize=6,
              title="Évolution des solutions", 
              xlabel="Instances", ylabel="Valeur Z",
              xticks=(indices, instances), xrotation=45,
              legend=:topleft, size=(900, 500), bottom_margin=10Plots.mm)
    
    plot!(p1, indices, z_local_vals, 
          label="Recherche Locale", 
          linewidth=3, color=:blue, marker=:square, markersize=6)
    
    # Graphique 2 : Amélioration absolue (COURBE)
    ameliorations = [r.z_local - r.z_construct for r in resultats]
    
    p2 = plot(indices, ameliorations,
              label="Amélioration (Z_local - Z_construct)",
              linewidth=3, color=:green, marker=:circle, markersize=6,
              title="Amélioration par Recherche Locale",
              xlabel="Instances", ylabel="Amélioration",
              xticks=(indices, instances), xrotation=45,
              legend=:topright, size=(900, 500), bottom_margin=10Plots.mm)
    hline!(p2, [0], label="Baseline", color=:black, linestyle=:dash)
    
    # Graphique 3 : Temps de calcul (COURBES log)
    t_construct_vals = [r.t_construct for r in resultats]
    t_total_vals = [r.t_total for r in resultats]
    t_opt_vals = [ismissing(r.t_opt) ? nothing : r.t_opt for r in resultats]
    
    p3 = plot(indices, t_construct_vals,
              label="Construction seule",
              linewidth=2, color=:orange, marker=:circle, markersize=5,
              title="Temps de calcul (échelle log)",
              xlabel="Instances", ylabel="Temps (s)",
              xticks=(indices, instances), xrotation=45,
              yscale=:log10, legend=:topleft, size=(900, 500), bottom_margin=10Plots.mm)
    
    plot!(p3, indices, t_total_vals,
          label="Construction + Local",
          linewidth=3, color=:red, marker=:square, markersize=5)
    
    # Ajouter temps GLPK comme POINT unique (pas courbe)
    if any(.!isnothing.(t_opt_vals))
        valid_indices = findall(.!isnothing.(t_opt_vals))
        scatter!(p3, valid_indices, [t_opt_vals[i] for i in valid_indices],
                 label="GLPK (optimal)",
                 color=:blue, marker=:diamond, markersize=8)
    end
    
    # Graphique 4 : Gaps (COURBES) - seulement pour instances avec Z*
    mask_opt = .!ismissing.([r.gap_construct for r in resultats])
    if any(mask_opt)
        indices_opt = findall(mask_opt)
        gap_c = [r.gap_construct for r in resultats[mask_opt]]
        gap_l = [r.gap_local for r in resultats[mask_opt]]
        
        p4 = plot(indices_opt, gap_c,
                  label="Gap Construction",
                  linewidth=3, color=:red, marker=:circle, markersize=6,
                  title="Gaps d'optimalité (instances avec Z*)",
                  xlabel="Instances", ylabel="Gap (%)",
                  xticks=(indices_opt, instances[mask_opt]), xrotation=45,
                  legend=:topright, size=(900, 500), bottom_margin=10Plots.mm)
        
        plot!(p4, indices_opt, gap_l,
              label="Gap Recherche Locale",
              linewidth=3, color=:blue, marker=:square, markersize=6)
        
        hline!(p4, [0], label="Optimal", color=:green, linestyle=:dash)
    else
        p4 = plot(title="Pas de Z* disponible", legend=false, 
                  size=(900, 500), showaxis=false)
    end
    
    # Sauvegarder
    plot(p1, p2, p3, p4, layout=(2,2), size=(1800, 1200))
    savefig("res/graphiques_EI1.pdf")
    
    println("✓ res/graphiques_EI1.pdf généré")
end


# ==============================================================================
# EI2 : GRASP VS REACTIVE GRASP
# ==============================================================================

"""
Expérimentation EI2 : GRASP vs Reactive GRASP
"""
function experimentationSPP_EI2(; n_runs=5, n_iter=100)
    println("\n" * "="^80)
    println("EXPÉRIMENTATION EI2 : GRASP vs REACTIVE GRASP")
    println("Paramètres: $n_runs runs, $n_iter itérations")
    println("="^80)
    
    # Instances triées par taille croissante
    instances = [
        "dat/didactic.dat",
        "dat/pb_100rnd0100.dat",
        "dat/pb_100rnd0300.dat",
        "dat/pb_200rnd0100.dat",
        "dat/pb_200rnd0300.dat",
        "dat/pb_500rnd0100.dat",
        "dat/pb_500rnd0300.dat",
        "dat/pb_500rnd1500.dat",
        "dat/pb_1000rnd0100.dat",
        "dat/pb_2000rnd0100.dat"
    ]
    
    resultats_grasp = []
    resultats_reactive = []
    
    for (idx, inst) in enumerate(instances)
        println("\n[$idx/$(length(instances))] Instance: ", basename(inst))
        
        C, A = loadSPP(inst)
        m, n = size(A)
        
        # GRASP Standard
        println("\n>>> GRASP Standard (alpha=0.7)")
        z_runs = Float64[]
        t_runs = Float64[]
        
        for run in 1:n_runs
            Random.seed!(run * 1000)
            t = @elapsed begin
                x, _, _, _ = grasp_complet(C, A, alpha=0.7, n_iter=n_iter, verbose=false)
                z = dot(C, x)
            end
            push!(z_runs, z)
            push!(t_runs, t)
            println("  Run $run: Z=$z ($(round(t, digits=2))s)")
        end
        
        push!(resultats_grasp, (
            instance = basename(inst), m = m, n = n,
            z_min = minimum(z_runs), z_max = maximum(z_runs), z_moy = mean(z_runs), t_moy = mean(t_runs)
        ))
        
        # Reactive GRASP
        println("\n>>> Reactive GRASP")
        z_runs_r = Float64[]
        t_runs_r = Float64[]
        probs_list = []
        
        for run in 1:n_runs
            Random.seed!(run * 2000)
            t = @elapsed begin
                x, _, _, _, _, probs = reactive_grasp(C, A, alphas=[0.2, 0.5, 0.75, 0.9, 1.0], n_iter=n_iter, update_frequency=20, verbose=false)
                z = dot(C, x)
            end
            push!(z_runs_r, z)
            push!(t_runs_r, t)
            push!(probs_list, probs)
            println("  Run $run: Z=$z ($(round(t, digits=2))s)")
        end
        
        push!(resultats_reactive, (
            instance = basename(inst), m = m, n = n,
            z_min = minimum(z_runs_r), z_max = maximum(z_runs_r), z_moy = mean(z_runs_r), t_moy = mean(t_runs_r),
            probs = mean(probs_list)
        ))
    end
    
    afficher_resultats_EI2(resultats_grasp, resultats_reactive)
    generer_latex_EI2(resultats_grasp, resultats_reactive, n_runs)
    generer_graphiques_EI2(resultats_grasp, resultats_reactive)
    
    println("\n✓ EXPÉRIMENTATION EI2 TERMINÉE")
    return resultats_grasp, resultats_reactive
end


function afficher_resultats_EI2(rg, rr)
    println("\n" * "="^100)
    println("TABLEAU COMPARATIF EI2")
    println("="^100)
    @printf "%-20s | %10s %10s %10s %8s || %10s %10s %10s %8s\n" "Instance" "G_min" "G_max" "G_moy" "T(s)" "R_min" "R_max" "R_moy" "T(s)"
    println("-"^100)
    for i in 1:length(rg)
        @printf "%-20s | %10.0f %10.0f %10.2f %8.2f || %10.0f %10.0f %10.2f %8.2f\n" rg[i].instance rg[i].z_min rg[i].z_max rg[i].z_moy rg[i].t_moy rr[i].z_min rr[i].z_max rr[i].z_moy rr[i].t_moy
    end
    println("="^100)
end


function generer_latex_EI2(rg, rr, n_runs)
    mkpath("res")
    open("res/tableau_EI2.tex", "w") do f
        write(f, "\\begin{table}[htbp]\n\\centering\n")
        write(f, "\\caption{Résultats EI2 : GRASP vs Reactive GRASP ($n_runs runs)}\n")
        write(f, "\\label{tab:ei2}\n\\small\n")
        write(f, "\\begin{tabular}{|l|rr|rrrr|rrrr|}\n\\hline\n")
        write(f, "\\textbf{Instance} & \\textbf{m} & \\textbf{n} & \\multicolumn{4}{c|}{\\textbf{GRASP}} & \\multicolumn{4}{c|}{\\textbf{Reactive GRASP}} \\\\\n")
        write(f, "& & & Z_{min} & Z_{max} & Z_{moy} & T(s) & Z_{min} & Z_{max} & Z_{moy} & T(s) \\\\\n\\hline\n")
        
        for i in 1:length(rg)
            inst = replace(rg[i].instance, "_" => "\\_")
            write(f, "$inst & $(rg[i].m) & $(rg[i].n) & ")
            write(f, "$(Int(rg[i].z_min)) & $(Int(rg[i].z_max)) & $(round(rg[i].z_moy, digits=1)) & $(round(rg[i].t_moy, digits=2)) & ")
            write(f, "$(Int(rr[i].z_min)) & $(Int(rr[i].z_max)) & $(round(rr[i].z_moy, digits=1)) & $(round(rr[i].t_moy, digits=2)) \\\\\n")
        end
        
        write(f, "\\hline\n\\end{tabular}\n\\end{table}\n")
    end
    println("✓ res/tableau_EI2.tex")
end


function generer_graphiques_EI2(rg, rr)
    mkpath("res")
    instances = [r.instance for r in rg]
    indices = 1:length(instances)
    
    # Graphique 1 : Comparaison Z moyen (COURBES)
    z_grasp = [r.z_moy for r in rg]
    z_reactive = [r.z_moy for r in rr]
    
    p1 = plot(indices, z_grasp,
              label="GRASP Standard",
              linewidth=3, color=:red, marker=:circle, markersize=6,
              title="Comparaison Z moyen : GRASP vs Reactive GRASP",
              xlabel="Instances", ylabel="Z moyen",
              xticks=(indices, instances), xrotation=45,
              legend=:topleft, size=(900, 500), bottom_margin=10Plots.mm)
    
    plot!(p1, indices, z_reactive,
          label="Reactive GRASP",
          linewidth=3, color=:blue, marker=:square, markersize=6)
    
    # Graphique 2 : Z_min et Z_max (COURBES avec zones)
    z_min_grasp = [r.z_min for r in rg]
    z_max_grasp = [r.z_max for r in rg]
    z_min_reactive = [r.z_min for r in rr]
    z_max_reactive = [r.z_max for r in rr]
    
    p2 = plot(indices, z_grasp,
              label="GRASP (moyen)",
              linewidth=2, color=:red, marker=:circle, markersize=5,
              title="Variabilité des solutions",
              xlabel="Instances", ylabel="Valeur Z",
              xticks=(indices, instances), xrotation=45,
              legend=:topleft, size=(900, 500), bottom_margin=10Plots.mm)
    
    plot!(p2, indices, z_min_grasp,
          label="GRASP (min-max)",
          linewidth=1, color=:red, linestyle=:dash, alpha=0.5)
    plot!(p2, indices, z_max_grasp,
          label=false,
          linewidth=1, color=:red, linestyle=:dash, alpha=0.5)
    
    plot!(p2, indices, z_reactive,
          label="Reactive (moyen)",
          linewidth=2, color=:blue, marker=:square, markersize=5)
    
    plot!(p2, indices, z_min_reactive,
          label="Reactive (min-max)",
          linewidth=1, color=:blue, linestyle=:dash, alpha=0.5)
    plot!(p2, indices, z_max_reactive,
          label=false,
          linewidth=1, color=:blue, linestyle=:dash, alpha=0.5)
    
    # Graphique 3 : Temps moyen (COURBES)
    t_grasp = [r.t_moy for r in rg]
    t_reactive = [r.t_moy for r in rr]
    
    p3 = plot(indices, t_grasp,
              label="GRASP Standard",
              linewidth=3, color=:red, marker=:circle, markersize=6,
              title="Comparaison Temps moyen",
              xlabel="Instances", ylabel="Temps (s)",
              xticks=(indices, instances), xrotation=45,
              legend=:topleft, size=(900, 500), bottom_margin=10Plots.mm)
    
    plot!(p3, indices, t_reactive,
          label="Reactive GRASP",
          linewidth=3, color=:blue, marker=:square, markersize=6)
    
    # Graphique 4 : Différence Reactive - GRASP (COURBE)
    diff_z = [rr[i].z_moy - rg[i].z_moy for i in 1:length(rg)]
    
    p4 = plot(indices, diff_z,
              label="Amélioration Reactive vs GRASP",
              linewidth=3, color=:green, marker=:circle, markersize=6,
              title="Gain de Reactive GRASP (Z_reactive - Z_grasp)",
              xlabel="Instances", ylabel="Différence Z moyen",
              xticks=(indices, instances), xrotation=45,
              legend=:topright, size=(900, 500), bottom_margin=10Plots.mm)
    hline!(p4, [0], label="Égalité", color=:black, linestyle=:dash)
    
    # Sauvegarder
    plot(p1, p2, p3, p4, layout=(2,2), size=(1800, 1200))
    savefig("res/graphiques_EI2.pdf")
    
    println("✓ res/graphiques_EI2.pdf généré")
end


# ==============================================================================
# MESSAGES D'ACCUEIL
# ==============================================================================

println("\n" * "="^80)
println("LIVRABLE MÉTAHEURISTIQUES SPP")
println("="^80)
println("\nCommandes disponibles:")
println("\n--- EI1 : Glouton + Recherche Locale ---")
println("  experimentationSPP()")
println("\n--- EI2 : GRASP vs Reactive GRASP ---")
println("  experimentationSPP_EI2()  # Par défaut: 5 runs, 100 itérations")
println("\n" * "="^80)