# ====================== Livrable EI1 ====================

using LinearAlgebra
using Printf
using JuMP, GLPK

include("loadSPP.jl")
include("setSPP.jl")
include("Glouton.jl")        # Heuristique gloutonne
include("Exploration.jl")    # Heuristique exploration locale


function resoudreSPP(fname::String)
    println("========== RÉSOLUTION SPP - Instance: ======== ", fname)
    
    # Chargement
    C, A = loadSPP(fname)
    m, n = size(A)
    println("Dimensions: m=$m contraintes, n=$n variables")
    
    # Phase 1: Construction gloutonne
    println("\n--- PHASE 1: Construction gloutonne ---")
    t_construct = @elapsed x_construct = construction_gloutonne(C, A)
    z_construct = dot(C, x_construct)
    println("Solution initiale: Z = $z_construct ($(sum(x_construct)) variables)")
    println("Temps: $(round(t_construct, digits=4))s")
    
    # Phase 2: Recherche locale multi-start
    println("\n--- PHASE 2: Recherche locale ---")
    t_local = @elapsed x_final = descente_multi_start(x_construct, C, A, max_restarts=5)
    z_final = dot(C, x_final)
    println("\nSolution finale: Z = $z_final ($(sum(x_final)) variables)")
    println("Amélioration: +$(round(z_final - z_construct, digits=2))")
    println("Temps total: $(round(t_construct + t_local, digits=4))s")
    
    println("\n" * "="^80)
    
    return x_final
end



"""Résolution exacte d'une instance SPP avec GLPK que pour les instances
de taille moyenne 
"""

function resoudreSPP_exact(fname::String)
    C, A = loadSPP(fname)
    m, n = size(A)
    
    # Construction du modèle JuMP
    spp = setSPP(C, A)
    set_optimizer(spp, GLPK.Optimizer)
    set_silent(spp)  
    
    # Résolution
    optimize!(spp)
    
    return (
        solution = value.(spp[:x]),
        objectif = objective_value(spp),
        temps = solve_time(spp),
        statut = termination_status(spp)
    )
end


function experimentationSPP()
    println("=========== EXPÉRIMENTATION NUMÉRIQUE - 10 INSTANCES SPP ========= ")
    println("="^80)
    
    # Les 10 instances tests qu'on a choisies
    instances = [
        "dat/pb_100rnd0100.dat",
        "dat/pb_200rnd0100.dat",
        "dat/pb_500rnd0100.dat",
        "dat/pb_1000rnd0100.dat",
        "dat/pb_2000rnd0100.dat",
        "dat/didactic.dat",
        "dat/pb_100rnd0300.dat",
        "dat/pb_200rnd0300.dat",
        "dat/pb_500rnd0300.dat",
        "dat/pb_2000rnd0800.dat"
    ]
    
    resultats = []
    
    for (idx, inst) in enumerate(instances)
        println("\n======== [$idx/$(length(instances))] Instance: $inst")
        
        # Chargement
        C, A = loadSPP(inst)
        m, n = size(A)
        
        # ===== SOLUTION EXACTE (seulement si n <= 100) =====
        z_opt = missing
        t_opt = missing
        
        if n <= 100
            println("\n")
            println("\n>>> Résolution exacte (GLPK)...")
            try
                res_exact = resoudreSPP_exact(inst)
                z_opt = res_exact.objectif
                t_opt = res_exact.temps
                println("Z* = $z_opt | Temps = $(round(t_opt, digits=3))s")
            catch e
                #timeout pour eviter d'attendre trop longtemps
                println("Résolution exacte échouée ou trop longue (timeout)")
                z_opt = missing
                t_opt = missing
            end
        else
            println("Attention ! Instance trop grande pour résolution exacte (n=$n)")
        end
        
        # ===== HEURISTIQUE CONSTRUCTION =====
        println("\n>>> Construction gloutonne...")
        t_construct = @elapsed x_construct = construction_gloutonne(C, A, verbose=false)
        z_construct = dot(C, x_construct)
        
        # Calcul du gap seulement si solution optimale disponible
        gap_construct = ismissing(z_opt) ? missing : 100 * (z_opt - z_construct) / z_opt
        
        if !ismissing(gap_construct)
            println("Z_construct = $z_construct | Gap = $(round(gap_construct, digits=2))%")
        else
            println("Z_construct = $z_construct | Gap = N/A (pas de solution optimale)")
        end
        
        # ===== RECHERCHE LOCALE =====
        println("\n>>> Recherche locale multi-start...")
        t_local = @elapsed x_local = descente_multi_start(x_construct, C, A, max_restarts=5, verbose=false)
        z_local = dot(C, x_local)
        t_total = t_construct + t_local
        
        # Calcul du gap et speedup seulement si solution optimale disponible
        gap_local = ismissing(z_opt) ? missing : 100 * (z_opt - z_local) / z_opt
        speedup = ismissing(t_opt) ? missing : t_opt / t_total
        
        if !ismissing(gap_local)
            println("Z_local = $z_local | Gap = $(round(gap_local, digits=2))%")
        else
            println("Z_local = $z_local | Gap = N/A")
        end
        
        # Stockage des résultats
        push!(resultats, (
            instance = basename(inst),
            m = m,
            n = n,
            z_opt = z_opt,
            t_opt = t_opt,
            z_construct = z_construct,
            t_construct = t_construct,
            gap_construct = gap_construct,
            z_local = z_local,
            t_total = t_total,
            gap_local = gap_local,
            speedup = speedup
        ))
    end
    
    afficher_resultats(resultats)
    sauvegarder_resultats(resultats)
    
    return resultats
end

function afficher_resultats(resultats)
    println("========== TABLEAU RÉCAPITULATIF DES RÉSULTATS =========")
    
    @printf "%-20s %5s %6s | %10s %10s | %10s %8s | %10s %10s %8s | %10s\n" "Instance" "m" "n" "Z*" "T*(s)" "Z_const" "Gap%" "Z_local" "T_tot(s)" "Gap%" "Speedup"
    println("-"^130)
    
    for r in resultats
        # Formatage conditionnel pour les valeurs manquantes
        z_opt_str = ismissing(r.z_opt) ? "N/A" : @sprintf("%.0f", r.z_opt)
        t_opt_str = ismissing(r.t_opt) ? "N/A" : @sprintf("%.3f", r.t_opt)
        gap_c_str = ismissing(r.gap_construct) ? "N/A" : @sprintf("%.2f", r.gap_construct)
        gap_l_str = ismissing(r.gap_local) ? "N/A" : @sprintf("%.2f", r.gap_local)
        speedup_str = ismissing(r.speedup) ? "N/A" : @sprintf("%.1f", r.speedup)
        
        @printf "%-20s %5d %6d | %10s %10s | %10.0f %8s | %10.0f %10.4f %8s | %10s\n" r.instance r.m r.n z_opt_str t_opt_str r.z_construct gap_c_str r.z_local r.t_total gap_l_str speedup_str
    end
    
    println("="^130)
    
    # Statistiques moyennes uniquement pour instances résolues avec solveur aussi
    resultats_avec_opt = filter(r -> !ismissing(r.gap_local), resultats)
    
    if !isempty(resultats_avec_opt)
        gap_moyen_construct = sum(r.gap_construct for r in resultats_avec_opt) / length(resultats_avec_opt)
        gap_moyen_local = sum(r.gap_local for r in resultats_avec_opt) / length(resultats_avec_opt)
        speedup_moyen = sum(r.speedup for r in resultats_avec_opt) / length(resultats_avec_opt)
        
        println("\n  STATISTIQUES MOYENNES (sur $(length(resultats_avec_opt)) instances avec solution optimale): ")
        println("  - Gap construction: $(round(gap_moyen_construct, digits=2))%")
        println("  - Gap recherche locale: $(round(gap_moyen_local, digits=2))%")
        println("  - Speedup moyen: $(round(speedup_moyen, digits=1))x")
    else
        println("\n  Aucune instance résolue à l'optimum (toutes trop grandes pour GLPK)")
    end
end

function sauvegarder_resultats(resultats)
    fichier = "res/resultats_experimentation.txt"
    
    # Créer le répertoire res/ s'il n'existe pas
    mkpath("res")
    
    open(fichier, "w") do f
        write(f, "RÉSULTATS EXPÉRIMENTATION SPP\n")
        write(f, "Date: $(now())\n")
        
        for r in resultats
            write(f, "Instance: $(r.instance)\n")
            write(f, "  Dimensions: m=$(r.m), n=$(r.n)\n")
            
            if !ismissing(r.z_opt)
                write(f, "  Z* = $(r.z_opt) ($(round(r.t_opt, digits=3))s)\n")
                write(f, "  Z_construct = $(r.z_construct) | Gap = $(round(r.gap_construct, digits=2))%\n")
                write(f, "  Z_local = $(r.z_local) | Gap = $(round(r.gap_local, digits=2))%\n")
                write(f, "  Speedup: $(round(r.speedup, digits=1))x\n")
            else
                write(f, "  Z* = N/A (instance trop grande)\n")
                write(f, "  Z_construct = $(r.z_construct)\n")
                write(f, "  Z_local = $(r.z_local)\n")
                write(f, "  Temps total heuristiques: $(round(r.t_total, digits=4))s\n")
            end
            write(f, "\n")
        end
    end
    println("\n Résultats sauvegardés dans: $fichier")
end

# Message d'accueil
println(" ======= LIVRABLE EI1 - SET PACKING PROBLEM =======")
println("Commandes disponibles:")
println("  1. resoudreSPP(\"dat/pb_100rnd0100.dat\")")
println("  2. experimentationSPP()")