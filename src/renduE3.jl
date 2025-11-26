using LinearAlgebra
using Statistics
using CSV
include("loadSPP.jl")
include("Glouton.jl")
include("Exploration.jl")
include("Grasp.jl")
include("ACO.jl")
include("AG.jl")

OUT_DIR = "../doc/"
function compare_metaheuristics(dir::String;verbose::Bool=false)

    # Fichier CSV de résultats (dans le même dossier que les données)
    results_file = joinpath(OUT_DIR, "meta_results.csv")
    if !isfile(results_file)
        open(results_file, "w") do io
            println(io, "filename,method,time,z")
        end
    end

    # Petit utilitaire pour ajouter une ligne dans le CSV (quote le nom de fichier)
    append_result(file::String, method::String, time::Real, z) = open(results_file, "a") do io       
        println(io, "$file,$method,$time,$z")
    end

    # Liste tous les fichiers du dossier
    files = readdir(dir)


    println("===================================================")
    println(" Lancement des expérimentations de Meta sur le dossier : $dir")
    println(" Fichiers détectés : ", length(files))
    println("===================================================\n")

    for file in files
        
        # Ignore les fichiers qui ne sont pas des .dat 
        endswith(file, ".dat") || continue
        
        path = joinpath(dir, file)
        
        
        println("\n---------------- FICHIER : $file ----------------")

        try
            C, A = loadSPP(path)
            conf = precompute_conflicts(A)
            
        
            println("Lancement de GRASP...")
            time_GRASP = @elapsed best_grasp,_,_,_ =  resoudreSPP_GRASP(path, methode="grasp", n_iter=50,verbose=false)
            sol_GRASP = dot(C, best_grasp)
            append_result(file, "GRASP", time_GRASP, sol_GRASP)
            
            #= println("Lancement de l'AG...")
            time_AG = @elapsed  best_val_ag, _ = algorithme_genetique_simple(C, A, 
                                                    taille_pop=80,
                                                    generations=250,
                                                    prob_croisement=0.9,
                                                    prob_mutation=0.02,
                                                    freq_recherche_locale=50,
                                                    verbose=false)

            sol_AG = dot(C, best_val_ag)
            append_result(file, "AG", time_AG, sol_AG) =#
        
        println("Lancement de l'ACO...")
            time_ACO = @elapsed best_sol, best_val = ACO_SPP(
                C, A, conf,
                maxIter = 50,
                maxAnt = 25,
                rhoE = 0.8,
                phiInit = 1.0,
                iterOnExploit = 0.75,
                iterStagnant = 8,
                verbose = false
            )
            append_result(file, "ACO", time_ACO, best_val)

            println("Résultat GRASP pour $file : Best value = $sol_GRASP")
            println("Résultat AG pour $file : Best value = $sol_AG")
            println("Résultat ACO pour $file : Best value = $best_val")

        catch e
            println("   ERREUR lors du traitement de $file :")
            println(e)
            continue
        end
    end

    println("\n===================================================")
    println("       EXPÉRIMENTATION META TERMINÉE")
    println("===================================================")
end

compare_metaheuristics("../dat/",verbose=true)


function compare_metaheuristics_GRASP_ACO(dir::String; verbose::Bool=false)

    # Fichier CSV principal
    results_file = joinpath(OUT_DIR, "meta_results_ACO_GRASP.csv")
    if !isfile(results_file)
        open(results_file, "w") do io
            println(io, "filename,method,time,z")
        end
    end

    # Fichier pour les courbes ACO
    trace_file = joinpath(OUT_DIR, "aco_trace.csv")
    if !isfile(trace_file)
        open(trace_file, "w") do io
            println(io, "filename,time,zmin,zmax,zmoy")
        end
    end

    append_result(file, method, time, z) = open(results_file, "a") do io
        println(io, "$file,$method,$time,$z")
    end

    append_trace(file, t, zmin, zmax, zmoy) = open(trace_file, "a") do io
        println(io, "$file,$t,$zmin,$zmax,$zmoy")
    end

    # Liste des fichiers
    files = readdir(dir)

    println("===================================================")
    println(" Lancement des expérimentations sur le dossier : $dir")
    println(" Fichiers détectés : ", length(files))
    println("===================================================\n")

    for file in files

        endswith(file, ".dat") || continue
        path = joinpath(dir, file)

        println("\n---------------- FICHIER : $file ----------------")

        try
            C, A = loadSPP(path)
            conf = precompute_conflicts(A)

            # === GRASP ===
            println("Lancement de GRASP...")
            time_GRASP = @elapsed best_grasp, _, _, _ =
                resoudreSPP_GRASP(path, methode="grasp", n_iter=50, verbose=false)

            sol_GRASP = dot(C, best_grasp)
            append_result(file, "GRASP", time_GRASP, sol_GRASP)

            # === ACO ===
            println("Lancement de l'ACO...")

            time_ACO = @elapsed begin
                best_sol, best_val, history = ACO_SPP(
                    C, A, conf;
                    maxIter = 50,
                    maxAnt = 25,
                    rhoE = 0.8,
                    phiInit = 1.0,
                    iterOnExploit = 0.75,
                    iterStagnant = 8,
                    verbose = false,
                    measure_interval = 10.0      
                )
            end

            append_result(file, "ACO", time_ACO, best_val)

            # Ajout des mesures temporelles ACO
            for (t, zmin, zmax, zmoy) in history
                append_trace(file, t, zmin, zmax, zmoy)
            end

            println("Résultat GRASP : Best = $sol_GRASP")
            println("Résultat ACO   : Best = $best_val")

        catch e
            println("   ERREUR sur le fichier $file :")
            println(e)
            continue
        end
    end

    println("\n===================================================")
    println("       EXPÉRIMENTATION META TERMINÉE")
    println("===================================================")
end

using Plots
using CSV
using DataFrames
using Measures

function plotting_results_time(file_result::String="meta_results.csv")
    results_file = joinpath(OUT_DIR, file_result)

    if !isfile(results_file)
        println(" Aucun fichier de résultats trouvé : $results_file")
        return
    end

    if !isdir(PLOT_DIR)
        mkpath(PLOT_DIR)
    end

    df = CSV.read(results_file, DataFrame)

    methods = unique(df.method)

    for m in methods
        println("\n--- Bar chart du temps pour : $m ---")

        sub = filter(row -> row.method == m, df)

        if nrow(sub) == 0
            println("Rien à tracer pour la méthode $m")
            continue
        end

        # tri pour avoir les instances dans l'ordre alphanumérique
        sort!(sub, :filename)

        files = sub.filename
        t = sub.time

        p = bar(
        files,
        t,
        title = "Temps d'exécution - Méthode $m",
        xlabel = "Instance",
        ylabel = "Temps (s)",
        legend = false,

        rotation = 60,
        size = (1100, 600),

        left_margin = 12mm,
        right_margin = 20mm,
        top_margin = 20mm,
        bottom_margin = 12mm,

        xtickfont = font(9),
        ytickfont = font(9),
        guidefont = font(10),
        titlefont = font(12)
    )

        savefig(p, joinpath(PLOT_DIR, "plot_time_bars_$m.png"))
        println(" Sauvegardé  plot_time_bars_$m.png")
    end

    println("\n Tous les bar charts des temps générés.")
end

#plotting_results_time()