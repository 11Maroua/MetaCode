using LinearAlgebra
include("loadSPP.jl")
include("Glouton.jl")  
include("ExplorationLocale.jl")   


function construction_grasp(C, A, α)
    m, n = size(A)
    x = zeros(Int, n)
    
    contraintes_non_couvertes = Set(1:m)
    candidats = Set(1:n)
    
    while !isempty(contraintes_non_couvertes) && !isempty(candidats)
        # RÉUTILISATION: même calcul d'utilité que la version gloutonne
        utilites = calcul_utilites_candidats(candidats, contraintes_non_couvertes, C, A)
        
        if isempty(utilites)
            break
        end
        
        # MODIFICATION GRASP: au lieu de prendre le meilleur, on construit une RCL
        rcl = construire_rcl(utilites, α)
        
        # MODIFICATION GRASP: sélection aléatoire au lieu de gloutonne
        j_choisi = rand(rcl)
        
        # RÉUTILISATION: même logique de mise à jour qu'en glouton
        x[j_choisi] = 1
        delete!(candidats, j_choisi)
        
        contraintes_couvertes = Set{Int}()
        for i in contraintes_non_couvertes
            if A[i, j_choisi] == 1
                push!(contraintes_couvertes, i)
            end
        end
        setdiff!(contraintes_non_couvertes, contraintes_couvertes)
        
        candidats_en_conflit = Set{Int}()
        for j in candidats
            if est_en_conflit_avec(j, j_choisi, A)
                push!(candidats_en_conflit, j)
            end
        end
        setdiff!(candidats, candidats_en_conflit)
    end
    
    return x
end


#Calcul des utilités pour les candidats

function calcul_utilites_candidats(candidats, contraintes_non_couvertes, C, A)
    utilites = Dict{Int, Float64}()
    
    for j in candidats
        nb_contraintes = count(i -> A[i, j] == 1, contraintes_non_couvertes)
        
        if nb_contraintes > 0 && C[j] > 0
            utilites[j] = nb_contraintes / C[j]
        end
    end
    
    return utilites
end


#construction de la RCL selon utilité limite définie

function construire_rcl(utilites, α)
    if isempty(utilites)
        return Int[]
    end
    
    u_min = minimum(values(utilites))
    u_max = maximum(values(utilites))
    u_limit = u_min + α * (u_max - u_min)
    
    rcl = [j for (j, u) in utilites if u >= u_limit]
    
    return rcl
end


function est_en_conflit_avec(j1, j2, A)
    m = size(A, 1)
    for i in 1:m
        if A[i, j1] == 1 && A[i, j2] == 1
            return true
        end
    end
    return false
end


function grasp(C, A, α, n_iterations; verbose=true)
    if verbose
       
        println("---------------DÉMARRAGE GRASP---------------")
        println("  Paramètres: α = $α, n_iterations = $n_iterations")
      
    end
    
    x_best = nothing
    z_best = -Inf
    
    historique_z_init = Float64[]
    historique_z_ls = Float64[]
    
    for iter in 1:n_iterations
        # PHASE 1: Construction gloutonne randomisée
        x_init = construction_grasp(C, A, α)
        z_init = dot(C, x_init)
        push!(historique_z_init, z_init)
        
        # PHASE 2: RÉUTILISATION de la recherche locale existante !
        x_ameliore, z_ameliore = recherche_locale(x_init, C, A)
        push!(historique_z_ls, z_ameliore)
        
        # Mise à jour de la meilleure solution
        if z_ameliore > z_best
            x_best = copy(x_ameliore)
            z_best = z_ameliore
            if verbose
                println("Iter $iter: ★ NOUVELLE MEILLEURE → z = $z_best (init: $z_init)")
            end
        else
            if verbose && iter % 20 == 0
                println("Iter $iter: z = $z_ameliore (init: $z_init) | Best = $z_best")
            end
        end
    end
    
    if verbose
        println("  Meilleure solution: z = $z_best")
        println("  Moyenne z_init: ", round(mean(historique_z_init), digits=2))
        println("  Moyenne z_LS: ", round(mean(historique_z_ls), digits=2))
        
    end
    
    return x_best, z_best, historique_z_init, historique_z_ls
end


# ========== MAIN ==========

function main()
    
    
end

@time begin
    main()
end