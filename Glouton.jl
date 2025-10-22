using LinearAlgebra

# ---- Exercice1 ------

function construction_gloutonne(C, A)
    m, n = size(A)
    x = zeros(Int, n)        
    A_restant = copy(A)
    variables_restantes = collect(1:n)

    println("Début construction gloutonne:")
    println("m = ", m, " contraintes, n = ", n, " variables")
    println()
    
    iteration = 1
    while !isempty(variables_restantes)
        utilites = Float64[]
        candidates = Int[]
        
        for j in variables_restantes
            if peut_ajouter(x, A, j)
                nb_contraintes = sum(A_restant[:, j])
                utilite = (nb_contraintes > 0) ? C[j] / nb_contraintes : 0.0
                push!(utilites, utilite)
                push!(candidates, j)
            end
        end
        
        if isempty(candidates)
            break
        end
        
        idx_meilleur = argmax(utilites)
        j_best = candidates[idx_meilleur]
        x[j_best] = 1
        
        
        
        contraintes_supprimees = findall(A_restant[:, j_best] .== 1)
        if !isempty(contraintes_supprimees)
            A_restant = A_restant[setdiff(1:size(A_restant,1), contraintes_supprimees), :]
        end
        
        idx_supprimer = findfirst(==(j_best), variables_restantes)
        deleteat!(variables_restantes, idx_supprimer)
        
       
        iteration += 1
    end
    
    println("=== CONSTRUCTION TERMINÉE ===")
    println("Solution: ", x)
    println("Valeur: ", dot(C, x))
    
    return x
end

function peut_ajouter(x, A, j_new)
    m, n = size(A)
    for i in 1:m
        if A[i, j_new] == 1
            for k in 1:n
                if x[k] == 1 && A[i, k] == 1
                    return false
                end
            end
        end
    end
    return true
end

# -------TEST SUR didactic.dat------

include("loadSPP.jl")

function tester_didactic()
    println("-----------TEST ÉTAPE 1 - CONSTRUCTION GLoutONNE--------")
    fname = "Data/pb_200rnd0100.dat"
    C, A = loadSPP(fname)
    
    println("Instance: ", fname)
    println("Coûts C = ", C)
    println("Dimensions: ", size(A, 1), " contraintes × ", size(A, 2), " variables")
    println()
    
    x_glouton = construction_gloutonne(C, A)
    
    println()
    println("Solution construite avec succès!")
end

@time begin
tester_didactic()
end 
