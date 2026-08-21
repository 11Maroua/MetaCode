using LinearAlgebra

# ---- Construction Gloutonne pour SPP ----

function construction_gloutonne(C, A; verbose=true)
    m, n = size(A)
    x = zeros(Int, n)        
    variables_restantes = collect(1:n)

    if verbose
        println("=== DEBUT CONSTRUCTION GLOUTONNE ===")
        println("m = ", m, " contraintes, n = ", n, " variables")
        println()
    end
    
    while !isempty(variables_restantes)
        meilleure_utilite = -Inf
        meilleur_j = -1
        
        # Évaluer toutes les variables restantes
        for j in variables_restantes
            if peut_ajouter(x, A, j)
                nb_contraintes = sum(A[:, j])
                #calcul des utilités des variables pour pouvoir choisir la meilleure d'abord
                if nb_contraintes > 0
                    utilite = C[j] / nb_contraintes
                else
                    utilite = C[j]  # Cas particulier: Variable sans contraintes !
                end
                
                if utilite > meilleure_utilite
                    meilleure_utilite = utilite
                    meilleur_j = j
                end
            end
        end
        
        if meilleur_j == -1
            break  # Plus aucune variable candidate
        end
        
        # Ajouter la meilleure variable
        x[meilleur_j] = 1
        
        # Supprimer la variable sélectionnée de notre ensemble de candidats
        idx_supprimer = findfirst(==(meilleur_j), variables_restantes)
        deleteat!(variables_restantes, idx_supprimer)
        
    end
    
    # RÉSULTAT FINAL
    valeur_totale = dot(C, x)
    nb_variables = sum(x)
    
    if verbose
        println("\n=== SOLUTION FINALE ===")
        println("Solution: ", x)
        println("Variables sélectionnées: ", findall(x->x==1, x))
        println("Nombre de variables: ", nb_variables)
        println("Valeur totale Z = : ", valeur_totale)
    end
    
    return x
end

"""
    peut_ajouter(x, A, j_new)

Teste si la variable `j_new` peut être ajoutée à la solution partielle `x`
sans violer de contrainte (i.e. sans entrer en conflit avec une variable
déjà sélectionnée). Fonction partagée par tous les algorithmes du projet
(glouton, recherche locale, GRASP).
"""
"""
    utilite_variable(C, A, j)

Ratio d'utilité de la variable `j` : c_j / (nombre de contraintes qu'elle occupe).
Même métrique que dans la construction gloutonne, réutilisée comme heuristique
locale (`eta`) par l'ACO.
"""
function utilite_variable(C, A, j)
    nb_contraintes = sum(@view A[:, j])
    return nb_contraintes > 0 ? C[j] / nb_contraintes : Float64(C[j])
end

function peut_ajouter(x, A, j_new)
    for i in 1:size(A, 1)
        if A[i, j_new] == 1
            for k in findall(x .== 1)  # ne parcourt que les variables actives
                if A[i, k] == 1
                    return false  # conflit détecté
                end
            end
        end
    end
    return true
end