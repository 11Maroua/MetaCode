using LinearAlgebra

# ---- Construction Gloutonne pour SPP ----

function construction_gloutonne(C, A)
    m, n = size(A)
    x = zeros(Int, n)        
    variables_restantes = collect(1:n)

    println("=== DEBUT CONSTRUCTION GLOUTONNE ===")
    println("m = ", m, " contraintes, n = ", n, " variables")
    println()
    
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
    
    println("\n=== SOLUTION FINALE ===")
    println("Solution: ", x)
    println("Variables sélectionnées: ", findall(x->x==1, x))
    println("Nombre de variables: ", nb_variables)
    println("Valeur totale Z = : ", valeur_totale)
    
    return x
end

function peut_ajouter(x, A, j_new)
   
    m, n = size(A)
    
    for i in 1:m
        if A[i, j_new] == 1
            # Vérifier si une variable déjà sélectionnée couvre cette contrainte
            for k in 1:n
                if x[k] == 1 && A[i, k] == 1
                    return false  # Conflit détecté
                end
            end
        end
    end
    return true
end

