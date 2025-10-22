using LinearAlgebra
include("loadSPP.jl")

# Structure pour maintenir l'état de couverture
mutable struct EtatCouverture
    nb_couverture::Vector{Int} #nb de variables qui couvrent une contrainte i 
    variables_actives::Set{Int}
    variables_inactives::Set{Int}
end



function creer_etat(x, A)
    m, n = size(A)
    nb_couverture = zeros(Int, m)
    
    for i in 1:m
        for j in 1:n
            if x[j] == 1 && A[i, j] == 1
                nb_couverture[i] += 1
            end
        end
    end
    
    variables_actives = Set(findall(x .== 1))
    variables_inactives = Set(findall(x .== 0))
    
    return EtatCouverture(nb_couverture, variables_actives, variables_inactives)
end





function peut_ajouter_retirer(etat, A, j_add, j_remove)
    m = size(A, 1)
    
    for i in 1:m
        if A[i, j_add] == 1
            nb_apres = etat.nb_couverture[i]
            if A[i, j_remove] == 1
                nb_apres -= 1
            end
            nb_apres += 1
            
            if nb_apres > 1
                return false
            end
        end
    end
    return true
end




function recherche_locale(x, C, A)
    x_courant = copy(x)
    z_courant = dot(C, x_courant)
    amelioration = true
    
    println("-----Début recherche locale...------")
    println("Solution initiale: z = ", z_courant)
    println()
    
    etat = creer_etat(x_courant, A)
    
    while amelioration
        amelioration = false
        
        #Voisinage 1-1
        x_ameliore, z_ameliore, etat_ameliore, trouve = chercher_amelioration_11(x_courant, z_courant, C, A, etat)
        if trouve
            x_courant = x_ameliore
            z_courant = z_ameliore
            etat = etat_ameliore
            amelioration = true
            println("✓ Amélioration trouvée: z = ", z_courant)
            continue
        end
        
        #Voisinage 2-1
        x_ameliore, z_ameliore, etat_ameliore, trouve = chercher_amelioration_21(x_courant, z_courant, C, A, etat)
        if trouve
            x_courant = x_ameliore
            z_courant = z_ameliore
            etat = etat_ameliore
            amelioration = true
            println("✓ Amélioration trouvée: z = ", z_courant)
        end
    end
    
    println("=== RECHERCHE LOCALE TERMINÉE ===")
    println("Solution améliorée: ", x_courant)
    println("Valeur finale: ", z_courant)
    
    return x_courant, z_courant
end




function chercher_amelioration_11(x, z_actuel, C, A, etat)
    inactives = collect(etat.variables_inactives)
    actives = collect(etat.variables_actives)
    
    for j_add in inactives
        for j_remove in actives
            gain = C[j_add] - C[j_remove]
            
            if gain > 0 && peut_ajouter_retirer(etat, A, j_add, j_remove)
                x_voisin = copy(x)
                x_voisin[j_add] = 1
                x_voisin[j_remove] = 0
                
                nouvel_etat = copier_etat(etat)
                mettre_a_jour_etat!(nouvel_etat, A, j_add, j_remove, nothing, nothing)
                
                return x_voisin, z_actuel + gain, nouvel_etat, true
            end
        end
    end
    
    return x, z_actuel, etat, false
end




function chercher_amelioration_21(x, z_actuel, C, A, etat)
    inactives = collect(etat.variables_inactives)
    actives = collect(etat.variables_actives)
    
    max_essais = 3000
    essais = 0
    
    for i1 in 1:length(inactives)
        j1_add = inactives[i1]
        for i2 in i1+1:length(inactives)
            j2_add = inactives[i2]
            for j_remove in actives
                essais += 1
                if essais > max_essais
                    return x, z_actuel, etat, false
                end
                
                gain = C[j1_add] + C[j2_add] - C[j_remove]
                
                if gain > 0 && peut_ajouter_retirer_2(etat, A, j1_add, j2_add, j_remove)
                    x_voisin = copy(x)
                    x_voisin[j1_add] = 1
                    x_voisin[j2_add] = 1
                    x_voisin[j_remove] = 0
                    
                    nouvel_etat = copier_etat(etat)
                    mettre_a_jour_etat!(nouvel_etat, A, j1_add, j_remove, j2_add, nothing)
                    
                    return x_voisin, z_actuel + gain, nouvel_etat, true
                end
            end
        end
    end
    
    return x, z_actuel, etat, false
end




function peut_ajouter_retirer_2(etat, A, j1_add, j2_add, j_remove)
    m = size(A, 1)
    
    for i in 1:m
        nb = etat.nb_couverture[i]
        
        if A[i, j_remove] == 1
            nb -= 1
        end
        if A[i, j1_add] == 1
            nb += 1
        end
        if A[i, j2_add] == 1
            nb += 1
        end
        
        if nb > 1
            return false
        end
    end
    return true
end



function copier_etat(etat)
    return EtatCouverture(copy(etat.nb_couverture), copy(etat.variables_actives), copy(etat.variables_inactives))
end



function mettre_a_jour_etat!(etat, A, j_add, j_remove, j_add2, j_remove2)
    m = size(A, 1)
    
    if j_remove !== nothing
        delete!(etat.variables_actives, j_remove)
        push!(etat.variables_inactives, j_remove)
        for i in 1:m
            if A[i, j_remove] == 1
                etat.nb_couverture[i] -= 1
            end
        end
    end
    
    if j_remove2 !== nothing
        delete!(etat.variables_actives, j_remove2)
        push!(etat.variables_inactives, j_remove2)
        for i in 1:m
            if A[i, j_remove2] == 1
                etat.nb_couverture[i] -= 1
            end
        end
    end
    
    if j_add !== nothing
        push!(etat.variables_actives, j_add)
        delete!(etat.variables_inactives, j_add)
        for i in 1:m
            if A[i, j_add] == 1
                etat.nb_couverture[i] += 1
            end
        end
    end
    
    if j_add2 !== nothing
        push!(etat.variables_actives, j_add2)
        delete!(etat.variables_inactives, j_add2)
        for i in 1:m
            if A[i, j_add2] == 1
                etat.nb_couverture[i] += 1
            end
        end
    end
end


function tester_recherche_locale()
    println("--------------TEST ÉTAPE 2 - RECHERCHE LOCALE----------")
    
    fname = "Data/pb_200rnd0100.dat"
    C, A = loadSPP(fname)
    
    println("Instance: ", fname)
    println("Coûts C = ", C)
    println("Dimensions: ", size(A, 1), " contraintes × ", size(A, 2), " variables")
    println()
    
    x_init =  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    z_init = dot(C, x_init)
    println("Solution initiale: ", x_init)
    println("Valeur initiale: ", z_init)
    println()
    
    x_ameliore, z_ameliore = recherche_locale(x_init, C, A)
    
    println()
    println("Recherche locale terminée avec succès!")
end



@time begin
    tester_recherche_locale()
end