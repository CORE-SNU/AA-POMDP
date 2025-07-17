for env in "mit" "TagAvoid" "under_water"; do
    for lamb in 1e-16 1e-24 1e-32; do
        python solve_AA_FIB.py FIB ${env} 100 1e-6 ${lamb} 10
        python evaluate_AA_FIB.py FIB ${env} 100 ${lamb}
    done
done

