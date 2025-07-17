for env in "mit" "TagAvoid" "under_water"; do
    for lamb in 1e-16 1e-24 1e-32; do
        python SAA-QMDP/main.py FIB ${env} 100 1e-6 ${lamb} 10
    done
done



for env in "mit" "TagAvoid" "under_water"; do
    for lamb in 1e-16 1e-24 1e-32; do
        python SAA-FIB/main.py FIB ${env} 100 1e-6 ${lamb} 10
    done
done

