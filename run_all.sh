'''
vanilla FIB pomdp solver

python solve_FIB.py FIB POMDP_file repeat_number precision (sample_number)
python evaluate_FIB.py FIB POMDP_file number_runs

-> POMDP_file : select the POMDP to be solved (among the files under examples/env)
-> repeat_number : number of initializations
-> precision : precision to stop the algorithm
-> sample_number : run simulated version of FIB update with sample number

AA-FIB pomdp solver

python solve_AA_FIB.py FIB POMDP_file repeat_number precision lambda (sample_number)
python evaluate_AA_FIB.py FIB POMDP_file number_runs lambda

-> POMDP_file : select the POMDP to be solved (among the files under examples/env)
-> repeat_number : number of initializations
-> lambda: regularization parameter eta in paper
-> number_runs : number of runs for the evaluation
-> precision : precision to stop the algorithm
-> sample_number : run simulated version of AA-FIB update with sample number
'''
'''

for env in "fourth" "TagAvoid"; do
    for max_type in "mellowmax" "logsumexp"; do
        for softmax_param in 1e-1 1 10 100 1e4; do
            python main.py --safeguard safe_global --env_name ${env} --max_type ${max_type} --softmax_param ${softmax_param}  --do_eval --filename softmax_test.txt
        done
    done
done

for env in "cit" "mit" "sunysb" "pentagon" "fourth" "TagAvoid"; do
    for max_type in "standard" "mellowmax"; do
        for safeguard in 0.2 0.4 0.6 0.8 1.0; do
            python main.py --safeguard opt_gain --env_name ${env} --safeguard_coeff ${safeguard} --max_type ${max_type} --filename test_0422.txt
        done
    done
done
'''

# Standard
python main.py --FPI --do_eval
python main.py --safeguard safe_global --do_eval
python main.py --safeguard strict_local --do_eval
for safeguard in 1e-2 1 1e2 1e4; do
    python main.py --safeguard safe_local --safeguard_coeff ${safeguard} --do_eval
done

# Regularized
for max_type in "logsumexp"; do
    # Reg parameter
    for softmax_param in 10 1e3 1e5; do
        python main.py --FPI --max_type ${max_type} --softmax_param ${softmax_param} --do_eval
        python main.py --safeguard safe_global --max_type ${max_type} --softmax_param ${softmax_param} --do_eval
        python main.py --safeguard strict_local --max_type ${max_type} --softmax_param ${softmax_param} --do_eval
        for safeguard in 1e-2 1 1e2 1e4; do
            python main.py --safeguard safe_local --safeguard_coeff ${safeguard} --max_type ${max_type} --softmax_param ${softmax_param} --do_eval
        done
    done
done

# Perform all over again with sampled operator
# Standard QMDP
python main.py --FPI --do_eval --QMDP --sample
python main.py --safeguard safe_global --do_eval --QMDP --sample
python main.py --safeguard strict_local --do_eval --QMDP --sample
for safeguard in 1e-2 1 1e2 1e4; do
    python main.py --safeguard safe_local --safeguard_coeff ${safeguard} --do_eval --QMDP --sample
done

# Regularized Bellman
for max_type in "logsumexp" "mellowmax"; do
    # Reg parameter
    for softmax_param in 10 1e3 1e5; do
        python main.py --FPI --max_type ${max_type} --softmax_param ${softmax_param} --do_eval --QMDP --sample
        python main.py --safeguard safe_global --max_type ${max_type} --softmax_param ${softmax_param} --do_eval --QMDP --sample
        python main.py --safeguard strict_local --max_type ${max_type} --softmax_param ${softmax_param} --do_eval --QMDP --sample
        for safeguard in 1e-2 1 1e2 1e4; do
            python main.py --safeguard safe_local --safeguard_coeff ${safeguard} --max_type ${max_type} --softmax_param ${softmax_param} --do_eval --QMDP --sample
        done
    done
done
