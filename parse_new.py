with open('underwater_fib.txt') as f:
    lines = f.readlines()

algs = ['FIB', 'safe_global', 'safe_local_0.01', 'safe_local_1.0', 'safe_local_100.0', 'safe_local_10000.0']
for alg in algs:
    it, aa, time, ret, ret_rand = "", "", "", "", ""
    it_, aa_, time_, ret_, ret_rand_ = "", "", "", "", ""
    record = False
    if alg == algs[0]:
        for line in lines:
            l = line.split()
            if len(l):
                if l[0] == 'Solver':
                    if l[-1] == algs[0]:
                        record = True
                    else:
                        record = False
                        if 'logsumexp_0.1' in l[-1]:
                            pass
                        elif 'mellowmax_0.1' in l[-1]:
                            pass
                        else:
                            it = it_
                            aa = aa_
                            time = time_
                            ret = ret_
                            ret_rand = ret_rand_

                if record:
                    if l[0] == 'Total':
                        it_ = it + " & " + l[-3] + " " + l[-2] + " " + l[-1]
                    if l[0] == '#':
                        aa_ = aa + " & " + l[-3] + " " + l[-2] + " " + l[-1]
                    if l[0] == 'Ellapsed':
                        time_ = time + " & " + l[-3] + " " + l[-2] + " " + l[-1]
                    if l[0] == 'Return' and l[1] == '(fixed)':
                        ret_ = ret + " & " + l[-3] + " " + l[-2] + " " + l[-1]
                    if l[0] == 'Return' and l[1] == '(rand)':
                        ret_rand_ = ret_rand + " & " + l[-3] + " " + l[-2] + " " + l[-1]
                        record = False

        print("& $\\#\\mathrm{iter}$ " + it + " \\\\")
        print("& $t_{\\mathrm{total}}$" + time + " \\\\")
        print("& $\\mathrm{return}_\\mathrm{fixed}$" + ret + " \\\\")
        print("& $\\mathrm{return}_\\mathrm{rand}$" + ret_rand + " \\\\")
        print("\hline\n")
    else:
        for line in lines:
            l = line.split()
            if len(l):
                if l[0] == 'Solver':
                    if alg in l[-1]:
                        record = True
                    else:
                        record = False

                    if 'logsumexp_0.1' in l[-1]:
                        record = False
                    if 'mellowmax_0.1' in l[-1]:
                        record = False

                if record:
                    if l[0] == 'Total':
                        it = it + " & " + l[-3] + " " + l[-2] + " " + l[-1]
                    if l[0] == '#':
                        aa = aa + " & " + l[-3] + " " + l[-2] + " " + l[-1]
                    if l[0] == 'Ellapsed':
                        time = time + " & " + l[-3] + " " + l[-2] + " " + l[-1]
                    if l[0] == 'Return' and l[1] == '(fixed)':
                        ret = ret + " & " + l[-3] + " " + l[-2] + " " + l[-1]
                    if l[0] == 'Return' and l[1] == '(rand)':
                        ret_rand = ret_rand + " & " + l[-3] + " " + l[-2] + " " + l[-1]
                        record = False

        print("& $\\#\\mathrm{iter}$ " + it + " \\\\")
        print("& $\\#\\mathrm{AA}$" + aa + " \\\\")
        print("& $t_{\\mathrm{total}}$" + time + " \\\\")
        print("& $\\mathrm{return}_\\mathrm{fixed}$" + ret + " \\\\")
        print("& $\\mathrm{return}_\\mathrm{rand}$" + ret_rand + " \\\\")
        print("\hline\n")