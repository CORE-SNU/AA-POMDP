with open('mit_final.txt') as f:
    lines = f.readlines()

algs = ['FIB', 'strict_local', 'safe_global', 'safe_local_0.01', 'safe_local_1.0', 'safe_local_100.0', 'safe_local_10000.0']
for alg in algs:
    it, aa, time, ret = "", "", "", ""
    it_, aa_, time_, ret_ = "", "", "", ""
    record = False
    if alg == 'FIB':
        for line in lines:
            l = line.split()
            if len(l):
                if l[0] == 'Solver':
                    if l[-1] == 'FIB':
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

                if record:
                    if l[0] == 'Total':
                        it_ = it + " & " + l[-3] + " " + l[-2] + " " + l[-1]
                    if l[0] == '#':
                        aa_ = aa + " & " + l[-3] + " " + l[-2] + " " + l[-1]
                    if l[0] == 'Ellapsed':
                        time_ = time + " & " + l[-3] + " " + l[-2] + " " + l[-1]
                    if l[0] == 'Return':
                        ret_ = ret + " & " + l[-3] + " " + l[-2] + " " + l[-1]
                        record = False

                if "==========" in l[0]:
                    print("& $\\#\\mathrm{iter}$ " + it + " \\\\")
                    # print("& $\\#\\mathrm{AA}$" + aa + " \\\\")
                    print("& $t_{\\mathrm{step}}$" + time + " \\\\")
                    print("& $\\mathrm{return}_\\mathrm{fixed}$" + ret + " \\\\")
                    ret, ret_ = "", ""

        print("& $\\mathrm{return}_\\mathrm{rand}$" + ret + " \\\\")
        print("")
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
                    if l[0] == 'Return':
                        ret = ret + " & " + l[-3] + " " + l[-2] + " " + l[-1]
                        record = False

                if "==========" in l[0]:
                    ret_ = ret
                    it, aa, time, ret = "", "", "", ""

        print("& $\\#\\mathrm{iter}$ " + it + " \\\\")
        print("& $\\#\\mathrm{AA}$" + aa + " \\\\")
        print("& $t_{\\mathrm{step}}$" + time + " \\\\")
        print("& $\\mathrm{return}_\\mathrm{fixed}$" + ret_ + " \\\\")
        print("& $\\mathrm{return}_\\mathrm{rand}$" + ret + " \\\\")
        print("")