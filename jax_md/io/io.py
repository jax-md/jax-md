def vec2str(x):
    return "\t".join([str(i) for i in x])

def write_xyz(filename, *args):
    vars = []
    for i in range(len(args)):
        if len(args[i].shape)==2:
            vars.append(args[i])
        elif len(args[i].shape)==1:
            vars.append(args[i].reshape(-1, 1))
        else:
            pass

    with open(filename, "w+") as f:
        N = len(vars[0])
        str_ = f"{N}" + "\n\n"
        f.write(str_)
        for j in range(N):
            str_ = f"{j+1}\t" + "\t".join(map(vec2str, [var[j, :] for var in vars])) + "\n"
            f.write(str_)
