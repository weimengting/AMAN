


def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

