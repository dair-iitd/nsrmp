
def str2bool(string):
    s = string.lower()
    if s in ['true', 'yes', 't', '1']:
        return True
    elif s in ['false', 'no', 'f', '0']:
        return False
    else:
        raise ValueError("Couldn't convert the string {string} into boolean")