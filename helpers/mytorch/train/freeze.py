
def mark_freezed(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad  = False

def mark_unfreezed(model):
    model.train()
    for p in model.parameters():
        p.requires_grad = True

