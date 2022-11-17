from torchmetrics.functional import mean_absolute_error

def calculate_metrics(inf, gt, type):
    if type == "mae":
        return mean_absolute_error(inf, gt).item()