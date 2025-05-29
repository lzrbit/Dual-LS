import re

def extract_metrics(file_path):
    fde_pattern = r"The averaged FDE of all tasks\s*:\s*(-?\d+\.\d+)"
    mr_pattern = r"The averaged Missing Rate of all tasks\s*:\s*(-?\d+\.\d+)"
    fde_bwt_pattern = r"FDE backward transfer\s*:\s*(-?\d+\.\d+)"
    mr_bwt_pattern = r"Missing Rate backward transfer\s*:\s*(-?\d+\.\d+)"
    
    metrics = {}

    with open(file_path, 'r') as file:
        content = file.read()

        fde_match = re.search(fde_pattern, content)
        mr_match = re.search(mr_pattern, content)
        fde_bwt_match = re.search(fde_bwt_pattern, content)
        mr_bwt_match = re.search(mr_bwt_pattern, content)

        if fde_match:
            metrics['FDE-AVE'] = float(fde_match.group(1))
        if mr_match:
            metrics['MR-AVE'] = float(mr_match.group(1))
        if fde_bwt_match:
            metrics['FDE_BWT'] = float(fde_bwt_match.group(1))
        if mr_bwt_match:
            metrics['MR_BWT'] = 100*float(mr_bwt_match.group(1)) 

    return metrics
