"""CPU-safe helpers for selecting partial encoder scopes and pairing images."""
import re


def parse_scope(spec):
    if spec == 'all':
        return None, None
    match = re.fullmatch(r'last_k:([A-Za-z0-9_.-]+):([1-9][0-9]*)', spec)
    if not match:
        raise ValueError(f'Invalid scope: {spec}')
    return match.group(1), int(match.group(2))


def discover_blocks(model):
    """Find repeated lists on the vision path; keys are named-module paths."""
    found = {}
    for name, module in model.named_modules():
        if (any(word in name.lower() for word in ('vision', 'visual', 'image'))
                and module.__class__.__name__ in ('ModuleList', 'Sequential')
                and len(module) and len({type(child) for child in module}) == 1):
            found[name] = module
    return found


def select_scope(model, spec, blocks, projections):
    name, k = parse_scope(spec)
    if name is not None and (name not in blocks or k > len(blocks[name])):
        raise ValueError(f'Unknown block list or k out of bounds: {spec}')
    for p in model.parameters():
        p.requires_grad = name is None
    if name is not None:
        for module in list(blocks[name])[-k:] + list(projections.values()):
            for p in module.parameters():
                p.requires_grad = True
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not count:
        raise ValueError('Scope has no trainable parameters')
    return count


def pair_batch(rows, image_index):
    """Return parallel image paths and labels, resolving the M3 image union."""
    drawings, renders, labels = [], [], []
    for row in rows:
        drawings.append(image_index[row['drawing_path']])
        renders.append(image_index[row['render_path']])
        label = row['label']
        if label not in (0, 1):
            raise ValueError('Pair labels must be binary')
        labels.append(label)
    return drawings, renders, labels
