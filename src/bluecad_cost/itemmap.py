"""Map official image references without changing item or candidate order."""
import json
from pathlib import Path, PurePosixPath

# Official vfms/retrieval_common.py VIEW_NAMES (not inferred from filenames).
VIEW_NAMES = ('front', 'back', 'left', 'right', 'top', 'bottom')
PREFIX = 'data/bluecad-30k'


def map_path(path):
    parts = PurePosixPath(path).parts
    if 'renders' in parts:
        tail = parts[parts.index('renders') + 1:]
        if len(tail) != 3:
            raise ValueError(f'Invalid render path: {path}')
        return f'{PREFIX}/renders/{tail[1]}/{tail[2]}'
    if 'png_out' in parts:
        tail = parts[parts.index('png_out') + 1:]
        if len(tail) != 4 or tail[2] not in ('normal', 'dim'):
            raise ValueError(f'Invalid drawing path: {path}')
        _, model, variant, filename = tail
        number = int(PurePosixPath(filename).stem)
        if not 0 <= number < len(VIEW_NAMES):
            raise ValueError(f'Invalid view: {path}')
        suffix = '_dim' if variant == 'dim' else ''
        return f'{PREFIX}/drawings/{model}/{variant}/{model}_{VIEW_NAMES[number]}{suffix}.png'
    raise ValueError(f'Unknown image path: {path}')


def load_items(path, task):
    items = json.loads(Path(path).read_text(encoding='utf-8'))['items']
    mapped = []
    for item in items:
        images = [map_path(p) for p in item['images']]
        drawings = [p for p in images if '/drawings/' in p]
        renders = [p for p in images if '/renders/' in p]
        if len(drawings) != 1 or len(renders) != (1 if str(task) == '1' else 4):
            raise ValueError(f"Invalid images for {item['id']}")
        record = {**item, 'drawing_path': drawings[0]}
        record['render_path' if str(task) == '1' else 'render_paths'] = renders[0] if str(task) == '1' else renders
        mapped.append(record)
    return mapped
