#!/usr/bin/env python3
"""Convert docs/paper/paper.md into a Typst source (paper.typ) laid out
in a standard two-column conference-paper style.

Handles only the constructs used in paper.md: headings, paragraphs,
bullet/enum lists, pipe tables (with optional **表 N** caption blocks),
figures with italic caption blocks, **bold**, `code`, and the header
block (title / author / Abstract / Keywords).
"""
import re
import sys
from pathlib import Path

SRC = Path(__file__).with_name('paper.md')
OUT = Path(__file__).with_name('paper.typ')

SPAN_FIGURES = {'fig_protocol.png', 'fig_head_detail.png', 'fig_lora_detail.png'}


def esc_text(s: str) -> str:
    s = s.replace('#', '\\#').replace('@', '\\@').replace('$', '\\$')
    s = s.replace('<', '\\<').replace('>', '\\>')
    s = s.replace('_', '\\_')
    return s


def inline(s: str) -> str:
    parts = s.split('`')
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            out.append('`' + part + '`')
        else:
            part = part.replace('\\*', '\x00')   # literal star from md \*
            part = part.replace('**', '\x01')    # bold marker
            part = esc_text(part)
            part = part.replace('*', '\\*')      # any remaining lone star
            part = re.sub('\x01(.+?)\x01', r'*\1*', part)
            part = part.replace('\x00', '\\*')
            out.append(part)
    return ''.join(out)


def join_lines(lines):
    out = ''
    for l in lines:
        s = l.strip()
        if out and s:
            a, b = out[-1], s[0]
            # Insert a space only at an ASCII/ASCII boundary (English text
            # wrapped in the md source); never inside paths/identifiers.
            if (a.isascii() and b.isascii() and not a.isspace() and not b.isspace()
                    and a not in '/_' and b not in '/_.'):
                out += ' '
        out += s
    return out


def split_blocks(text):
    blocks, cur = [], []
    for line in text.split('\n'):
        if line.strip() == '':
            if cur:
                blocks.append(cur)
                cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append(cur)
    return blocks


def is_table(block):
    return block[0].lstrip().startswith('|')


def parse_table(block):
    rows = []
    for line in block:
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if all(re.fullmatch(r':?-{2,}:?', c) for c in cells):
            continue
        rows.append(cells)
    return rows


def emit_table(rows, caption):
    ncol = max(len(r) for r in rows)
    lines = ['#figure(', f'  table(columns: {ncol}, stroke: none,']
    lines.append('    table.hline(stroke: 0.8pt),')
    for ri, row in enumerate(rows):
        cells = row + [''] * (ncol - len(row))
        rendered = []
        for ci, c in enumerate(cells):
            body = inline(c)
            if ri == 0:
                body = f'*{body}*' if body else ''
            rendered.append(f'[{body}]' if ci == 0 else f'align(center)[{body}]')
        lines.append('    ' + ', '.join(rendered) + ',')
        if ri == 0:
            lines.append('    table.hline(stroke: 0.4pt),')
    lines.append('    table.hline(stroke: 0.8pt),')
    lines.append('  ),')
    if caption:
        lines.append(f'  caption: [{inline(caption)}],')
    lines.append(')')
    return '\n'.join(lines)


def emit_figure(block, caption_block):
    m = re.search(r'\(([^)]+)\)', block[0])
    path = m.group(1)
    caption = ''
    if caption_block:
        raw = join_lines(caption_block).strip()
        if raw.startswith('*') and raw.endswith('*'):
            raw = raw[1:-1]
        caption = raw
    name = Path(path).name
    span = name in SPAN_FIGURES
    body = f'#image("{path}", width: {"82" if span else "100"}%)'
    cap = f'\n  #text(size: 8.5pt)[{inline(caption)}]' if caption else ''
    if span:
        return ('#place(top, scope: "parent", float: true)[\n'
                f'  #align(center)[{body}]{cap}\n]')
    return f'#figure[\n  #align(center)[{body}]{cap}\n]'


PREAMBLE = r'''#set page(
  paper: "us-letter",
  margin: (x: 1.7cm, top: 1.9cm, bottom: 2.1cm),
  columns: 2,
  numbering: "1",
  number-align: center,
)
#set text(font: "Noto Serif CJK JP", size: 9.5pt, lang: "ja")
#set par(justify: true, leading: 0.85em, first-line-indent: 1em, spacing: 0.55em)
#set heading(numbering: none)
#show heading.where(level: 1): it => block(
  above: 1.15em, below: 0.5em,
  text(size: 11.5pt, weight: "bold")[#it.body],
)
#show heading.where(level: 2): it => block(
  above: 0.95em, below: 0.4em,
  text(size: 10pt, weight: "bold")[#it.body],
)
#set list(indent: 0.4em, body-indent: 0.7em, spacing: 0.45em)
#set enum(indent: 0.4em, body-indent: 0.7em, spacing: 0.45em)
#show figure.where(kind: table): set text(size: 8pt)
#show figure.caption: set text(size: 8.5pt)
'''


def main():
    blocks = split_blocks(SRC.read_text(encoding='utf8'))
    title = author = None
    abstract_blocks = []
    body = []
    i = 0
    mode = 'pre'
    while i < len(blocks):
        b = blocks[i]
        head = b[0].strip()
        if head.startswith('# '):
            title = head[2:].strip()
            i += 1
            continue
        if 'Riku Imanishi' in head:
            author = b
            i += 1
            continue
        if head == '## Abstract':
            mode = 'abstract'
            i += 1
            continue
        if mode == 'abstract' and head.startswith('## '):
            mode = 'body'
        if mode == 'abstract':
            abstract_blocks.append(b)
            i += 1
            continue
        body.append(b)
        i += 1

    out = [PREAMBLE]
    # ---- title block (spans both columns) ----
    note = ''
    if author:
        for line in author[1:]:
            if line.startswith('草稿'):
                note = line.strip()
    out.append('#place(top, scope: "parent", float: true)[\n'
               '  #set par(first-line-indent: 0em)\n'
               f'  #align(center)[#text(size: 15.5pt, weight: "bold")[{inline(title)}]]\n'
               '  #v(0.5em)\n'
               '  #align(center)[#text(size: 11pt)[Riku Imanishi（所属: TODO）]]\n'
               f'  #align(center)[#text(size: 8.5pt, fill: gray)[{inline(note)}]]\n'
               '  #v(0.7em)\n'
               '  #block(inset: (x: 1.1cm))[\n'
               '    #text(size: 9.5pt, weight: "bold")[Abstract]\n')
    for b in abstract_blocks:
        text = join_lines(b)
        out.append(f'    #par(first-line-indent: 0em)[#text(size: 9pt)[{inline(text)}]]\n')
    out.append('  ]\n]\n')

    # ---- body ----
    j = 0
    while j < len(body):
        b = body[j]
        head = b[0].strip()
        if head.startswith('### '):
            out.append(f'= {inline(head[4:])}'.replace('= ', '== ', 1))
            j += 1
            continue
        if head.startswith('## '):
            out.append(f'= {inline(head[3:])}')
            j += 1
            continue
        if head.startswith('!['):
            cap = body[j + 1] if j + 1 < len(body) else None
            out.append(emit_figure(b, cap))
            j += 2 if cap is not None else 1
            continue
        if head.startswith('**表'):
            caption = join_lines(b).strip().strip('*')
            rows = parse_table(body[j + 1])
            out.append(emit_table(rows, caption))
            j += 2
            continue
        if is_table(b):
            rows = parse_table(b)
            out.append(emit_table(rows, None))
            j += 1
            continue
        if head.startswith('- '):
            items, cur = [], None
            for line in b:
                if line.startswith('- '):
                    if cur is not None:
                        items.append(cur)
                    cur = line[2:]
                else:
                    cur += line.strip()
            items.append(cur)
            for it in items:
                out.append(f'- {inline(it)}')
            j += 1
            continue
        if re.match(r'^\d+\. ', head):
            items, cur = [], None
            for line in b:
                if re.match(r'^\d+\. ', line.strip()):
                    if cur is not None:
                        items.append(cur)
                    cur = re.sub(r'^\d+\. ', '', line.strip())
                else:
                    cur += line.strip()
            items.append(cur)
            for it in items:
                out.append(f'+ {inline(it)}')
            j += 1
            continue
        out.append(inline(join_lines(b)))
        j += 1

    OUT.write_text('\n\n'.join(out) + '\n', encoding='utf8')
    print(f'wrote {OUT} ({OUT.stat().st_size} bytes)')


if __name__ == '__main__':
    sys.exit(main())
