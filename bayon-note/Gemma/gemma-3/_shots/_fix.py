import re
p = r'E:\transformers\gemma-3\index.html'
html = open(p, encoding='utf-8').read()
m = re.search(r'(<svg viewBox="0 0 1280 660".*?</svg>)', html, re.S)
assert m, 'main svg not found'
svg = m.group(1)
fixed = svg.replace('<span class="dv"', '<tspan class="dv"').replace('</span>', '</tspan>')
html = html.replace(svg, fixed)
open(p, 'w', encoding='utf-8').write(html)
print('replaced spans in main svg:', fixed.count('<tspan'))
