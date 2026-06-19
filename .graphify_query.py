import json
from networkx.readwrite import json_graph
import networkx as nx
from pathlib import Path

data = json.loads(Path('graphify-out/graph.json').read_text(encoding='utf-8'))
G = json_graph.node_link_graph(data, edges='links')
UG = G.to_undirected()

# find grounding-related nodes
print('=== grounding-related nodes ===')
for nid, d in G.nodes(data=True):
    lab = d.get('label','')
    if any(k in lab.lower() for k in ['ground', 'guardrail', 'citation', 'quote_is', 'neutral']):
        print('  [c%s] %s | %s' % (d.get('community','?'), lab[:70], d.get('source_file','')))

# hyperedges captured as nodes? check edges for grounding relation
def node_for_sub(sub):
    for nid, d in G.nodes(data=True):
        if d.get('label','').strip().lower() == sub.lower():
            return nid
    return None

for name in ['NewsAgent', 'SECFilingAgent', 'quote_is_grounded()']:
    nid = node_for_sub(name)
    print('\n== %s | comm %s ==' % (name, G.nodes[nid].get('community') if nid else '?'))
    if not nid: continue
    for nb in UG.neighbors(nid):
        d = UG.edges[nid, nb]
        lab = G.nodes[nb].get('label', nb)
        if any(k in lab.lower() for k in ['ground','guard','neutral','confidence','driver','finding','citation','classif','headline','quote']) or d.get('relation') in ('rationale_for','semantically_similar_to','implements'):
            print('   --%s [%s %.2f]-- %s | c%s' % (d.get('relation',''), d.get('confidence',''), d.get('confidence_score',0), lab[:60], G.nodes[nb].get('community','?')))
