import networkx as nx
import pandas as pd

reverse_props = {
    'author': 'created_by',
    'characters': 'present_in',
    'doctoral_advisor': 'doctoral_student',
    'doctoral_student': 'doctoral_advisor',
    'educated_at': 'educated_here',
    'employer': 'has_employed',
    'facet_of': 'has_facet',
    'field_of_this_occupation': 'is_field_of_this_occupation',
    'field_of_work': 'is_field_of_work',
    'followed_by': 'follows',
    'influenced_by': 'influenced',
    'interested_in': 'is_field_of_interest_of',
    'main_subject': 'is_main_subject_of',
    'member_of': 'has_member',
    'movement': 'is_movement_of',
    'named_after': 'gives_name',
    'notable_work': 'work_of',
    'occupation': 'is_occupation_of',
    'opposite_of': 'is_opposite_of',
    'part_of': 'has_part',
    'position_held': 'position_held_by',
    'present_in_work': 'work_includes',
    'pseudonym': 'is_psudonym_of',
    'significant_event': 'is_significant_event_of',
    'sponsor': 'is_sponsor_of',
    'student': 'student_of',
    'student_of': 'student',
    'studied_by': 'studies',
    'studies': 'studied_by',
    'subclass_of': 'superclass_of',
    'work_location': 'is_work_location_of'
}

triples = pd.read_csv("wikidata_triples_new.csv", names=["s", "p", "o"], delimiter="\t")
G = nx.MultiDiGraph()

for index, row in triples.iterrows():
    print(index)

    subj = row['s']
    rel = row['p']
    obj = row['o']

    G.add_node(subj, node_type='wd_ent')
    G.add_node(obj, node_type='wd_ent')
    G.add_edge(subj, obj, rel_type=rel)
    G.add_edge(obj, subj, rel_type=reverse_props[rel])

nx.write_gpickle(G,'wikidata_graph.pkl')
