def preprocess_wikibio(example, hits_per_query=10, snippets_per_hit=2):
    hit_snippets = []
    hits_added = 0
    to_drop = False
    for one_hit in example['hits']:
        if any([restricted_domain in one_hit['url'] for restricted_domain in ("wikipedia", "wikidata", "wikiwand")]):
            continue
        if len(one_hit['snippets']) == 0:
            hit_snippets.append(f"Title: {one_hit['title']} - Snippet: {one_hit['description']}")
            snippets_added = 1
        else:
            snippets_added = 0
            for one_snippet in one_hit['snippets']:
                if len(one_snippet) > 0:
                    hit_snippets.append(f"Title: {one_hit['title']} - Snippet: {one_snippet}")
                    snippets_added += 1
                if snippets_added >= snippets_per_hit:
                    break
        if snippets_added > 0:
            hits_added += 1
        if hits_added >= hits_per_query:
            break
    if len(hit_snippets) == 0:
        to_drop = True
    
    user_message = "\n".join(hit_snippets) + '\n\n' + example['query']
    out_data = example.copy()
    out_data['to_drop'] = to_drop
    out_data['user_message'] = user_message
    return out_data
