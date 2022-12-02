import random
import pickle
import torch

random.seed(0)

QUERY_STRUCTURES = ['1p','2p','3p','2i','3i','pi','ip']

query_name_dict = {('e',('r',)): '1p', 
                    ('e', ('r', 'r')): '2p',
                    ('e', ('r', 'r', 'r')): '3p',
                    (('e', ('r',)), ('e', ('r',))): '2i',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                    ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                    (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                    (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                    ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                    (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                    (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                    (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                    ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
                }
name_query_dict = {value: key for key, value in query_name_dict.items()}
all_tasks = list(name_query_dict.keys())
index_to_query_dict = {i:list(name_query_dict.keys())[i] for i in range(len(name_query_dict.keys()))}
query_to_index_dict = {value: key for key, value in index_to_query_dict.items()}

def tensorize_p(q):
    s = torch.tensor([q[0]])
    rs = torch.tensor(list(q[1]))
    return {'sources': s, 'relations': rs}

def tensorize_i(q):
    s = torch.tensor([t[0] for t in q])
    rs = torch.tensor([t[1][0] for t in q])
    return {'sources': s, 'relations': rs}
    
def tensorize_ip(q):
    s = torch.tensor([q[0][0][0], q[0][1][0]])
    rs = torch.tensor([q[0][0][1][0], q[0][1][1][0], q[1][0]])
    return {'sources': s, 'relations': rs}

def tensorize_pi(q):
    s = torch.tensor([q[0][0], q[1][0]])
    rs = torch.tensor(list(q[0][1]) + [q[1][1][0]])
    return {'sources': s, 'relations': rs}

def tensorize(q, query_structure):
    if query_structure in ['1p', '2p', '3p']:
        t = tensorize_p(q)
    elif query_structure in ['2i', '3i']:
        t = tensorize_i(q)
    elif query_structure == 'ip':
        t = tensorize_ip(q)
    elif query_structure == 'pi':
        t = tensorize_pi(q)
    else:
        raise ValueError(f'query structure {query_structure} not implemented')
    t['structure'] = query_structure
    return t

def map_triples(queries, answers, 
                query_structures=QUERY_STRUCTURES, random_sample=False, filter_fun=None, remap_fun=None, max_samples=1):
    mapped_triples = {}
    for query_structure in query_structures:
        print(f'loading query structure {query_structure}')
        qs = queries[name_query_dict[query_structure]]

        num_filtered = 0
        qlist = []
        for q in qs:
            qtens = tensorize(q, query_structure)
            ans = list(answers[q])
            if remap_fun is not None:
                qtens, ans = remap_fun(qtens, ans)
            if filter_fun is not None:
                if not filter_fun(qtens, ans):
                    num_filtered += 1
                    continue
            if random_sample and len(ans) > 0:
                aa = random.sample(ans, max_samples) if len(ans) > max_samples else random.sample(ans, len(ans))
            else:
                aa = ans
            for a in aa:
                qtens['target'] = torch.LongTensor([a])
                qtens['others'] = torch.LongTensor([o for o in ans if o != a])
                qlist.append(qtens)
        print(f'filtered {num_filtered} queries of {len(qs)} possible for query {query_structure}')
        mapped_triples[query_structure] = qlist
    return mapped_triples
    
def generate_mapped_triples(query_loc, answer_loc, query_structures=QUERY_STRUCTURES,
                            random_sample=False, filter_fun=None, remap_fun=None, max_samples=1):
    with open(query_loc, 'rb') as f:
        queries = pickle.load(f)
    with open(answer_loc, 'rb') as f:
        answers = pickle.load(f)

    return map_triples(queries, answers, query_structures=query_structures, random_sample=random_sample,    
                        filter_fun=filter_fun, remap_fun=remap_fun, max_samples=max_samples)

def generate_mapped_triples_both(query_loc_easy, query_loc_hard, answer_loc_easy, answer_loc_hard, 
                                query_structures=QUERY_STRUCTURES, random_sample=False, filter_fun=None, remap_fun=None, 
                                max_samples=1):
    with open(query_loc_easy, 'rb') as f:
        easy_queries = pickle.load(f)
    with open(answer_loc_easy, 'rb') as f:
        easy_answers = pickle.load(f)
    with open(query_loc_hard, 'rb') as f:
        hard_queries = pickle.load(f)
    with open(answer_loc_hard, 'rb') as f:
        hard_answers = pickle.load(f)

    mapped_triples = {}
    for query_structure in query_structures:
        print(f'loading query structure {query_structure}')
        qs = hard_queries[name_query_dict[query_structure]] # same for easy and hard queries

        num_filtered = 0
        qlist = []
        for q in qs:
            qtens = tensorize(q, query_structure)
            easy_ans = list(easy_answers[q])
            num_easy = len(easy_ans)
            hard_ans = list(hard_answers[q])
            num_hard = len(hard_ans)
            ans = hard_ans + easy_ans # combine easy and hard answers
            if remap_fun is not None:
                qtens, ans = remap_fun(qtens, ans)
            if filter_fun is not None:
                if not filter_fun(qtens, ans):
                    num_filtered += 1
                    continue
            # only evaluate on hard answers (:num_hard)
            if random_sample and num_hard > 0:
                aa = random.sample(ans[:num_hard], max_samples) if num_hard > max_samples else random.sample(ans[:num_hard], num_hard)
            else:
                aa = ans
            for a in aa[:num_hard]:
                qtens['target'] = torch.LongTensor([a])
                qtens['others'] = torch.LongTensor([o for o in ans if o != a])
                qlist.append(qtens)
        print(f'filtered {num_filtered} queries of {len(qs)} possible for query {query_structure}')
        mapped_triples[query_structure] = qlist
    return mapped_triples
