import torch

def prepare_query_for_prediction(hrt_batch, query_structure, device):
    if query_structure in ['1p', '2p', '3p']:
        sources = []
        relations = []
        for b in hrt_batch:
            sources.append(b['sources'])
            relations.append(b['relations'].unsqueeze(0))
            # targets.append(b['target'])
        sources = torch.cat(sources, dim=0)
        relations = torch.cat(relations, dim=0)
        # targets = torch.cat(targets, dim=0)
        batch = torch.cat([sources.unsqueeze(-1), relations], dim=1).to(device)
        return batch

    if query_structure in ['2i', '3i']:
        sources = []
        relations = []
        for b in hrt_batch:
            sources.append(b['sources'].unsqueeze(0))
            relations.append(b['relations'].unsqueeze(0))
            # targets.append(b['target'])
        sources = torch.cat(sources, dim=0)
        relations = torch.cat(relations, dim=0)
        # targets = torch.cat(targets, dim=0)
        # get batch into size (nbatch, 2, num_entities/num_relations)
        # where the second dimension is entities (0 index) then relations (1 index)
        batch = torch.cat([sources.unsqueeze(1), relations.unsqueeze(1)], dim=1).to(device)
        return batch

    if query_structure in ['pi', 'ip']:
        # just send as-is
        return hrt_batch