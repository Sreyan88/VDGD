import pickle
import math
from tqdm import tqdm 

import matplotlib.pyplot as plt
import sys 

def norm_prob(token_list):
    max_logit = max(token["logit"] for token in token_list)
    sum_exp_logit = sum(math.exp(token["logit"] - max_logit) for token in token_list)

    for token in token_list:
        logit_diff = token["logit"] - max_logit
        token["norm_prob"] = math.exp(logit_diff) / sum_exp_logit

    return token_list

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    similarity = intersection / union
    return similarity

def compute_similarity(token_list_1, token_list_2, metric="jaccard", top_k=100):
    token_list_1 = norm_prob(token_list_1)
    token_list_2 = norm_prob(token_list_2) 
    list1 = [i["token"] for i in token_list_1[:top_k]]
    list2 = [i["token"] for i in token_list_2[:top_k]]
    set1 = set(list1)
    set2 = set(list2)
    similarity = 0.0
    if metric == "jaccard":
        similarity = jaccard_similarity(set1, set2)
    elif metric == "weighted_sum":
        all_tokens = set1.union(set2)
        for t in all_tokens:
            w1 = token_list_1[list1.index(t)]["norm_prob"] if t in list1 else 0
            w2 = token_list_2[list2.index(t)]["norm_prob"] if t in list2 else 0
            # if w1 > 0 and w2 > 0:
            similarity += w1 * w2
    elif metric == "KL":
        all_tokens = set1.union(set2)
        for t in all_tokens:
            w1 = token_list_1[list1.index(t)]["norm_prob"] if t in list1 else 0
            w2 = token_list_2[list2.index(t)]["norm_prob"] if t in list2 else 0
            if w1 > 0 and w2 > 0:
                similarity += w1 * math.log(w1 / w2)
    elif metric == "nDCG":
        relevance_scores = {}
        for i, token in enumerate(token_list_2):
            relevance_scores[token["token_id"]] = 1 / math.log2(
                i + 2
            )  # Assign relevance scores based on position

        dcg = 0.0
        for i, token in enumerate(token_list_1):
            relevance_score = relevance_scores.get(token["token_id"], 0.0)
            dcg += relevance_score / math.log2(i + 2)  # Compute DCG

        # Compute IDCG by sorting token_list_2 in descending order of relevance scores
        ideal_scores = sorted(relevance_scores.values(), reverse=True)
        idcg = sum(ideal_scores[i] / math.log2(i + 2) for i in range(len(token_list_1)))

        if idcg == 0:
            return 0.0  # To avoid division by zero, return 0 if IDCG is 0

        ndcg = dcg / idcg
        similarity = ndcg
    elif metric == "top_rank":
        top_token = list2[0]
        rank = top_k
        if top_token in list1:
            rank = list1.index(top_token)
        similarity = rank
    elif metric == "top_prob":
        top_token = list2[0] 
        prob = 0.0
        if top_token in list1:
            rank = list1.index(top_token)
            prob = token_list_1[rank]["norm_prob"]
        similarity = prob
    elif metric == "bi_top_prob":
        top_token = list2[0] 
        prob1 = 0.0
        if top_token in list1:
            rank = list1.index(top_token)
            prob1 = token_list_1[rank]["norm_prob"]
        # --- 
        top_token = list1[0] 
        prob2 = 0.0
        if top_token in list2:
            rank = list2.index(top_token)
            prob2 = token_list_2[rank]["norm_prob"]
        similarity = (prob1 + prob2)/2 
    return similarity

def get_top_k_avg(pkl_path):
    with open(pkl_path, "rb") as f:
        pkl_data = pickle.load(f)

    avg_token_len = []

    curves = {}
    for eid in tqdm(range(len(pkl_data))):
        temp_curves = []
        avg_token_len.append(len(pkl_data[eid]["formatted_inst"]))
        for x in range(min(len(pkl_data[eid]["formatted_base"]), len(pkl_data[eid]["formatted_inst"]))):
            list1 = pkl_data[eid]["formatted_base"][f"position_{x}"]["candidates"]
            list2 = pkl_data[eid]["formatted_inst"][f"position_{x}"]["candidates"]

            temp_curves.append(
                {
                    "step": x,
                    "jaccard": compute_similarity(
                        list1, list2, metric="jaccard", top_k=50
                    ),
                    "jaccard@10": compute_similarity(
                        list1, list2, metric="jaccard", top_k=10
                    ), 
                    "nDCG": compute_similarity(
                        list1, list2, metric="nDCG", top_k=50
                    ),
                    "nDCG@10": compute_similarity(
                        list1, list2, metric="nDCG", top_k=10
                    ),
                    "KL": compute_similarity(
                        list1, list2, metric="KL", top_k=50
                    ),
                    "KL@10": compute_similarity(
                        list1, list2, metric="KL", top_k=10
                    ), 
                    "WS": compute_similarity(
                        list1, list2, metric="weighted_sum", top_k=50
                    ),
                    "WS@10": compute_similarity(
                        list1, list2, metric="weighted_sum", top_k=10
                    ),
                    "TR": compute_similarity(
                        list1, list2, metric="top_rank", top_k=10
                    ),
                    "TP": compute_similarity(
                        list1, list2, metric="top_prob", top_k=10
                    ),
                }
            )
        curves[eid] = temp_curves

    token_pos_avg = {}

    for k,v in curves.items():
        for idx, dic in enumerate(v):
            if idx not in token_pos_avg:
                token_pos_avg[idx] = {}
            for k1, v1 in dic.items():
                if k1!='step':
                    if k1 not in token_pos_avg[idx]:
                        token_pos_avg[idx][k1] = []
                    token_pos_avg[idx][k1].append(v1)

    for k, v in token_pos_avg.items():
        for k1, v1 in v.items():
            token_pos_avg[k][k1] = sum(token_pos_avg[k][k1])/len(token_pos_avg[k][k1])

    return token_pos_avg, sum(avg_token_len)/len(avg_token_len)

if __name__ == "__main__":

    pkl_path1 = f"..//AlignTDS/src/demo/just_eval+cogvlm_benchmark_1500_ck_correctness_tp.pkl"
    pkl_path2 = f"..//AlignTDS/src/demo/just_eval+cogvlm_cogvlm_benchmark_1500_ck_desc_v2_correctness_tp.pkl"

    token_pos_avg1, avg_len1 = get_top_k_avg(pkl_path1)
    token_pos_avg2, avg_len2 = get_top_k_avg(pkl_path2)

    print(avg_len1, avg_len2)

    metrics = ["KL@10", "TP", "TR"]

    pkl_save = {}

    lim = 100
    x = [i for i in range(lim)]
    pkl_save["x"] = x

    for mt in metrics:
        y1 = []
        y2 = []
        for k, v in token_pos_avg1.items():
            y1.append(v[mt])
        
        for k, v in token_pos_avg2.items():
            y2.append(v[mt])

        # lim = 50#int(min(avg_len1, avg_len2)/2)#min(len(y1), len(y2))

        print(len(y1), len(y2))

        pkl_save[f"y_{mt}_greedy"] = y1[:lim]
        pkl_save[f"y_{mt}_vdgd"] = y2[:lim]

        plt.plot(x[:lim], y1[:lim], label=f'{mt}_greedy', color='blue', linestyle='-')
        plt.plot(x[:lim], y2[:lim], label=f'{mt}_vdgd', color='red', linestyle='-')
        plt.xlabel('Position')
        plt.ylabel('Score')
        plt.title('Sample Graph')
        plt.legend()
        plt.xlim(0, lim)

        plt.savefig(f'graphs/sample_plot_{mt}.png')
        plt.close('all')
    
    with open('cogvlm_vdgd_vs_greedy.pkl', 'wb') as f:
        pickle.dump(pkl_save, f)

    print(pkl_save.keys())