import os
import math
import json
import logging
import argparse
import numpy as np
import threading
import concurrent.futures

from tqdm.auto import tqdm
from src.utils import *
from src.llm_backend import LLMBackend
from src.peit_client import PeitClient
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.util import ngrams
from rouge import Rouge

def main():
    from src.prompts import bigfive_score_criteria, system_prompt, judge_prompt
    
    parser = argparse.ArgumentParser(description="Social LLMs Benchmark - Generation")
    parser.add_argument('--platform', type=str, default="", help='platform.')
    parser.add_argument('--base-url', type=str, default="", help='base url.')
    parser.add_argument('--api-key', type=str, default="", help='api key.')
    parser.add_argument('--model', type=str, default="", help='model.')
    parser.add_argument('--max-tokens', type=int, default=1024, help='max_new_tokens.')
    parser.add_argument('--temperature', type=float, default=0.6, help='temperature.')
    parser.add_argument('--top-p', type=float, default=0.7, help='top_p.')
    
    parser.add_argument('--platform-critic', type=str, default="", help='critic platform.')
    parser.add_argument('--base-url-critic', type=str, default="", help='critic base url.')
    parser.add_argument('--api-key-critic', type=str, default="", help='critic api key.')
    parser.add_argument('--model-critic', type=str, default="", help='critic model.')
    parser.add_argument('--max-tokens-critic', type=int, default=1024, help='critic max_new_tokens.')
    parser.add_argument('--temperature-critic', type=float, default=0.01, help='critic temperature.')
    parser.add_argument('--top-p-critic', type=float, default=None, help='critic top_p.')
    
    parser.add_argument('--convs-per-chunk', type=int, default=10, help='conversations per chunk.')
    parser.add_argument('--qps', type=int, default=30, help='qps.')
    parser.add_argument('--qps-critic', type=int, default=30, help='critic qps.')
    parser.add_argument('--max-retry-times', type=int, default=5, help='max retry times.')
    parser.add_argument('--ablation', type=str, default="", help='platform.')
    parser.add_argument('--mode', type=str, default="pcit", help='fuse mode.')
    args = parser.parse_args()
    
    logs_path = "logs"
    data_path = "data"
    config_path = "config"
    output_path = "output"
    
    if args.ablation == "profile":
        data_path = os.path.join(data_path, "ptit_profile_ablation.json")
    elif args.ablation == "personality":
        data_path = os.path.join(data_path, "ptit_personality_ablation.json")
    elif args.ablation == "potential":
        data_path = os.path.join(data_path, "ptit_potential_ablation.json")
    else:
        if args.ablation == "without_psychology":
            data_path = os.path.join(data_path, "ptit_without_psychology.json")
        elif args.ablation == "without_psychology_media":
            data_path = os.path.join(data_path, "ptit_without_psychology_media.json")
        else:
            if args.mode == "psit":
                data_path = os.path.join(data_path, "psit_propose.json")
            else:
                data_path = os.path.join(data_path, "ptit_propose.json")

    if args.ablation:
        ab_str = "-ablation-" + args.ablation
    else:
        ab_str = ""
    model_str = (args.model.replace("/", "-").replace("\\", "-") + ab_str).strip()
    result_path = os.path.join("output", model_str)
    logs_failed_path = os.path.join(logs_path, model_str)
    logs_file_path = os.path.join(logs_path, f"{model_str}.log")
    generation_path = os.path.join(result_path, "generation")
    score_path = os.path.join(result_path, "score")
    judge_path = os.path.join(result_path, "judge")
    
    check_dirs(logs_path)
    check_dirs(logs_failed_path)
    check_dirs(output_path)
    check_dirs(generation_path)
    check_dirs(score_path)
    check_dirs(judge_path)

    logger = logging.getLogger('social_bench_logger')
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(logs_file_path)
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    rouge = Rouge()

    # if args.mode == "peit":
    #     actor = PeitClient(
    #         base_url=args.base_url, 
    #         max_tokens=args.max_tokens,
    #         temperature=args.temperature,
    #         top_p=args.top_p
    #     )
    # else:
    actor = LLMBackend(
        platform=args.platform,
        base_url=args.base_url, 
        api_key=args.api_key,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
    logger.debug(actor.test())
    
    critic = LLMBackend(
        platform=args.platform_critic,
        base_url=args.base_url_critic, 
        api_key=args.api_key_critic,
        model=args.model_critic,
        max_tokens=args.max_tokens_critic,
        temperature=args.temperature_critic,
        top_p=args.top_p_critic
    )
    
    logger.debug(critic.test())
    
    def generator(llm, prompt_id, prompt, output_path, phar=None, extract_func=None, extra_info={}, peit_mode=False):
        try_time = 0
        while try_time < args.max_retry_times:
            try:
                if peit_mode:
                    score = extra_info["score"]
                    re = llm.request(prompt, score)
                else:
                    re = llm.request(prompt)
                file_path = os.path.join(output_path, f"{prompt_id}.json")
                if extract_func:
                    sj = extract_func(re)
                else:
                    sj = re
                if extra_info:
                    sj["extra_info"] = extra_info
                # for k, v in extra_info.items():
                #     sj["extra_info"][k] = v
                save_dic2json(file_path, sj)
                break
            except Exception as e:
                logger.error(f"{prompt_id}:\t{e}")
                file_path = os.path.join(logs_failed_path, f"error-{prompt_id}-{try_time}.txt")
                # 打开文件以写入文本
                with open(file_path, 'w') as file:
                   # 写入文本
                   file.write(f"{prompt_id}\n{e}\n{re}\n\n")
                try_time += 1
        if phar:
            phar.update(1)
    
    with open(data_path, 'r', encoding="utf8") as f:
        dataset = json.load(f)
    
    user_profiles = {}
    user_traits = {}
    user_ground_truth_scores = {}
    selected_users = []
    num_all_posts = 0
    for userName, user_data in dataset.items():
        user_profiles[userName] = user_data["meta_data"]["profile"]
        user_traits[userName] = user_data["meta_data"]["personality"]
        user_ground_truth_scores[userName] = user_data["meta_data"]["score"]
        num_all_posts += len(user_data["data"])
        selected_users.append(userName)
    save_dic2json(os.path.join(result_path, "selected_users.json"), selected_users)
    
    already_generated = []
    for filename in tqdm(os.listdir(generation_path)):
        if filename.endswith('.json'):
            file_path = os.path.join(generation_path, filename)
            base_name, extension = os.path.splitext(filename)
            already_generated.append(base_name)
    
    logger.debug(f"already_generated numbers:\t{len(already_generated)}")
    
    toal_request_times = num_all_posts
    phar_generation = tqdm(total=toal_request_times)
    
    logger.info("Generating...")
    
    if args.ablation == "without_psychology_media":
        extract_func = load_json_or_pure
    else:
        extract_func = load_json
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.qps) as executor:
        for userName, user_data in dataset.items():
            meta_data = user_data["meta_data"]
            userName = meta_data["userName"]
            data = user_data["data"]
            score = user_data["meta_data"]["score"]
            for i, item in enumerate(data):
                prompt_id = f"{userName}-{i}"
                if prompt_id in already_generated:
                    continue
                extra_info = {
                    "id": prompt_id,
                    "userName": userName,
                    "output": item["output"],
                    "score": score,
                    "up_stream": item["up_stream"],
                    "knowledge": item["knowledge"],
                    "profile_related": meta_data["profile_related"],
                    "personality_related": meta_data["personality_related"]
                }
                future = executor.submit(generator, actor, prompt_id, item["instruction"], generation_path, phar_generation, extract_func=extract_func, extra_info=extra_info, peit_mode=False)

    logger.info("Auto scoring...")
    
    actor_generated = []
    post_length_arr = []
    for filename in tqdm(os.listdir(generation_path)):
        if filename.endswith('.json'):
            file_path = os.path.join(generation_path, filename)
            with open(file_path, 'r', encoding="utf8") as f:
                response = json.load(f)
            base_name, extension = os.path.splitext(filename)
            response["id"] = base_name
            if "Post Content" not in response:
                continue
            post_length_arr.append(len(response["Post Content"]))
            actor_generated.append(response)
            
    average_post_length = np.mean(np.array(post_length_arr))
    
    contrastives = []
    references = []
    candidates = []
    for i in tqdm(actor_generated):
        if "Post Content" not in i:
            print(i)
            continue
        reference = i['extra_info']["output"]["Post Content"]
        candidate = i["Post Content"]
        references.append([reference])
        candidates.append(candidate)
        contrastives.append([
            i['id'],
            reference,
            candidate
        ])
    
    contrastives_path = os.path.join(result_path, "contrastives.json")
    save_dic2json(contrastives_path, contrastives)
    logger.debug(contrastives[0])

    bleu_score = corpus_bleu(references, candidates)
    
    # all_bleu = []
    all_distinct = []
    all_rouge_1 = []
    all_rouge_2 = []
    all_rouge_l = []
    for c_id, reference, generated in tqdm(contrastives):
        # bleu_score = sentence_bleu([reference.split()], generated.split())
        # all_bleu.append(bleu_score)
        distinct_score = calculate_distinct(generated)
        all_distinct.append(distinct_score)
        rouge_score = rouge.get_scores(reference, generated)
        all_rouge_1.append(rouge_score[0]["rouge-1"]["f"])
        all_rouge_2.append(rouge_score[0]["rouge-2"]["f"])
        all_rouge_l.append(rouge_score[0]["rouge-l"]["f"])

    avg_bleu = bleu_score
    avg_distinct = np.mean(np.array(all_distinct))
    avg_rouge_1 = np.mean(np.array(all_rouge_1))
    avg_rouge_2 = np.mean(np.array(all_rouge_2))
    avg_rouge_l = np.mean(np.array(all_rouge_l))
    
    overlap_result = {
        "avg_bleu": avg_bleu,
        "avg_distinct": avg_distinct,
        "avg_rouge_1": avg_rouge_1,
        "avg_rouge_2": avg_rouge_2,
        "avg_rouge_l": avg_rouge_l,
        "avg_post_length": average_post_length
    }
    logger.debug(overlap_result)
    
    user_conversations = {}
    for i in tqdm(actor_generated):
        userName = i['extra_info']["userName"]
        if userName not in user_conversations:
            user_conversations[userName] = []
        generated_post = i["Post Content"]
        if not generated_post:
            continue
        if "Media" not in i:
            media = ""
        else:
            media = i["Media"]
        user_post = {
            "User": userName,
            "Post": generated_post,
            "Quote": {},
            "Media": media
        }
        upstream = i['extra_info']["up_stream"]
        if upstream is not None:
            conversation = [
                upstream,
                user_post
            ]
        else:
            conversation = [
                user_post
            ]
        item = {
            "meta_data": {
                'user': userName
            },
            "conversation": conversation
        }
        user_conversations[userName].append(item)
    
    chunks = []
    for user, convs in tqdm(user_conversations.items()):
        chunk_id = 0
        convs_len = len(convs)
        chunks_num = math.ceil(convs_len / args.convs_per_chunk)
        while chunk_id < chunks_num:
            start_index = chunk_id * args.convs_per_chunk
            if chunk_id == chunks_num - 1:
                end_index = None
            else:
                end_index = (chunk_id + 1) * args.convs_per_chunk
            chunk_conv = []
            for item in convs[start_index: end_index]:
                chunk_conv.append(item["conversation"])  
            chunk_id_str = f"{user}-{chunk_id}"
            dic = {
                "id": chunk_id_str,
                "content": chunk_conv
            }
            chunks.append(dic)
            chunk_id += 1

    already_scored = []
    for filename in tqdm(os.listdir(score_path)):
        if filename.endswith('.json'):
            file_path = os.path.join(score_path, filename)
            with open(file_path, 'r', encoding="utf8") as f:
                score_info = json.load(f)
            base_name, extension = os.path.splitext(filename)
            already_scored.append(base_name)
    
    print(f"already_scored numbers:\t{len(already_scored)}")
    
    queries = []
    for i, chunk in tqdm(enumerate(chunks)):
        conv_id = chunk["id"]
        if conv_id in already_scored:
            continue
        conv = chunk["content"]
        user_name = conv_id.split("-")[0]
        query = system_prompt.format(criteria=bigfive_score_criteria, name=user_name, conversation=conv)
        query_dic = {
            "id": conv_id,
            "content": query
        }
        queries.append(query_dic)
        
    if len(queries) > 0:
        logger.debug(queries[0])
    
        re = critic.request(queries[0]["content"])
        
        logger.debug(re)

    toal_score_queries = len(queries)
    phar_score = tqdm(total=toal_score_queries)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.qps_critic) as executor:
        for query in queries:
            if query["id"] in already_scored:
                continue
            future = executor.submit(generator, critic, query["id"], query["content"], score_path, phar_score, extract_func=load_json)
    
    from pydantic import BaseModel, Field
    
    class DeduceInfo(BaseModel):
        profile: bool = Field(description="whether my Post shows the content of my profile.")
        knowledge: bool = Field(description="whether my Post shows the content of potential knowledge.")
        personality: bool = Field(description="whether my Post provides explicit evidence of my personality traits.")
        
    from langchain.output_parsers import PydanticOutputParser
    
    output_parser = PydanticOutputParser(pydantic_object=DeduceInfo)
    format_instructions = output_parser.get_format_instructions()
    logger.debug(format_instructions)
    
    def judge_extractor(output):
        parsed_output = output_parser.parse(output)
        parsed_output_dict = parsed_output.dict()
        return parsed_output_dict
    
    already_judged = []
    judged = []
    for filename in tqdm(os.listdir(judge_path)):
        if filename.endswith('.json'):
            file_path = os.path.join(judge_path, filename)
            with open(file_path, 'r', encoding="utf8") as f:
                judge_sample = json.load(f)
            judged.append(judge_sample)
            base_name, extension = os.path.splitext(filename)
            already_judged.append(base_name)
            
    print(f"already_judged numbers:\t{len(already_judged)}")
    
    judge_queries = []
    missing_count = 0
    for item in tqdm(actor_generated):
        extra_info = item['extra_info']
        userName = extra_info["userName"]
        prompt_id = extra_info["id"]
        meida_content = ""
        try:
            post_content = item["Post Content"]
            if args.ablation != "without_psychology_media":
                meida_content = item["Media"]
        except Exception as e:
            logger.error(e)
            logger.error(item)
            missing_count += 1
            if missing_count >= 10:
                raise("Too many missing content samples.")
        if prompt_id in already_judged:
            continue
        upstream = extra_info["up_stream"]
        if not upstream:
            conversation = [
                {
                    "User": userName,
                    "Post": post_content,
                    "Media": [
                        {
                            "type": "image",
                            "content": meida_content
                        }
                    ]
                }
            ]
        else:
            conversation = [
                upstream,
                {
                    "User": userName,
                    "Post": post_content,
                    "Media": [
                        {
                            "type": "image",
                            "content": meida_content
                        }
                    ]
                }
            ]
        prompt = judge_prompt.format(
            user=userName,
            profile=user_profiles[userName], 
            traits=user_traits[userName],
            conversation=conversation,
            pk=extra_info["knowledge"],
            format_instructions=format_instructions
        )
        judge_queries.append(
            {
                "id": prompt_id,
                "content": prompt,
                "extra_info": extra_info
            }
        )
    if len(judge_queries) > 0:
        logger.debug(judge_queries[0])
    
    toal_request_times = len(judge_queries)
    phar_judge = tqdm(total=toal_request_times)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.qps_critic) as executor:
        for i in judge_queries:
            future = executor.submit(
                generator, 
                critic, 
                i["id"], 
                i["content"], 
                judge_path, 
                phar_judge, 
                extract_func=judge_extractor,
                extra_info=i["extra_info"]
            )
    
    judged = []
    for filename in tqdm(os.listdir(judge_path)):
        if filename.endswith('.json'):
            file_path = os.path.join(judge_path, filename)
            with open(file_path, 'r', encoding="utf8") as f:
                judge_sample = json.load(f)
            judged.append(judge_sample)
    
    all_profile_related = 0
    all_personality_related = 0
    all_potential_related = 0

    profile_related_count = 0
    personality_related_count = 0

    for item in tqdm(judged):
        extra_info = item['extra_info']
        if extra_info["profile_related"]:
            profile_related_count += 1
        if extra_info["personality_related"]:
            personality_related_count += 1
        
        if item["profile"] and extra_info["profile_related"]:
            all_profile_related += 1
        if item["knowledge"]:
            all_potential_related += 1
        if item["personality"] and extra_info["personality_related"]:
            all_personality_related += 1
            
    total_query = len(judged)
    avg_profile_related = all_profile_related / profile_related_count
    avg_personality_related = all_personality_related / personality_related_count
    avg_potential_related = all_potential_related / total_query
    
    related_result = {
        "profile_related": avg_profile_related,
        "personality_related": avg_personality_related,
        "potential_related": avg_potential_related
    }
    
    logger.debug(related_result)
    
    def read_user_scores(user_scores_path):
        directory_path = user_scores_path
        files = os.listdir(directory_path)
        files = [f for f in files if os.path.isfile(os.path.join(directory_path, f))]  
        sorted_files = sorted(files)  
        user_scores = {}
        for filename in tqdm(sorted_files):
            if filename.endswith('.json'):
                file_path = os.path.join(user_scores_path, filename)
                with open(file_path, 'r', encoding="utf8") as f:
                    score = json.load(f)
                base_name, extension = os.path.splitext(filename)
                user_scores[base_name] = score
        return user_scores
    
    user_scores = read_user_scores(score_path)
    
    def add_arrays(arr1, arr2):
        return np.add(arr1, arr2).tolist()
    
    user_average_scores = {}
    user_sum_explanations = {}
    
    current_sample = list(user_scores.keys())[0]
    current_user = "-".join(current_sample.split("-")[:-1])
    avg_scores = [0] * 35
    explanations = []
    chunk_num = 0
    i = 0
    total_chunk_num =  len(user_scores.keys())
    for user, score in tqdm(user_scores.items()):
        user_name = "-".join(user.split("-")[:-1])
        if user_name != current_user:
            avg_scores = [x / chunk_num for x in avg_scores]
            user_average_scores[current_user] = avg_scores
            user_sum_explanations[current_user] = explanations
            avg_scores = [0] * 35
            current_user = user_name
            chunk_num = 0
            avg_score = [0] * 35
            explanations = []
            
        guard = 0
        avg_score = [0] * 35
        try:
            for bigfive, sub_dimention in score.items():
                if isinstance(sub_dimention, str):
                    explanations.append(sub_dimention)
                    guard += 1
                    continue
                sub_scores = list(sub_dimention.values())
                if len(sub_scores) != 6:
                    logging.error(f"miss some sub-dimention scores:\n{sub_scores}")
                    guard += 1
                    continue
                bigfive_score = sum(sub_scores)
                avg_score[guard*7] = bigfive_score
                avg_score[guard*7+1:guard*7+7] = sub_scores
                guard += 1
        except Exception as e:
            file_path = os.path.join(logs_failed_path, f"avg_score_error_{chunk_num}.txt")
            with open(file_path, 'w') as file:
               file.write(f"{chunk_num}\n{e}\n{score}\n\n")
            continue
        avg_scores = add_arrays(avg_scores, avg_score)
        chunk_num += 1

        i += 1
        if i == total_chunk_num:
            avg_scores = [x / chunk_num for x in avg_scores]
            user_average_scores[current_user] = avg_scores
            user_sum_explanations[current_user] = explanations
    
    logger.debug(f"total_chunk_num:\t{total_chunk_num}")
    
    user_avg_scores_path = os.path.join(result_path, "user_avg_scores_social_bench.json")
    
    save_dic2json(user_avg_scores_path, user_average_scores)
    
    user_explanations_path = os.path.join(result_path, "user_explanations_social_bench.json")
    
    save_dic2json(user_explanations_path, user_sum_explanations)
    
    with open(user_avg_scores_path, 'r', encoding="utf8") as f:
        user_average_scores = json.load(f)

    similarity_result = {
        "cosine_similarity": None,
        "euclidean_distance": None,
        "manhattan_distance": None
    }
    similarity_function_map = {
        "cosine_similarity": cosine_similarity,
        "euclidean_distance": euclidean_distance,
        "manhattan_distance": manhattan_distance
    }
    for function_name in similarity_function_map.keys():
        all_personality_similarities = []
        # all_O = []
        # all_C = []
        # all_E = []
        # all_A = []
        # all_N = []
        similarity_func = similarity_function_map[function_name]
        
        for user, score in tqdm(user_average_scores.items()):
            ground_truth_score = user_ground_truth_scores[user]
            ground_truth_score = np.array(ground_truth_score)
            ground_truth_score[ground_truth_score == 0] = 0.1
            
            score = np.array(score)
            score[score == 0] = 0.01

            indices = np.array([0, 7, 14, 21, 28])
            
            ocean = score[indices]
            ground_truth_ocean = ground_truth_score[indices]
            
            similarity_score = similarity_func(ocean, ground_truth_ocean)
            all_personality_similarities.append(similarity_score)
            
            # cosine_similarit_score = similarity_func(score, ground_truth_score)
            # all_personality_similarities.append(cosine_similarit_score)
            
            # cosine_similarit_score_O = similarity_func(score[:7], ground_truth_score[:7])
            # all_O.append(cosine_similarit_score_O)
    
            # cosine_similarit_score_C = similarity_func(score[7:14], ground_truth_score[7:14])
            # all_C.append(cosine_similarit_score_C)
    
            # cosine_similarit_score_E = similarity_func(score[14:21], ground_truth_score[14:21])
            # all_E.append(cosine_similarit_score_E)
    
            # cosine_similarit_score_A = similarity_func(score[21:28], ground_truth_score[21:28])
            # all_A.append(cosine_similarit_score_A)
    
            # cosine_similarit_score_N = similarity_func(score[28:35], ground_truth_score[28:35])
            # all_N.append(cosine_similarit_score_N)
    
        avg_All = np.mean(np.array(all_personality_similarities))
        # avg_O = np.mean(np.array(all_O))
        # avg_C = np.mean(np.array(all_C))
        # avg_E = np.mean(np.array(all_E))
        # avg_A = np.mean(np.array(all_A))
        # avg_N = np.mean(np.array(all_N))
    
        # avg_scores = {
        #     "avg_All": avg_All,
        #     "avg_O": avg_O,
        #     "avg_C": avg_C,
        #     "avg_E": avg_E,
        #     "avg_A": avg_A,
        #     "avg_N": avg_N
        # }
        avg_scores = {
            "avg_All": avg_All
        }
        similarity_result[function_name] = avg_scores
    
    logger.debug(similarity_result)
    
    final_result = {
        "overlap": overlap_result,
        "related": related_result,
        "similarity": similarity_result
    
    }
    
    final_result_path = os.path.join(result_path, "social_evaluation_result.json")
    
    save_dic2json(final_result_path, final_result)
    
    logger.info(f"Final evaluation result already saved at:\t{final_result_path}")


if __name__ == "__main__":
    main()

