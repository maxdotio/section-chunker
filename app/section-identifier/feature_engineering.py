import pandas as pd
import numpy as np
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# def get_unique_ids(records):
#     unq_ids = set([row[0] for row in records])
#     return unq_ids

def make_features(records, unq_ids, patterns, discards, columns):
    datasets = {}
    pages_data = {id:[] for id in unq_ids}
    document_data = pd.read_csv('app\\section-identifier\\data\\document_data.csv')
    for id in unq_ids:
        metadata = document_data.loc[document_data['id'] == id, 'metadata'].item()
        # cursor.execute(f"SELECT metadata FROM documents WHERE id={id}")
        # metadata_tuple = cursor.fetchall()
        # metadata = metadata_tuple[0][0]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        temp_dataset = {i:[] for i in columns}
        for i, page in enumerate(metadata["contract"]["pages"],start=1):
            page_data = {}
            xy1 = [(min(line["polygon"][0]["x"],line["polygon"][3]["x"]), min(line["polygon"][0]["y"],line["polygon"][1]["y"])) for line in page["lines"]]
            xy2 = [(max(line["polygon"][1]["x"],line["polygon"][2]["x"]), max(line["polygon"][2]["y"],line["polygon"][3]["y"])) for line in page["lines"]]
            temp_dataset["length"].extend([line["spans"][0]["length"] for line in page["lines"]])
            # temp_dataset["offset"].extend([line["spans"][0]["offset"] for line in page["lines"]])
            temp_dataset["content"].extend([line["content"] for line in page["lines"]])
            # temp_dataset["x1"].extend([xy1[i][0] for i in range(len(xy1))])
            # temp_dataset["x2"].extend([xy2[i][0] for i in range(len(xy2))])
            # temp_dataset["y1"].extend([xy1[i][1] for i in range(len(xy1))])
            # temp_dataset["y2"].extend([xy2[i][1] for i in range(len(xy2))])
            temp_label = [0 for k in range(len(page["lines"]))]
            temp_pattern_type = [0 for k in range(len(page["lines"]))]
            temp_dataset["line_height"].extend([(xy2[k][1]-xy1[k][1]) for k in range(len(xy1))])
            temp_dataset["line_gap"].append(0)
            temp_dataset["line_gap"].extend([(xy1[k][1]-xy2[k-1][1]) for k in range(1,len(xy1))])
            sorted_idx = np.argsort([xy1[k][1] for k in range(len(xy1))])
            temp_dataset["sorted_line_gap"].append(0)
            temp_dataset["sorted_line_gap"].extend([(xy1[sorted_idx[k]][1]-xy2[sorted_idx[k-1]][1]) for k in range(1,len(sorted_idx))])
            temp_dataset["prev_line_diff"].append(0)
            temp_dataset["prev_line_diff"].extend([(page["lines"][k]["spans"][0]["length"]-page["lines"][k-1]["spans"][0]["length"]) for k in range(1,len(page["lines"]))])
            temp_dataset["next_line_diff"].extend([(page["lines"][k]["spans"][0]["length"]-page["lines"][k-1]["spans"][0]["length"]) for k in range(1,len(page["lines"]))])
            temp_dataset["next_line_diff"].append(0)
            temp_pattern = [0 for k in range(len(page["lines"]))]
            temp_discarded_pattern = [0 for k in range(len(page["lines"]))]
            for j, line in enumerate(page["lines"]):
                for row in records:
                    if (id == row[0]) and (i == row[1]) and (j == row[2]):
                        # if row[3] == 2:
                        #     temp_pattern_type[j] = section_type
                        # else:
                        #     for idx, pattern in enumerate(patterns, start=1):
                        #         matches = re.findall(pattern, line["content"].strip(), re.IGNORECASE)
                        #         if matches:
                        #             section_type = idx 
                        #             temp_pattern_type[j] = section_type
                        #         else:
                        #             section_type = 0
                        temp_label[j] = row[3]
                    else:
                        continue
                for idx, pattern in enumerate(patterns, start=1):
                    flag = True
                    matches = re.findall(pattern, line["content"].strip(), re.IGNORECASE)
                    if matches:
                        temp_pattern_type[j] = idx
                        for discard in discards:
                            if re.search(discard, line["content"], re.IGNORECASE):
                                temp_discarded_pattern[j] = 1
                                flag = False
                                break
                        if flag:
                            temp_pattern[j] = 1        
            temp_dataset["label"].extend(temp_label)
            temp_dataset["pattern"].extend(temp_pattern)
            temp_dataset["discarded_pattern"].extend(temp_discarded_pattern)
            temp_dataset["pattern_type"].extend(temp_pattern_type)
            temp_dataset["line_number"].extend([(k+1) for k in range(len(page["lines"]))])
            page_data["width"] = page["width"]
            page_data["height"] = page["height"]
            page_data["bottom-margin"] = round(page["height"] - xy2[-1][1],4)
            page_data["top-margin"] = round(xy1[0][1],4)
            page_data["left-margin"] = round(min(xy1)[0],4)
            page_data["right-margin"] = round(page["width"] - max(xy2)[0],4)
            temp_dataset["left-align"].extend([(xy1[i][0]-page_data["left-margin"]) for i in range(len(xy1))])
            pages_data[id].append(page_data)
        temp_dataset["normalised_line_height"] = temp_dataset["line_height"] - np.median(temp_dataset["line_height"])
        temp_dataset["normalised_line_gap"] = temp_dataset["line_gap"] - np.median(temp_dataset["line_gap"])
        datasets[id] = pd.DataFrame(temp_dataset)
    return datasets, pages_data


# vectorizer = TfidfVectorizer()
# vectorized_content = vectorizer.fit_transform(datasets[1]["content"])
# print(vectorized_content.shape)
# pca = PCA(n_components=300,svd_solver="auto")
# reduced_vectorized_content = pca.fit_transform(vectorized_content)
# print(reduced_vectorized_content.shape)
# explained_variance_ratio = pca.explained_variance_ratio_
# print(explained_variance_ratio)
# cumulative_explained_variance = np.cumsum(explained_variance_ratio)
# print(cumulative_explained_variance)

def extract_features(patterns, discards, columns, metadata):
    temp_dataset = {i:[] for i in columns}
    for i, page in enumerate(metadata["contract"]["pages"],start=1):
        xy1 = [(min(line["polygon"][0]["x"],line["polygon"][3]["x"]), min(line["polygon"][0]["y"],line["polygon"][1]["y"])) for line in page["lines"]]
        xy2 = [(max(line["polygon"][1]["x"],line["polygon"][2]["x"]), max(line["polygon"][2]["y"],line["polygon"][3]["y"])) for line in page["lines"]]
        temp_dataset["length"].extend([line["spans"][0]["length"] for line in page["lines"]])
        temp_dataset["content"].extend([line["content"] for line in page["lines"]])
        temp_pattern_type = [0 for k in range(len(page["lines"]))]
        temp_dataset["line_height"].extend([(xy2[k][1]-xy1[k][1]) for k in range(len(xy1))])
        temp_dataset["line_gap"].append(0)
        temp_dataset["line_gap"].extend([(xy1[k][1]-xy2[k-1][1]) for k in range(1,len(xy1))])
        sorted_idx = np.argsort([xy1[k][1] for k in range(len(xy1))])
        temp_dataset["sorted_line_gap"].append(0)
        temp_dataset["sorted_line_gap"].extend([(xy1[sorted_idx[k]][1]-xy2[sorted_idx[k-1]][1]) for k in range(1,len(sorted_idx))])
        temp_dataset["prev_line_diff"].append(0)
        temp_dataset["prev_line_diff"].extend([(page["lines"][k]["spans"][0]["length"]-page["lines"][k-1]["spans"][0]["length"]) for k in range(1,len(page["lines"]))])
        temp_dataset["next_line_diff"].extend([(page["lines"][k]["spans"][0]["length"]-page["lines"][k-1]["spans"][0]["length"]) for k in range(1,len(page["lines"]))])
        temp_dataset["next_line_diff"].append(0)
        temp_pattern = [0 for k in range(len(page["lines"]))]
        temp_discarded_pattern = [0 for k in range(len(page["lines"]))]
        for j, line in enumerate(page["lines"]):
            for idx, pattern in enumerate(patterns, start=1):
                flag = True
                matches = re.findall(pattern, line["content"].strip(), re.IGNORECASE)
                if matches:
                    temp_pattern_type[j] = idx
                    for discard in discards:
                        if re.search(discard, line["content"], re.IGNORECASE):
                            temp_discarded_pattern[j] = 1
                            flag = False
                            break
                    if flag:
                        temp_pattern[j] = 1        
        temp_dataset["pattern"].extend(temp_pattern)
        temp_dataset["discarded_pattern"].extend(temp_discarded_pattern)
        temp_dataset["pattern_type"].extend(temp_pattern_type)
        temp_dataset["line_number"].extend([(k+1) for k in range(len(page["lines"]))])
        temp_dataset["left-align"].extend([(xy1[k][0]-round(min(xy1)[0],4)) for k in range(len(xy1))])
    temp_dataset["normalised_line_height"] = temp_dataset["line_height"] - np.median(temp_dataset["line_height"])
    temp_dataset["normalised_line_gap"] = temp_dataset["line_gap"] - np.median(temp_dataset["line_gap"])
    dataset = pd.DataFrame(temp_dataset)
    return dataset