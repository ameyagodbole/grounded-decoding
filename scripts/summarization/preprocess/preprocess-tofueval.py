from datasets import load_dataset
import json
import logging
import os
import pandas as pd

logging.basicConfig(
    format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.DEBUG
)
logger = logging.getLogger(__name__)


if not os.path.isdir("./data"):
    os.makedirs("./data")

if os.path.exists("./data/meetingbank_dev_doc.csv"):
    raise AssertionError("ERROR: Output files already exist!")

def obtain_dialogue_mediasum(dialogue_selected):
    dialogue_df = pd.DataFrame(columns=['doc_id', 'source'])
    for dialogue in dialogue_selected:
        dialogue_id = dialogue['id']
        speakers = dialogue['speaker']
        utts = dialogue['utt']
        transcript = ''
        for speaker, utt in zip(speakers, utts):
            transcript += f"{speaker}: {utt}\n"
        transcript = transcript.strip()
        dialogue_df.loc[len(dialogue_df)] = [dialogue_id, transcript]
    return dialogue_df

####
logger.info("Load TOFUEVAL dev/test ids")
with open("document_ids_dev_test_split.json") as file:
    document_mapping = json.load(file)

meetingbank_dev_ids = document_mapping['dev']['meetingbank']
meetingbank_test_ids = document_mapping['test']['meetingbank']
mediasum_dev_ids = document_mapping['dev']['mediasum']
mediasum_test_ids = document_mapping['test']['mediasum']
logger.debug(f"MEETINGBANK: n_docs: DEV = {len(meetingbank_dev_ids)}, TEST = {len(meetingbank_test_ids)}")
logger.debug(f"MEDIASUM: n_docs: DEV = {len(mediasum_dev_ids)}, TEST = {len(mediasum_test_ids)}")

####
logger.info("Load TOFUEVAL topics")
meetingbank_topics, mediasum_topics = {}, {}
for toful_datafile in ["./factual_consistency/meetingbank_factual_eval_dev.csv",
                       "./factual_consistency/meetingbank_factual_eval_test.csv"]:
    tofu_df = pd.read_csv(toful_datafile)
    for _, row in tofu_df[['doc_id', 'topic']].drop_duplicates().iterrows():
        if row['doc_id'] not in meetingbank_topics:
            meetingbank_topics[row['doc_id']] = []
        meetingbank_topics[row['doc_id']].append(row['topic'])
logger.debug(f"MEETINGBANK: n_docs = {len(meetingbank_topics)}, n_topics = {sum(len(v_) for v_ in meetingbank_topics.values())}")

for toful_datafile in ["./factual_consistency/mediasum_factual_eval_dev.csv",
                       "./factual_consistency/mediasum_factual_eval_test.csv"]:
    tofu_df = pd.read_csv(toful_datafile)
    for _, row in tofu_df[['doc_id', 'topic']].drop_duplicates().iterrows():
        if row['doc_id'] not in mediasum_topics:
            mediasum_topics[row['doc_id']] = []
        mediasum_topics[row['doc_id']].append(row['topic'])
logger.debug(f"MEDIASUM: n_docs = {len(mediasum_topics)}, n_topics = {sum(len(v_) for v_ in mediasum_topics.values())}")

####
logger.info("Loading and filtering MEETINGBANK")
meetingbank = pd.DataFrame(load_dataset("lytang/MeetingBank-transcript")['test'])
meetingbank_dev = meetingbank[meetingbank.meeting_id.isin(meetingbank_dev_ids)][['meeting_id', 'source']].reset_index(drop=True)
meetingbank_test = meetingbank[meetingbank.meeting_id.isin(meetingbank_test_ids)][['meeting_id', 'source']].reset_index(drop=True)
logger.debug(f"Raw data counts: DEV = {len(meetingbank_dev)}, TEST = {len(meetingbank_test)}")

meetingbank_dev_data = []
for _, row in meetingbank_dev.iterrows():
    for topic_ctr, topic in enumerate(meetingbank_topics[row['meeting_id']]):
        new_data = {'ex_id': f"{row['meeting_id']}_{topic_ctr}",
                    'meeting_id': row['meeting_id'],
                    'topic_ctr': topic_ctr,
                    'topic': topic,
                    'source': row['source']
                    }
        meetingbank_dev_data.append(new_data)
meetingbank_test_data = []
for _, row in meetingbank_test.iterrows():
    for topic_ctr, topic in enumerate(meetingbank_topics[row['meeting_id']]):
        new_data = {'ex_id': f"{row['meeting_id']}_{topic_ctr}",
                    'meeting_id': row['meeting_id'],
                    'topic_ctr': topic_ctr,
                    'topic': topic,
                    'source': row['source']
                    }
        meetingbank_test_data.append(new_data)
logger.debug(f"Data counts w topics: DEV = {len(meetingbank_dev_data)}, TEST = {len(meetingbank_test_data)}")

with open("./data/meetingbank_dev_doc.jsonl", 'w') as fout:
    for new_data in meetingbank_dev_data:
        fout.write(json.dumps(new_data) + '\n')
with open("./data/meetingbank_test_doc.jsonl", 'w') as fout:
    for new_data in meetingbank_test_data:
        fout.write(json.dumps(new_data) + '\n')

####
logger.info("Loading and filtering MEDIASUM")
with open("/home/ameya/datasets/MediaSum/news_dialogue.json") as file:
    news_dialogue = json.load(file)
dialogue_dev = [dialogue for dialogue in news_dialogue if dialogue['id'] in mediasum_dev_ids]
dialogue_test = [dialogue for dialogue in news_dialogue if dialogue['id'] in mediasum_test_ids]

mediasum_dev = obtain_dialogue_mediasum(dialogue_dev)
mediasum_test = obtain_dialogue_mediasum(dialogue_test)
logger.debug(f"Raw data counts: DEV = {len(mediasum_dev)}, TEST = {len(mediasum_test)}")

mediasum_dev_data = []
for _, row in mediasum_dev.iterrows():
    for topic_ctr, topic in enumerate(mediasum_topics[row['doc_id']]):
        new_data = {'ex_id': f"{row['doc_id']}_{topic_ctr}",
                    'doc_id': row['doc_id'],
                    'topic_ctr': topic_ctr,
                    'topic': topic,
                    'source': row['source']
                    }
        mediasum_dev_data.append(new_data)
mediasum_test_data = []
for _, row in mediasum_test.iterrows():
    for topic_ctr, topic in enumerate(mediasum_topics[row['doc_id']]):
        new_data = {'ex_id': f"{row['doc_id']}_{topic_ctr}",
                    'doc_id': row['doc_id'],
                    'topic_ctr': topic_ctr,
                    'topic': topic,
                    'source': row['source']
                    }
        mediasum_test_data.append(new_data)
logger.debug(f"Data counts w topics: DEV = {len(mediasum_dev_data)}, TEST = {len(mediasum_test_data)}")

with open("./data/mediasum_dev_doc.jsonl", 'w') as fout:
    for new_data in mediasum_dev_data:
        fout.write(json.dumps(new_data) + '\n')
with open("./data/mediasum_test_doc.jsonl", 'w') as fout:
    for new_data in mediasum_test_data:
        fout.write(json.dumps(new_data) + '\n')