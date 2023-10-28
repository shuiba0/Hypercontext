import pickle

import jsonlines
import numpy as np
from torch.utils.data import Dataset
import os, gzip, json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import random, time
import asyncio
import aiofiles
import aiohttp
import asyncio
import gzip
from concurrent.futures import ThreadPoolExecutor
import itertools

# 创建一个包含10个线程的线程池
def extract_percentage_data(lst, percentage):
    length = len(lst)
    num_elements = int(length * percentage)
    return lst[:num_elements]



class Seq2SeqC4Pretrain(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        data_split,
        max_length=32,
        training_data_ratio=1.0,
        minimum_token_length=5,
        # return_view=False,
        # all_views=False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.data = []
        train_data_paths =[os.path.join(data_dir, each) for each in os.listdir(data_dir) if "train" in each]
        train_data_paths = extract_percentage_data(train_data_paths, training_data_ratio)
        val_data_paths =[os.path.join(data_dir, each) for each in os.listdir(data_dir) if "validation" in each]

        if data_split == "train":
            self.file_paths = train_data_paths
        elif data_split == "validation":
            self.file_paths = val_data_paths
        else:
            raise NotImplementedError


        self.samples_counts_file = os.path.join("/zhangshuibai/KnowledgeEditor", f"sample_counts_{data_split}.pkl")
        if os.path.exists(self.samples_counts_file):
            with open(self.samples_counts_file, "rb") as f:
                self.samples_per_file = pickle.load(f)
        else:
            self.samples_per_file = [self.get_sample_count(file) for file in self.file_paths]
            with open(self.samples_counts_file, "wb") as f:
                pickle.dump(self.samples_per_file, f)

        self.cumulative_samples = np.cumsum([0] + self.samples_per_file)

        self.valid_indices_file = os.path.join("/zhangshuibai/KnowledgeEditor", f'valid_indices_{data_split}_{str(minimum_token_length)}.pkl')
        if os.path.exists(self.valid_indices_file):
            with open(self.valid_indices_file, 'rb') as f:
                self.valid_indices = pickle.load(f)
        else:
            # self.valid_indices = self.get_valid_indices(minimum_token_length)
            global sem
            sem = asyncio.Semaphore(100)
            self.valid_indices = asyncio.get_event_loop().run_until_complete(self.get_valid_indices(minimum_token_length, data_split))
            with open(self.valid_indices_file, 'wb') as f:
                pickle.dump(self.valid_indices, f)

        # self.sentences = []
        #
        # for idx, file_path in enumerate(data_paths):
        #     with gzip.open(file_path, 'rt') as file:
        #         content = file.readlines()
        #         for line in content:
        #             text = json.loads(line.strip())["text"]
        #             self.sentences.append(text)
        #         if data_split == "train":
        #             print(f"Training dataset processing idx: {idx} done")

        # self.sentences = [sentence for sentence in self.sentences if len( self.tokenizer.tokenize(sentence) ) >10 ]

        self.max_length = max_length
        # self.all_views = all_views
        # self.return_view = return_view

        self.random_num_generator = self.random_number_generator(2023)

    # def __len__(self):
    #     return len(self.sentences)
    def __len__(self):
        # return self.cumulative_samples[-1]
        return len(self.valid_indices)

    # async def read_gzip(file):
    #     with ThreadPoolExecutor() as executor:
    #         loop = asyncio.get_event_loop()
    #         # Open the gzip file
    #         with gzip.open(file, 'rt') as f:
    #             while True:
    #                 # Read each line in the gzip file
    #                 line = await loop.run_in_executor(executor, f.readline)
    #                 if not line:
    #                     break
    #                 # Process line
    #                 print(line.strip())
    #
    def process_file(self, file_idx, file, minimum_token_length, pbar, data_split):
        valid_indices_tmp_file_path = os.path.join("/zhangshuibai/KnowledgeEditor", f"valid_indices_{data_split}_tmp_file", f"minimumu_token_length_{minimum_token_length}", f"{os.path.basename(file)}_{file_idx}.pkl")
        if os.path.exists(valid_indices_tmp_file_path):
            with open(valid_indices_tmp_file_path, "rb") as f:
                valid_indices = pickle.load(f)
        else:
            os.makedirs(os.path.dirname(valid_indices_tmp_file_path), exist_ok=True)
            valid_indices = []
            with gzip.open(file, 'rt') as f:
                data = f.readlines()
            for sample_idx, sample in enumerate(data):
                if len(self.tokenizer.tokenize(sample)) >= minimum_token_length:
                    valid_indices.append(self.cumulative_samples[file_idx] + sample_idx)

            with open(valid_indices_tmp_file_path, 'wb') as f:
                pickle.dump(valid_indices, f)

        pbar.update(len(valid_indices))

        return valid_indices

    async def async_process_file(self, file_idx, file, minimum_token_length, pbar, data_split):

        async with sem:
            return await asyncio.get_event_loop().run_in_executor(None, self.process_file, file_idx, file, minimum_token_length, pbar, data_split)

    async def get_valid_indices(self, minimum_token_length, data_split):
        pbar = tqdm(total=self.cumulative_samples[-1], desc= "Processing valid indices")
        tasks = []
        tmp_paths_zip = [(file_idx, file_path) for file_idx, file_path in enumerate(self.file_paths)]
        rng = random.Random()

        rng.seed(time.time())

        rng.shuffle(tmp_paths_zip)

        for file_idx, file in tmp_paths_zip:
            task = self.process_file(file_idx, file, minimum_token_length, pbar, data_split)
            tasks.append((file_idx, task))

        tasks = sorted(tasks, key=lambda x: x[0])
        tasks = [x[1] for x in tasks]
        tasks = list(itertools.chain.from_iterable(tasks))

        # results = await asyncio.gather(*tasks)
        results = tasks
        pbar.close()
        return results

    # def get_valid_indices(self, minimum_token_length):
    #     valid_indices = []
    #     for file_idx, file in enumerate(self.file_paths):
    #         # data = np.load(os.path.join(self.directory, file))
    #         with gzip.open(file, 'rt') as file:
    #             data = file.readlines()
    #         for sample_idx, sample in enumerate(data):
    #             if len(self.tokenizer.tokenize(sample)) >= minimum_token_length:
    #                 valid_indices.append(self.cumulative_samples[file_idx] + sample_idx)
    #     return valid_indices

    # def process_file(self, file, minimum_token_length, file_idx, valid_indices, pbar):
    #     with gzip.open(file, 'rt') as f:
    #         data = f.readlines()
    #     for sample_idx, sample in enumerate(data):
    #         if len(self.tokenizer.tokenize(sample)) >= minimum_token_length:
    #             valid_indices.append(self.cumulative_samples[file_idx] + sample_idx)
    #         pbar.update(1)

    # def get_valid_indices(self, minimum_token_length):
    #     valid_indices = []
    #     with ThreadPoolExecutor() as executor, tqdm(total=self.cumulative_samples[-1], desc="Processing valid indices") as pbar:
    #         for file_idx, file in enumerate(self.file_paths):
    #             executor.submit(self.process_file, file, minimum_token_length, file_idx, valid_indices, pbar)
    #     return valid_indices

    def get_sample_count(self, file):
        with gzip.open(file, 'rt') as file:
            content = file.readlines()
            return len(content)

    def get_each_content(self, file_name, sample_idx):
        with gzip.open(file_name, 'rt') as file:
            content = file.readlines()
            return content[sample_idx]

    def __getitem__(self, item):
        valid_idx = self.valid_indices[item]
        file_idx = np.searchsorted(self.cumulative_samples, valid_idx, side="right") - 1
        sample_idx = valid_idx - self.cumulative_samples[file_idx]
        file_name = self.file_paths[file_idx]
        data = self.get_each_content(file_name, sample_idx)

        trunc_length = next(self.random_num_generator)
        seq_length = len( self.tokenizer.tokenize(data))

        trunc_length = trunc_length if trunc_length <= seq_length-2 else seq_length-2

        tokens = self.tokenizer.tokenize(data)
        src_sentence = self.tokenizer.convert_tokens_to_string(tokens[:trunc_length])
        trg_sentence = self.tokenizer.convert_tokens_to_string(tokens[trunc_length:])

        src_sentence = self.tokenizer.pad_token + src_sentence
        # trg_sentence = self.tokenizer.pad_token + trg_sentence#   trg_sentence需要加pad_token吗？
        trg_sentence = trg_sentence


        if not src_sentence:
            print("trg_sentence:")
            print(trg_sentence)
            print("full_sentence")
            print(self.sentences[item])
            raise ValueError

        if not trg_sentence:
            print("src_sentence:")
            print(src_sentence)
            print("full_sentence")
            print(self.sentences[item])
            raise ValueError

        return {
            "src":src_sentence,
            "trg":trg_sentence
        }

    def random_number_generator(self, seed):
        random.seed(seed)  # 设置种子值
        while True:
            yield random.randint(2, self.max_length-2)

    # def get_batch(self, sentences, condition):#sentences是干嘛的， get_batch这个函数是干嘛的
        # batch = {
        #     "{}_{}".format(k1, k2): v2
        #     for k1, v1 in {
        #         "src": sentences
        #         + [condition.split("|| ")[1]] * (1 + int(self.return_view)),
        #         "trg": [condition.split(" || ")[0].split(" >> ")[1]]
        #         * (len(sentences) + 1 + int(self.return_view)),
        #         "cond": [condition],
        #     }.items()
        #     for k2, v2 in self.tokenizer(
        #         v1,
        #         return_tensors="pt",
        #         padding=True,
        #         max_length=self.max_length,
        #         truncation=True,
        #     ).items()
        # }
        # batch = {
        #     "{}_{}".format(k1, k2): v2
        #     for k1, v1 in {
        #         "src_sentence": src_sentences
        #                + [condition.split("|| ")[1]] * (1 + int(self.return_view)),
        #         "trg_sentence": [condition.split(" || ")[0].split(" >> ")[1]]
        #                * (len(sentences) + 1 + int(self.return_view)),
        #     }.items()
        #     for k2, v2 in self.tokenizer(
        #         v1,
        #         return_tensors="pt",
        #         padding=True,
        #         max_length=self.max_length,
        #         truncation=True,
        #     ).items()
        # }
        # batch["trg_input_ids"][:, 0] = self.tokenizer.eos_token_id

        # print("-----------------------")
        # print("output batch:")
        # print(batch)
        # input("666666666")
        # print("-----------------------")

        # return batch

    def collate_fn(self, batch):
        src = [b["src"] for b in batch]
        trg = [b["trg"] for b in batch]
        batches = {
            "{}_{}".format(k1, k2): v2
            for k1, v1 in {
                "src": src,
                "trg": trg,
            }.items()
            for k2, v2 in self.tokenizer(
                v1,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
                # padding="max_length",

            ).items()
        }

        # print(batches["src_input_ids"].shape)
        # input(("88press to continue"))

        batches["raw"] = batch

        # print("-----------------------")
        # print("output batch:")
        # for each in batches["raw"]:
        #     print(each)
        # # print(batches["raw"])
        # input("666666666")
        # print("-----------------------")
        return batches
