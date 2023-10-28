import jsonlines
import numpy as np
from torch.utils.data import Dataset


class Seq2SeqAugmentedKILT(Dataset):
    def __init__(
        self,
        tokenizer,
        data_path,
        max_length=32,
        return_view=False,
        all_views=False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = []

        with jsonlines.open(data_path) as f:
            for d in f:
                if len(d["alternatives"]) > 0 and len(d["filtered_rephrases"]) > 0:
                    self.data.append(
                        {
                            k: d[k]
                            for k in (
                                "input",
                                "prediction",
                                "alternatives",
                                "filtered_rephrases",
                                "output",
                            )
                        }
                    )

        self.max_length = max_length
        self.all_views = all_views
        self.return_view = return_view

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item, seed=None):
        alt = np.random.RandomState(seed=seed).choice(self.data[item]["alternatives"])
        output = {
            "src": self.data[item]["input"],#问题
            "pred": self.data[item]["prediction"],
            "alt": alt,
            "answers": [x["answer"] for x in self.data[item]["output"]],
            "cond": "{} >> {} || {}".format(
                self.data[item]["prediction"],
                alt,
                self.data[item]["input"],
            ),
        }

        if self.return_view:#return_view的作用是什么
            output["view"] = (
                self.data[item]["filtered_rephrases"]
                if self.all_views
                else np.random.choice(self.data[item]["filtered_rephrases"])
            )
        # print("-----------------------")
        # print("output example:")
        # print(output)
        # print("-----------------------")

        return output

    def get_batch(self, sentences, condition):#sentences是干嘛的， get_batch这个函数是干嘛的
        batch = {
            "{}_{}".format(k1, k2): v2
            for k1, v1 in {
                "src": sentences
                + [condition.split("|| ")[1]] * (1 + int(self.return_view)),
                "trg": [condition.split(" || ")[0].split(" >> ")[1]]
                * (len(sentences) + 1 + int(self.return_view)),
                "cond": [condition],
            }.items()
            for k2, v2 in self.tokenizer(
                v1,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            ).items()
        }
        batch["trg_input_ids"][:, 0] = self.tokenizer.eos_token_id

        # print("-----------------------")
        # print("output batch:")
        # print(batch)
        # input("666666666")
        # print("-----------------------")

        return batch

    def collate_fn(self, batch):
        src = [b["src"] for b in batch]
        trg = [b["pred"] for b in batch[:-1]] + [batch[-1]["alt"]]
        #为什么数据集里有些pred是正确的，有些是错误的？
        print(src)
        print(trg)
        if self.return_view:
            src += batch[-1]["view"] if self.all_views else [batch[-1]["view"]]
            trg += [batch[-1]["alt"]] * (
                len(batch[-1]["view"]) if self.all_views else 1
            )
        print(src)
        print(trg)
        print(batch[-1]["cond"])
        batches = {
            "{}_{}".format(k1, k2): v2
            for k1, v1 in {
                "src": src,
                "trg": trg,
                "cond": [batch[-1]["cond"]],#只取batch里最后一个，其他样本用于计算约束loss（KL和Lp norm）
            }.items()
            for k2, v2 in self.tokenizer(
                v1,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            ).items()
        }

        batches["trg_input_ids"][:, 0] = self.tokenizer.eos_token_id
        batches["raw"] = batch

        # print("-----------------------")
        # print("output batch:")
        # for each in batches["raw"]:
        #     print(each)
        # # print(batches["raw"])
        # input("666666666")
        # print("-----------------------")
        return batches
