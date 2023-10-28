import torch
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from higher.patch import monkeypatch as make_functional
from transformers import GPT2Model, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config


class ConditionedParameter(torch.nn.Module):
    def __init__(self, parameter, condition_dim=1024, hidden_dim=128, max_scale=1):
        super().__init__()
        self.parameter_shape = parameter.shape#得到需要映射出的参数的shape

        if len(self.parameter_shape) == 2:
            self.conditioners = torch.nn.Sequential(
                torch.nn.utils.weight_norm(torch.nn.Linear(condition_dim, hidden_dim)),
                torch.nn.Tanh(),
                torch.nn.utils.weight_norm(
                    torch.nn.Linear(
                        hidden_dim, 2 * (parameter.shape[0] + parameter.shape[1]) + 1
                    )
                ),
            )
        elif len(self.parameter_shape) == 1:
            self.conditioners = torch.nn.Sequential(
                torch.nn.utils.weight_norm(torch.nn.Linear(condition_dim, hidden_dim)),
                torch.nn.Tanh(),
                torch.nn.utils.weight_norm(
                    torch.nn.Linear(hidden_dim, 2 * parameter.shape[0] + 1)
                ),
            )
        else:
            raise RuntimeError()

        self.max_scale = max_scale

    def forward(self, inputs, grad):#输入为condition和grads，得到对应的参数

        if len(self.parameter_shape) == 2:
            (
                conditioner_cola,
                conditioner_rowa,
                conditioner_colb,
                conditioner_rowb,
                conditioner_norm,
            ) = self.conditioners(inputs).split(
                [
                    self.parameter_shape[1],
                    self.parameter_shape[0],
                    self.parameter_shape[1],
                    self.parameter_shape[0],
                    1,
                ],
                dim=-1,
            )

            # print("self.parameter_shape:")
            # print(self.parameter_shape)
            #
            # print("inputs.shape:")
            # print(inputs.shape)
            #
            # print("conditioner_rowa: ")
            # print(conditioner_rowa.shape)
            # print("conditioner_rowa.T: ")
            # print(conditioner_rowa.T.shape)
            # print("conditioner_cola: ")
            # print(conditioner_cola.shape)
            #
            # print("conditioner_rowb: ")
            # print(conditioner_rowb.shape)
            # print("conditioner_rowb.T: ")
            # print(conditioner_rowb.T.shape)
            # print("conditioner_colb: ")
            # print(conditioner_colb.shape)

            a = conditioner_rowa.softmax(-1).T @ conditioner_cola
            b = conditioner_rowb.softmax(-1).T @ conditioner_colb


            # print("a.shape")
            # print(a.shape)
            # print("b.shape")
            # print(b.shape)
            # print("conditioner_norm.shape")
            # print(conditioner_norm.shape)
            #
            # print("grad.shape")
            # print(grad.shape)

            # temp0 = grad * a.squeeze() + b.squeeze()
            # print("temp0")
            # print(temp0.shape)
            # temp1 = temp0 * conditioner_norm.sigmoid().squeeze()
            # print("temp1")
            # print(temp1)
            # print("max_scale")
            # print(self.max_scale)
            conditioner_norm = conditioner_norm.squeeze().mean()

        elif len(self.parameter_shape) == 1:
            a, b, conditioner_norm = self.conditioners(inputs).split(
                [self.parameter_shape[0], self.parameter_shape[0], 1], dim=-1
            )
            a = a.mean(dim=0)
            b = b.mean(dim=0)
            conditioner_norm = conditioner_norm.mean(dim=0)

        else:
            raise RuntimeError()

        return (
            self.max_scale
            * conditioner_norm.sigmoid().squeeze()
            * (grad * a.squeeze() + b.squeeze())#利用梯度信息，
        )

#
# class LSTMConditioner(torch.nn.Module):#把这个部分换成gpt2Conditioner
#     def __init__(
#         self,
#         vocab_dim=30522,
#         embedding_dim=768,
#         hidden_dim=256,
#         output_dim=1024,
#         embedding_init=None,
#     ):
#         super().__init__()
#         self.embedding = torch.nn.Embedding(
#             num_embeddings=vocab_dim,
#             embedding_dim=embedding_dim,
#             padding_idx=0,
#             _weight=embedding_init,
#         )
#         self.lstm = PytorchSeq2VecWrapper(
#             torch.nn.LSTM(
#                 input_size=embedding_dim,
#                 hidden_size=hidden_dim,
#                 num_layers=1,
#                 bidirectional=True,
#                 batch_first=True,
#             )
#         )
#         self.linear = FeedForward(
#             input_dim=hidden_dim * 2,
#             num_layers=1,
#             hidden_dims=[output_dim],
#             activations=[torch.nn.Tanh()],
#         )
#
#     def forward(self, inputs, masks):
#         return self.linear(self.lstm(self.embedding(inputs), masks))


from transformers import GPT2Model, GPT2Tokenizer

class GPT2Conditioner(torch.nn.Module):
    def __init__(
        self,
        # vocab_dim=30522,
        # hidden_dim=768,
        output_dim=1024,
        conditioner_model_path = None,
    ):
        super().__init__()
        self.conditioner_model = conditioner_model_path
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.conditioner_model)
        self.gpt2 = GPT2Model.from_pretrained(self.conditioner_model)
        # self.gpt2.eval()

        self.model_config = GPT2Config.from_pretrained(self.conditioner_model)
        self.linear = torch.nn.Linear(self.model_config.hidden_size, output_dim)

    def forward(self, inputs, masks):
        outputs = self.gpt2(input_ids=inputs, attention_mask=masks)
        hidden_states = outputs.last_hidden_state
        # print("last_hidden_satate size: ")
        # print(hidden_states.shape)
        # 获取最后一个 token 的下标
        # last_token_idx = (masks == 1).squeeze(-1).sum(dim=1) - 1
        # hidden_states = hidden_states[torch.arange(hidden_states.shape[0]):, last_token_idx, :]
        ########
        last_token_pos = torch.sum(masks, dim=-1)
        last_token_pos -= 1
        last_token_pos = last_token_pos.unsqueeze(1)#[batch_size*1]

        batch_size = hidden_states.shape[0]
        #batch_size*seq_length
        temp = torch.tensor(range(hidden_states.shape[1])).unsqueeze(0).repeat(batch_size, 1).to(last_token_pos.device)
        mask = temp == last_token_pos
        pooler = hidden_states[mask, :]
        ##########
        # print("last seq hidden states:")
        # print(pooler.shape)
        # input("press to continue")
        output = self.linear(pooler)
        return output


class OneShotLearner(torch.nn.Module):
    def __init__(
        self,
        model,#downstream model
        # vocab_dim=30522,
        # embedding_dim=768,
        hidden_dim=128,
        condition_dim=1024,
        include_set={},
        max_scale=1e-3,
        embedding_init=None,
        conditioner_model_path= None,
    ):
        super().__init__()

        self.param2conditioner_map = {
            n: "{}_conditioner".format(n).replace(".", "_")
            for n, p in model.named_parameters()
            if n in include_set
        }

        self.conditioners = torch.nn.ModuleDict(
            {
                self.param2conditioner_map[n]: ConditionedParameter(
                    p,
                    condition_dim,
                    hidden_dim,
                    max_scale=max_scale,
                )
                for n, p in model.named_parameters()
                if n in include_set
            }
        )
        # print(self.conditioners)
        # input("press to continue")

        # self.condition = LSTMConditioner(
        #     vocab_dim,
        #     embedding_dim,
        #     hidden_dim,
        #     condition_dim,
        #     embedding_init=embedding_init,
        # )
        self.condition = GPT2Conditioner(
            # vocab_dim,
            # embedding_dim,
            # hidden_dim,
            condition_dim,
            conditioner_model_path=conditioner_model_path
            # embedding_init=embedding_init,
        )

    def forward(self, inputs, masks, grads=None):
        condition = self.condition(inputs, masks)
        return {
            p: self.conditioners[self.param2conditioner_map[p]](#将同一个condition送入每个conditioner中，得到每个参数部分对应的参数更新量
                condition,
                grad=grads[p] if grads else None,
            )
            for p, c in self.param2conditioner_map.items()#得到include_set里每个参数部分的参数
        }
