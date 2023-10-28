import os
from argparse import ArgumentParser
from pprint import pprint
import sys
sys.path.append("/zhangshuibai/KnowledgeEditor")
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
# from pytorch_lightning.plugins.training_type.rpc_sequential import RPCSequentialPlugin
from src.models.bart_seq2seq_augmented_kilt import BartSeq2SeqAugmented


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--log_dirpath",
        type=str,
        default="/zhangshuibai/KnowledgeEditor/logs",
    )
    parser.add_argument("--save_top_k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--model_name", type=str, default="/zhangshuibai/data1_models/gpt2")#select the model
    parser.add_argument("--conditioner_model_path", type=str, default="/zhangshuibai/data1_models/gpt2")#select the conditioner

    # parser.add_argument("--max_epochs", default=1)
    # parser.add_argument("--accelerator", default="gpu")
    # parser.add_argument("--devices", default=4)

    parser = BartSeq2SeqAugmented.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args, _ = parser.parse_known_args()
    pprint(args.__dict__)

    seed_everything(seed=args.seed)

    logger = TensorBoardLogger(args.log_dirpath, name=None)

    callbacks = [
        ModelCheckpoint(
            monitor="validation_perlexity",#through self.log in validation_step
            mode="min",
            dirpath=os.path.join(logger.log_dir, "checkpoints"),
            # save_top_k=args.save_top_k,
            filename="test_model",
        ),
        LearningRateMonitor(
            logging_interval="step",
        ),
    ]

    trainer = Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks,
                                         limit_train_batches=0.001,
                                         limit_val_batches=0.1,
                                         )

    model = BartSeq2SeqAugmented(**vars(args))

    # plugin = RPCSequentialPlugin(balance=[1, 1])
    # trainer = Trainer(accelerator='ddp', gpus=2, plugins=[plugin])

    # summary = ModelSummary(model, max_depth=-1)
    # print("99999")
    # print(summary)

    trainer.fit(model)
