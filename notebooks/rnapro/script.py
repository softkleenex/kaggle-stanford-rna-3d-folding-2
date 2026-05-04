!wget https://huggingface.co/nvidia/RNAPro-Private-Best-500M/resolve/main/rnapro-private-best-500m.ckpt
!git clone https://github.com/NVIDIA-Digital-Bio/RNAPro.git
!mkdir wheels
cd wheels
pip download -r ../RNAPro/requirements.txt
cd /kaggle/working/
pip install -r RNAPro/requirements.txt --find-links=./wheels/
cd RNAPro
pip install -e . --no-deps
!python preprocess/convert_templates_to_pt_files.py --input_csv /kaggle/input/stanford-rna-3d-folding-part-2-templates/templates_tbm.csv --output_name templates.pt
DIST = "/kaggle/working/RNAPro/release_data/ccd_cache/"
!mkdir -p $DIST
# !python preprocess/gen_ccd_cache.py -c $DIST
!cp /kaggle/input/protenix-checkpoints/components.v20240608.cif $DIST
!cp /kaggle/input/protenix-checkpoints/components.v20240608.cif.rdkit_mol.pkl $DIST
cd /kaggle/working/RNAPro
%%python
import pandas as pd
df = pd.read_csv("/kaggle/input/stanford-rna-3d-folding-2/test_sequences.csv")
df = df.head(5)
df.to_csv('/kaggle/working/sample_sequences.csv', index=False)
%%writefile runner/inference.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import logging
import traceback
import warnings
import argparse
from contextlib import nullcontext
from os.path import join as opjoin
from typing import Any, Mapping

import json
import torch
import pandas as pd
import numpy as np
from biotite.structure.io import pdbx

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from runner.dumper import DataDumper

from rnapro.config import parse_sys_args
from rnapro.config.config import ConfigManager, ArgumentNotSet
from rnapro.data.infer_data_pipeline import get_inference_dataloader
from rnapro.model.RNAPro import RNAPro
from rnapro.utils.distributed import DIST_WRAPPER
from rnapro.utils.seed import seed_everything
from rnapro.utils.torch_utils import to_device

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger(__name__)

# Silence all info logging
logging.basicConfig(level=logging.WARNING)
# Silence dataloader logging specifically
logging.getLogger("rnapro.data").setLevel(logging.WARNING)
logging.getLogger("rnapro").setLevel(logging.WARNING)


def parse_configs(
    configs: dict, arg_str: str = None, fill_required_with_null: bool = False
):
    """
    Parses and merges configuration settings from a dictionary and command-line arguments.

    Args:
        configs (dict): A dictionary containing initial configuration settings.
        arg_str (str, optional): A string representing command-line arguments. Defaults to None.
        fill_required_with_null (bool, optional):
            A boolean flag indicating whether required values should be filled with `None` if not provided. Defaults to False.

    Returns:
        ConfigDict: The merged configuration dictionary.
    """
    manager = ConfigManager(configs, fill_required_with_null=fill_required_with_null)
    parser = argparse.ArgumentParser()

    # This is new
    parser.add_argument(
        "--max_len",
        type=int,
        default=10000,
        required=False,
        help="Maximum length of the sequence. Longer sequences will be skipped during inference"
    )

    # Register arguments
    for key, (
        dtype,
        default_value,
        allow_none,
        required,
    ) in manager.config_infos.items():
        # All config use str type, strings will be converted to real dtype later
        parser.add_argument(
            "--" + key, type=str, default=ArgumentNotSet(), required=required
        )
    # Merge user commandline pargs with default ones
    merged_configs = manager.merge_configs(
        vars(parser.parse_args(arg_str.split())) if arg_str else {}
    )

    max_len = parser.parse_args(arg_str.split()).max_len
    merged_configs.max_len = max_len

    return merged_configs


class dotdict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


class InferenceRunner(object):
    def __init__(self, configs: Any) -> None:
        self.configs = configs
        self.init_env()
        self.init_basics()
        self.init_model()
        self.load_checkpoint()
        self.init_dumper(
            need_atom_confidence=configs.need_atom_confidence,
            sorted_by_ranking_score=configs.sorted_by_ranking_score,
        )

    def init_env(self) -> None:
        self.print(
            f"Distributed environment: world size: {DIST_WRAPPER.world_size}, "
            + f"global rank: {DIST_WRAPPER.rank}, local rank: {DIST_WRAPPER.local_rank}"
        )
        self.use_cuda = torch.cuda.device_count() > 0
        if self.use_cuda:
            self.device = torch.device("cuda:{}".format(DIST_WRAPPER.local_rank))
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            all_gpu_ids = ",".join(str(x) for x in range(torch.cuda.device_count()))
            devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
            logging.info(
                f"LOCAL_RANK: {DIST_WRAPPER.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]"
            )
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        if self.configs.use_deepspeed_evo_attention:
            env = os.getenv("CUTLASS_PATH", None)
            self.print(f"env: {env}")
            assert (
                env is not None
            ), "if use ds4sci, set `CUTLASS_PATH` env as https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/"
            if env is not None:
                logging.info(
                    "The kernels will be compiled when DS4Sci_EvoformerAttention is called for the first time."
                )
        use_fastlayernorm = os.getenv("LAYERNORM_TYPE", None)
        if use_fastlayernorm == "fast_layernorm":
            logging.info(
                "The kernels will be compiled when fast_layernorm is called for the first time."
            )

        logging.info("Finished init ENV.")

    def init_basics(self) -> None:
        self.dump_dir = self.configs.dump_dir
        self.error_dir = opjoin(self.dump_dir, "ERR")
        os.makedirs(self.dump_dir, exist_ok=True)
        os.makedirs(self.error_dir, exist_ok=True)

    def init_model(self) -> None:
        self.model = RNAPro(self.configs).to(self.device)
        num_params = sum(p.numel() for p in self.model.parameters())
        self.print(f"Total number of parameters: {num_params:,}")

    def load_checkpoint(self) -> None:
        checkpoint_path = self.configs.load_checkpoint_path

        if not os.path.exists(checkpoint_path):
            raise Exception(f"Given checkpoint path not exist [{checkpoint_path}]")
        self.print(
            f"Loading from {checkpoint_path}, strict: {self.configs.load_strict}"
        )
        checkpoint = torch.load(checkpoint_path, self.device)

        sample_key = [k for k in checkpoint["model"].keys()][0]
        # self.print(f"Sampled key: {sample_key}")
        if sample_key.startswith("module."):  # DDP checkpoint has module. prefix
            checkpoint["model"] = {
                k[len("module."):]: v for k, v in checkpoint["model"].items()
            }
        self.model.load_state_dict(
            state_dict=checkpoint["model"],
            strict=True,
        )
        self.model.eval()
        # self.print("Finish loading checkpoint.")

    def init_dumper(
        self, need_atom_confidence: bool = False, sorted_by_ranking_score: bool = True
    ):
        self.dumper = DataDumper(
            base_dir=self.dump_dir,
            need_atom_confidence=need_atom_confidence,
            sorted_by_ranking_score=sorted_by_ranking_score,
        )

    def print_dict(self, d):
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: ", v.shape)
            else:
                pass
                # print(f"{k}: {v}")

    # Adapted from runner.train.Trainer.evaluate
    @torch.no_grad()
    def predict(self, data: Mapping[str, Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        eval_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]
        # print("eval_precision: ", eval_precision)
        enable_amp = (
            torch.autocast(device_type="cuda", dtype=eval_precision)
            if torch.cuda.is_available()
            else nullcontext()
        )
        #         print('input_feature_dict: ', self.print_dict(data["input_feature_dict"]))
        #         exit(0)

        data = to_device(data, self.device)
        with enable_amp:
            prediction, _, _ = self.model(
                input_feature_dict=data["input_feature_dict"],
                label_full_dict=None,
                label_dict=None,
                mode="inference",
            )

        return prediction

    def print(self, msg: str):
        if DIST_WRAPPER.rank == 0:
            # logger.info(msg)
            print(msg)

    def update_model_configs(self, new_configs: Any) -> None:
        self.model.configs = new_configs


def update_inference_configs(configs: Any, N_token: int):
    # Setting the default inference configs for different N_token and N_atom
    # when N_token is larger than 3000, the default config might OOM even on a
    # A100 80G GPUS,
    if N_token > 3840:
        configs.skip_amp.confidence_head = False
        configs.skip_amp.sample_diffusion = False
    elif N_token > 2560:
        configs.skip_amp.confidence_head = False
        configs.skip_amp.sample_diffusion = True
    else:
        configs.skip_amp.confidence_head = True
        configs.skip_amp.sample_diffusion = True
    return configs


def infer_predict(runner: InferenceRunner, configs: Any) -> None:
    # Data
    # logger.info(f"Loading data from {configs.input_json_path}")
    try:
        dataloader = get_inference_dataloader(configs=configs)
    except Exception as e:
        error_message = f"{e}:\n{traceback.format_exc()}"
        logger.info(error_message)
        with open(opjoin(runner.error_dir, "error.txt"), "a") as f:
            f.write(error_message)
        return

    num_data = len(dataloader.dataset)
    for seed in configs.seeds:
        seed_everything(seed=seed, deterministic=configs.deterministic)
        for batch in dataloader:
            try:
                data, atom_array, data_error_message = batch[0]
                sample_name = data["sample_name"]

                if len(data_error_message) > 0:
                    logger.info(data_error_message)
                    with open(opjoin(runner.error_dir, f"{sample_name}.txt"), "a") as f:
                        f.write(data_error_message)
                    continue

                logger.info(
                    (
                        f"[Rank {DIST_WRAPPER.rank} ({data['sample_index'] + 1}/{num_data})] {sample_name}: "
                        f"N_asym {data['N_asym'].item()}, N_token {data['N_token'].item()}, "
                        f"N_atom {data['N_atom'].item()}, N_msa {data['N_msa'].item()}"
                    )
                )
                new_configs = update_inference_configs(configs, data["N_token"].item())
                runner.update_model_configs(new_configs)
                prediction = runner.predict(data)
                runner.dumper.dump(
                    dataset_name="",
                    pdb_id=sample_name,
                    seed=seed,
                    pred_dict=prediction,
                    atom_array=atom_array,
                    entity_poly_type=data["entity_poly_type"],
                )

                logger.info(
                    f"[Rank {DIST_WRAPPER.rank}] {data['sample_name']} succeeded - "
                    f"Results saved to {configs.dump_dir}"
                )
                torch.cuda.empty_cache()
            except Exception as e:
                error_message = f"[Rank {DIST_WRAPPER.rank}]{data['sample_name']} {e}:\n{traceback.format_exc()}"
                logger.info(error_message)
                # Save error info
                with open(opjoin(runner.error_dir, f"{sample_name}.txt"), "a") as f:
                    f.write(error_message)
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()


# data helper
def make_dummy_solution(valid_df):
    solution = dotdict()
    for i, row in valid_df.iterrows():
        target_id = row.target_id
        sequence = row.sequence
        solution[target_id] = dotdict(
            target_id=target_id,
            sequence=sequence,
            coord=[],
        )
    return solution


def solution_to_submit_df(solution):
    submit_df = []
    for k, s in solution.items():
        df = coord_to_df(s.sequence, s.coord, s.target_id)
        submit_df.append(df)

    submit_df = pd.concat(submit_df)
    return submit_df


def coord_to_df(sequence, coord, target_id):
    L = len(sequence)
    df = pd.DataFrame()
    df["ID"] = [f"{target_id}_{i + 1}" for i in range(L)]
    df["resname"] = [s for s in sequence]
    df["resid"] = [i + 1 for i in range(L)]

    num_coord = len(coord)
    for j in range(num_coord):
        df[f"x_{j+1}"] = coord[j][:, 0]
        df[f"y_{j+1}"] = coord[j][:, 1]
        df[f"z_{j+1}"] = coord[j][:, 2]
    return df


def main(configs: Any) -> None:
    # Runner
    runner = InferenceRunner(configs)
    infer_predict(runner, configs)


def create_input_json(sequence, target_id):
    # print("input_no_msa")
    input_json = [
        {
            "sequences": [
                {
                    "rnaSequence": {
                        "sequence": sequence,
                        "count": 1,
                    }
                }
            ],
            "name": target_id,
        }
    ]
    return input_json


def extract_c1_coordinates(cif_file_path):
    try:
        # Read the CIF file using the correct biotite method
        with open(cif_file_path, "r") as f:
            cif_data = pdbx.CIFFile.read(f)

        # Get structure from CIF data
        atom_array = pdbx.get_structure(cif_data, model=1)

        # Clean atom names and find C1' atoms
        atom_names_clean = np.char.strip(atom_array.atom_name.astype(str))
        mask_c1 = atom_names_clean == "C1'"
        c1_atoms = atom_array[mask_c1]

        if len(c1_atoms) == 0:
            print(f"Warning: No C1' atoms found in {cif_file_path}")
            return None

        # Sort by residue ID and return coordinates
        sort_indices = np.argsort(c1_atoms.res_id)
        c1_atoms_sorted = c1_atoms[sort_indices]
        c1_coords = c1_atoms_sorted.coord

        return c1_coords
    except Exception as e:
        print(f"Error extracting C1' coordinates from {cif_file_path}: {e}")
        return None


def process_sequence(sequence, target_id, temp_dir):
    # Create input JSON
    input_json = create_input_json(sequence, target_id)

    # Save JSON to temporary file
    os.makedirs(temp_dir, exist_ok=True)
    input_json_path = os.path.join(temp_dir, f"{target_id}_input.json")
    with open(input_json_path, "w") as f:
        json.dump(input_json, f, indent=4)


def run_ptx(target_id, sequence, configs, solution, template_idx, runner):
    # Create directories
    temp_dir = f"./{configs.dump_dir}/input"  # Same as in kaggle_inference.py
    output_dir = f"./{configs.dump_dir}/output"  # Same as in kaggle_inference.py
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    process_sequence(sequence=sequence, target_id=target_id, temp_dir=temp_dir)
    configs.input_json_path = os.path.join(temp_dir, f"{target_id}_input.json")
    configs.template_idx = int(template_idx)

    infer_predict(runner, configs)

    cif_file_path = (
        f"{configs.dump_dir}/{target_id}/seed_42/predictions/{target_id}_sample_0.cif"
    )
    cif_new_path = f"{configs.dump_dir}/{target_id}/seed_42/predictions/{target_id}_sample_{template_idx}_new.cif"
    shutil.copy(cif_file_path, cif_new_path)
    coord = extract_c1_coordinates(cif_file_path)
    if coord is None:
        coord = np.zeros((len(sequence), 3), dtype=np.float32)
    elif coord.shape[0] < (len(sequence)):
        pad_len = len(sequence) - coord.shape[0]
        pad = np.zeros((pad_len, 3), dtype=np.float32)
        coord = np.concatenate([coord, pad], axis=0)
    solution[target_id].coord.append(coord)


def run() -> None:
    LOG_FORMAT = "%(asctime)s,%(msecs)-3d %(levelname)-8s [%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    logging.basicConfig(
        format=LOG_FORMAT,
        level=logging.WARNING,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )
    # Silence dataloader and rnapro module logging
    logging.getLogger("rnapro.data").setLevel(logging.WARNING)
    logging.getLogger("rnapro").setLevel(logging.WARNING)
    configs_base["use_deepspeed_evo_attention"] = (
        os.environ.get("USE_DEEPSPEED_EVO_ATTENTION", False) == "true"
    )
    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    configs = parse_configs(
        configs=configs,
        arg_str=parse_sys_args(),
        fill_required_with_null=True,
    )

    valid_df = pd.read_csv(configs.sequences_csv)
    print(f"\n -> Loaded {len(valid_df)} sequence(s)")

    # Build model and load checkpoint once before looping over sequences

    print('\n -> Building model and loading checkpoint')
    runner = InferenceRunner(configs)
    print('\n -> Done, starting inference...')

    solution = make_dummy_solution(valid_df)
    for idx, row in valid_df.iterrows():
        print(f"\n -> Sequence {row.target_id}: {row.sequence}")

        if len(row.sequence) > configs.max_len:
            print(f'Sequence is too long ({len(row.sequence)} > {configs.max_len}), skipping')
            for template_idx in range(5):
                coord = np.zeros((len(row.sequence), 3), dtype=np.float32)
                solution[row.target_id].coord.append(coord)
            continue

        try:
            target_id = row.target_id
            sequence = row.sequence
            for template_idx in range(5):
                print()
                run_ptx(
                    target_id=target_id,
                    sequence=sequence,
                    configs=configs,
                    solution=solution,
                    template_idx=template_idx,
                    runner=runner,
                )
        except Exception as e:
            print(f"Error processing {row.target_id}: {e}")
            continue

    print('\n\n -> Inference done ! Saving to submission.csv')
    submit_df = solution_to_submit_df(solution)
    submit_df = submit_df.fillna(0.0)
    submit_df.to_csv("./submission.csv", index=False)


if __name__ == "__main__":
    run()
%%writefile rnapro_inference_kaggle.sh

export LAYERNORM_TYPE=torch # fast_layernorm, torch


# Inference parameters (RNAPro)
SEED=42
N_SAMPLE=1
N_STEP=200
N_CYCLE=10

# Paths
DUMP_DIR="../output"
# Set a valid checkpoint file path below
CHECKPOINT_PATH="../rnapro-private-best-500m.ckpt"

# Template/MSA settings
TEMPLATE_DATA="./release_data/kaggle/templates.pt"
# Note: template_idx supports 5 choices and maps to top-k:
# 0->top1, 1->top2, 2->top3, 3->top4, 4->top5
TEMPLATE_IDX=0
RNA_MSA_DIR="/kaggle/input/stanford-rna-3d-folding-2/MSA"

# SEQUENCES_CSV="/kaggle/input/stanford-rna-3d-folding-2/test_sequences.csv"
SEQUENCES_CSV="/kaggle/working/sample_sequences.csv"

# RibonanzaNet2 path (keep as-is per request)
RIBONANZA_PATH="/kaggle/input/ribonanzanet2/pytorch/alpha/1/"

# Model selection: keep to an existing key to align defaults (N_step=200, N_cycle=10)
MODEL_NAME="rnapro_base"
mkdir -p "${DUMP_DIR}"

python3 runner/inference.py \
    --model_name "${MODEL_NAME}" \
    --seeds ${SEED} \
    --dump_dir "${DUMP_DIR}" \
    --load_checkpoint_path "${CHECKPOINT_PATH}" \
    --use_msa true \
    --use_template "ca_precomputed" \
    --model.use_template "ca_precomputed" \
    --model.use_RibonanzaNet2 true \
    --model.template_embedder.n_blocks 2 \
    --model.ribonanza_net_path "${RIBONANZA_PATH}" \
    --template_data "${TEMPLATE_DATA}" \
    --template_idx ${TEMPLATE_IDX} \
    --rna_msa_dir "${RNA_MSA_DIR}" \
    --model.N_cycle ${N_CYCLE} \
    --sample_diffusion.N_sample ${N_SAMPLE} \
    --sample_diffusion.N_step ${N_STEP} \
    --load_strict true \
    --num_workers 0 \
    --triangle_attention "torch" \
    --triangle_multiplicative "torch" \
    --sequences_csv "${SEQUENCES_CSV}" \
    --max_len 1000


# --triangle_attention supports 'triattention', 'cuequivariance', 'deepspeed', 'torch'
# --triangle_multiplicative supports 'cuequivariance', 'torch'
# --max_len 1000: Sequences longer than max_len will be skipped to avoid oom
!bash ./rnapro_inference_kaggle.sh
!mv submission.csv ..
cd ..
!head submission.csv
