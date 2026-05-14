# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession

# model_id = "ibm-granite/granite-3.0-3b-a800m-instruct"
model_id = "ibm-granite/granite-3.0-1b-a400m-instruct"
EXPERT_BLOCKING_NUM_NSP = int(os.environ.get("EXPERT_BLOCKING_NUM_NSP", 16))
EXPERT_BLOCKING_PACKED_CHUNK_SIZE = int(os.environ.get("EXPERT_BLOCKING_PACKED_CHUNK_SIZE", 256))

prompt = """
Quantum computing is a rapidly evolving field that leverages the principles of quantum mechanics to process information in fundamentally different ways than classical computers. Unlike classical bits, which represent either 0 or 1, quantum bits (qubits) can exist in superpositions of both states simultaneously, enabling quantum computers to explore many solutions at once. Entanglement, another quantum phenomenon, allows qubits to be correlated in ways that have no classical analogue, enabling powerful parallelism and communication protocols. Quantum interference is used to amplify correct answers and cancel out wrong ones, forming the basis of most quantum algorithms.

Key quantum algorithms such as Shor's algorithm for integer factorization and Grover's algorithm for unstructured search demonstrate exponential and quadratic speedups respectively over the best known classical algorithms. These speedups have profound implications for cryptography, optimization, drug discovery, materials science, and machine learning. However, building practical quantum computers is extremely challenging due to decoherence, gate errors, and the difficulty of scaling qubit counts while maintaining low error rates.

Current quantum hardware includes superconducting qubits (used by IBM and Google), trapped ions (used by IonQ and Honeywell), photonic systems, and topological qubits (pursued by Microsoft). Each platform has different trade-offs in coherence time, gate fidelity, connectivity, and scalability. Quantum error correction codes such as the surface code require thousands of physical qubits per logical qubit, making fault-tolerant quantum computing a long-term engineering challenge.

Hybrid quantum-classical algorithms like the Variational Quantum Eigensolver (VQE) and the Quantum Approximate Optimization Algorithm (QAOA) are designed to run on near-term noisy intermediate-scale quantum (NISQ) devices. These algorithms use a classical optimizer to tune the parameters of a parameterized quantum circuit, aiming to find approximate solutions to hard optimization and simulation problems. Despite their promise, demonstrating a clear quantum advantage on practically relevant problems with NISQ devices remains an open research question.

The quantum software stack includes quantum programming languages and frameworks such as Qiskit, Cirq, PennyLane, and Braket, which allow researchers to design, simulate, and execute quantum circuits. Quantum simulators running on classical hardware can faithfully emulate small quantum systems but become exponentially expensive as qubit count grows, motivating the need for real quantum hardware. Cloud-based quantum computing services from IBM, Google, Amazon, and Microsoft are making quantum hardware accessible to researchers and developers worldwide.

Looking ahead, the roadmap toward fault-tolerant quantum computing involves improving qubit quality, developing better error correction schemes, and scaling to millions of physical qubits. Quantum networking and quantum internet protocols are also being explored to enable distributed quantum computation and provably secure quantum key distribution. The intersection of quantum computing with artificial intelligence, particularly quantum machine learning, is an active area of research with both theoretical and practical open problems.
"""

# prompt = """
# Explain quantum computing in simple terms. and do not generate first token as newline.
# """

# print(f'Prompt : {prompt}\n\n')
print(f'EXPERT_BLOCKING_NUM_NSP : {EXPERT_BLOCKING_NUM_NSP}')
print(f'EXPERT_BLOCKING_PACKED_CHUNK_SIZE : {EXPERT_BLOCKING_PACKED_CHUNK_SIZE}')
config = AutoConfig.from_pretrained(model_id)
config.num_hidden_layers = 1
# print(f'model config : {config}')
tokenizer = AutoTokenizer.from_pretrained(model_id)
PREFILL_SEQ_LEN = 512
CTX_LEN = 4096

qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, config=config)
prefill_qpc_path = qeff_model.compile(
    prefill_seq_len=PREFILL_SEQ_LEN,
    ctx_len=CTX_LEN,
    num_cores=16,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    num_devices=1,
    split_retained_state_io=True,
    mos=1,
    aic_enable_depth_first=True,
    num_speculative_tokens=None,
    prefill_only=True,
    enable_chunking=True,
    # stats_level=2,
)
print(f'\nQPC path: {prefill_qpc_path}')
# breakpoint()
tokenizer.padding_side = "right"
inputs = tokenizer(prompt, return_tensors="np", padding=True)
position_ids = inputs["attention_mask"].sum(1, keepdims=True)
padded_len = inputs["input_ids"].shape[1]
num_chunks = -(padded_len // -PREFILL_SEQ_LEN)  # ceil divide without float
padded_len = num_chunks * PREFILL_SEQ_LEN  # Convert to a multiple of prompt_len
inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(padded_len), -1)
inputs.pop("token_type_ids", None)
inputs = {k: torch.from_numpy(v) for k, v in inputs.items()}
inputs.pop("past_key_values", None)
inputs = {k: v.detach().numpy() for k, v in inputs.items()}

prefill_session = QAICInferenceSession(prefill_qpc_path)
full_position_ids = inputs["position_ids"].copy()

total_prefill_start = time.time()
total_time = 0
for i in range(num_chunks):
    chunk_inputs = {
        "input_ids": inputs["input_ids"][:, i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN],
        "position_ids": inputs["position_ids"][:, i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN],
    }
    ins = time.time()
    qpc_out = prefill_session.run(chunk_inputs)
    chunk_time = time.time() - ins
    total_time += chunk_time
    print(f"chunk {i + 1}/{num_chunks} prefill time = {chunk_time} sec")
    for j in range(config.num_hidden_layers):
        inputs[f"past_key.{j}"] = qpc_out[f"past_key.{j}_RetainedState"]
        inputs[f"past_value.{j}"] = qpc_out[f"past_value.{j}_RetainedState"]

total_prefill_time = time.time() - total_prefill_start
prompt_token_count = int((full_position_ids >= 0).sum())
print(f"\ntotal prefill time = {total_prefill_time:.4f} sec")
print(f'total_time : {total_time}')
print(f"prompt tokens = {prompt_token_count}")
print(f"prefill tok/sec = {prompt_token_count / total_prefill_time:.2f}")
print(f"first output token = {tokenizer.decode([np.argmax(qpc_out['logits'])])}")
