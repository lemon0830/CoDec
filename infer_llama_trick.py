#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import sys
from urllib.request import urlopen
import argparse
import torch
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline, GenerationConfig, GenerationMixin
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters

def post_processing(text):
    replace_words = ["#PR_SORG#", "#PRS=ORG#", "#PRS_ORG#_ORG#", "#PRS_MUSIC#", "#PURO_BOX#", "#PRS_SIGNUP#", "#prs_org#",
                     "#PERS_ACC#", "#PRS-ORG#", "##PRS_ORG##", "#PRI_ORG", "PRS _ORG#", "PRS.ORG#", "#PRS _ ORG#", "#PROS#",
                     "#PRS_SIG#", "#PR_SYS#", "#PERSO#", "#PrsOrg#", "ＰＲＳＯＧＥ", "PRS ORG", "#PR_SIG#", "#PRS__ORG#",
                     "#PRS_Org#", "PRS #ORG#"]

    for word in replace_words:
        if word in text:
            text = text.replace(word, "#PRS_ORG#")

    if "#PRS" in text:
        text = text.replace("#PRS", "#PRS_ORG#").replace("#PRS_ORG#_ORG#", "#PRS_ORG#")

    text = text.replace("<v>", "").replace("</v>", "")

    return text

def func(tokenizer, model, text, ifsample=False, draft=None, topk=1):

    text = "Write a response that appropriately completes the request.\n\n" + \
            f"### Request:\n{text}\n\n### Response:"

    if ifsample:
        gen_config = GenerationConfig(temperature=0.1,
                                      top_p=0.7,
                                      do_sample=True,
                                      num_beams=1,
                                      max_new_tokens=256,
                                      eos_token_id=tokenizer.eos_token_id,
                                      pad_token=tokenizer.pad_token_id,
                                      )
    else:
        gen_config = GenerationConfig(do_sample=False,
                                       num_beams=4,
                                       max_new_tokens=256,
                                       no_repeat_ngram_size=8,
                                       pad_token_id=tokenizer.pad_token_id,
                                       eos_token_id=tokenizer.eos_token_id,
                                       num_return_sequences=1,
                                       output_attentions=True
                                      )

    torch.manual_seed(0)

    tokenized = tokenizer(text, padding=True, return_tensors="pt", add_special_tokens=False)

    input_ids = tokenized.input_ids.cuda()
    attn_mask = tokenized.attention_mask.cuda()

    if draft is not None:
        token_draft = tokenizer(draft, padding=True, return_tensors="pt", add_special_tokens=False)
        draft_ids = token_draft.input_ids.cuda()
        draft_attn_mask = token_draft.attention_mask.cuda()

        Len_prompt = input_ids.shape[1]
        draft_input_ids = torch.cat([input_ids, draft_ids], dim=1)
        draft_attn_mask = torch.cat([attn_mask, draft_attn_mask], dim=1)

        draft_outputs = model.model(
            draft_input_ids,
            attention_mask=draft_attn_mask,
            use_cache=True)

        hidden_states = draft_outputs[0]
        lm_logits = model.lm_head(hidden_states)[:, Len_prompt-1:-1, :]

        _, top_ids = torch.topk(lm_logits, k=topk, dim=-1)

        for idx, tidxs in zip(draft_ids[0], top_ids[0]):
            if idx in tidxs:
                Len_prompt += 1
            else:
                break

        input_ids = draft_input_ids[:, :Len_prompt]
        attn_mask = draft_attn_mask[:, :Len_prompt]

    with torch.no_grad():
        try:
            generated_ids = model.generate(inputs=input_ids, attention_mask=attn_mask, generation_config=gen_config)
        except:
            generated_ids = draft_input_ids
        # generated_ids = model.contrastive_search(**tokenized, penalty_alpha=0.6, top_k=4, stopping_criteria=stopping_criteria)

    decoded_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    hyp = decoded_tokens[0].split("### Response:")[-1].replace("\n", " ").strip() #.replace("</s>", "").strip()
    # hyp = "".join(hyp.split("\n"))

    return [hyp]

def init_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=".",  type=str)
    parser.add_argument("-i", "--input_file", help="name of the input file")
    parser.add_argument("-o", "--output_file", help="name of the output file", type=str)
    parser.add_argument("-d", "--draft_file", default=None, help="name of the output file", type=str)

    parser.add_argument("-l","--iflora", default=False, action='store_true')
    parser.add_argument("--src", default="en", type=str)
    parser.add_argument("--tgt", default="zh", type=str)
    parser.add_argument("--ifhint", default=False, action='store_true')
    parser.add_argument("--rootmodel", default=".",  type=str)
    parser.add_argument("--vocab", default=None,  type=str)
    parser.add_argument("--reverse", default=False, action='store_true')
    parser.add_argument("--ifreranking", default=False, action='store_true')
    parser.add_argument("--ifsample", default=False, action='store_true')
    parser.add_argument("-k", "--topk", help="top k of cooperatvie", type=int)


    return parser

LAN_dict = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "fr": "French",
    "cs": "Czech",
    "ja": "Japanese",
    "ru": "Russian",
    "uk": "Ukrainian"
}

cnLAN_dict = {
    "en": "英语",
    "zh": "中文",
    "de": "德语"
}

de_LAN_dict = {'de': "Deutsch", 'en': "Englisch", 'ja': "Japanisch", 'zh': "Chinesisch"}


def get_vocab(file_name, reverse=False):
    zh2en_vocab = {}

    if os.path.exists(file_name):
        f = open(file_name)
        for line in f:
            en, zh = line.strip().split(" ")

            if reverse:
                en, zh = zh, en

            zh2en_vocab[en+"\n"+zh] = 1

        return zh2en_vocab

    return []

prompt_dict = [
        "Translate the following sentences from [SRC] to [TGT].",
        "What do the following sentences mean in [TGT]?",
        "Please provide the [TGT] translation for the following sentences.",
        "Convert the subsequent sentences from [SRC] into [TGT].",
        "Render the listed sentences in [TGT] from their original [SRC] form.",
        "Transform the upcoming sentences from [SRC] language to [TGT] language.",
        "Change the given sentences from [SRC] to [TGT] format.",
        "Turn the following sentences from their [SRC] version to the [TGT] version.",
        "Adapt the mentioned sentences from [SRC] to the [TGT] language.",
        "Transpose the next sentences from the [SRC] format to the [TGT] format.",
        "Switch the specified sentences from their [SRC] form to [TGT] form.",
        "Reinterpret the ensuing sentences from [SRC] to [TGT] language.",
        "Modify the forthcoming sentences, converting them from [SRC] to [TGT].",
        "How can the subsequent sentences be interpreted in [TGT]?",
        "What is the meaning of these sentences when translated to [TGT]?",
        "In the context of [TGT], what do the upcoming sentences signify?",
        "How would you express the meaning of the following sentences in [TGT]?",
        "What is the significance of the mentioned sentences in [TGT]?",
        "In [TGT], what do the given sentences convey?",
        "When translated to [TGT], what message do these sentences carry?",
        "What is the intended meaning of the ensuing sentences in [TGT]?",
        "How should the following sentences be comprehended in [TGT]?",
        "In terms of [TGT], what do the next sentences imply?",
        "Kindly furnish the [TGT] translation of the subsequent sentences.",
        "Could you supply the [TGT] translation for the upcoming sentences?",
        "Please offer the [TGT] rendition for the following statements.",
        "I'd appreciate it if you could present the [TGT] translation for these sentences.",
        "Can you deliver the [TGT] translation for the mentioned sentences?",
        "Please share the [TGT] version of the given sentences.",
        "It would be helpful if you could provide the [TGT] translation of the ensuing sentences.",
        "Kindly submit the [TGT] interpretation for the next sentences.",
        "Please make available the [TGT] translation for the listed sentences.",
        "Can you reveal the [TGT] translation of the forthcoming sentences?"
]

if __name__ == '__main__':
    arg_parser = init_opt()

    args = arg_parser.parse_args()

    modelpath = args.model_path
    rootmodel = args.rootmodel
    infile = args.input_file
    outfile = args.output_file
    ifLoRA = args.iflora
    src = LAN_dict[args.src]
    tgt = LAN_dict[args.tgt]
    ifhint = args.ifhint
    reverse = args.reverse
    ifreranking = args.ifreranking
    ifsample = args.ifsample
    topk = args.topk

    if args.vocab is not None:
        vocab = get_vocab(args.vocab, reverse=reverse)
        print(f"Loading vocab {len(vocab)}")

    tokenizer = AutoTokenizer.from_pretrained(modelpath, cache_dir=modelpath, use_fast=False)
    # tokenizer.padding_side = "left"
    if ifLoRA:
        import torch

        lora_state_dict_path = modelpath + "/lora_pytorch_model.bin"

        model = AutoModelForCausalLM.from_pretrained(rootmodel, cache_dir=rootmodel).half()
        # model = AutoModel.from_pretrained(modelpath, cache_dir=modelpath)

        lora_module_name = "query_key_value".split(",")
        lora_dim = 8
        lora_alpha = 16
        lora_droppout = 0.05

        model = convert_linear_layer_to_lora(model, lora_module_name=lora_module_name, lora_dim=lora_dim,
                                             lora_alpha=lora_alpha, lora_droppout=lora_droppout).cuda()

        state_dict = torch.load(lora_state_dict_path)

        new_state_dict = {}
        for k in state_dict:
            # newk = k.replace("model.", "transformer.")
            newk = "transformer." + k
            print(newk)
            new_state_dict[newk] = state_dict[k]

        model.load_state_dict(new_state_dict, strict=False)

        model = convert_lora_to_linear_layer(model)
        model.eval()
    elif "biatt" in modelpath:
        from utils.llama.modeling_llama import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(modelpath, cache_dir=modelpath).half().cuda()
    else:
        model = AutoModelForCausalLM.from_pretrained(modelpath, cache_dir=modelpath).half().cuda()
        model.eval()

    inf = open(infile, "r")
    outf = open(outfile, "w")

    if args.draft_file:
        draftf = open(args.draft_file, "r")

    NUM=0

    Total_TNUM = 0.
    elapsed_time = 0.

    for line in inf:
        line = line.strip("\n").replace("\\n", "\n")

        if args.draft_file:
            draft = draftf.readline().strip()
        else:
            draft = None

        # print(line, draft)

        # line = f"Translate to {tgt}.\n{line}"
        if args.vocab is not None:
            aligns = {}
            for align in vocab:
                ws, wt = align.split("\n")
                if ws in line:
                    aligns[ws] = wt

            Hint_prompt = ""
            if len(aligns) > 0:
                Hint_prompt = "\n\n###Note:"
                for ws in aligns:
                    Hint_prompt += " " + ws + " means " + aligns[ws] + "."

                # print(line + Hint_prompt)

            line = line + Hint_prompt

        # line = f"Translate from {src} to {tgt}.\n{line}"
        line = f"Please provide the {tgt} translation.\n{line}"

        if ifhint:
            line = line + "\n\n###Note: A translation with no errors could be"

        start = time.time()
        trans = func(tokenizer, model, line, ifsample=args.ifsample, draft=draft, topk=topk)
        end = time.time()

        tok_trans = tokenizer(trans, add_special_tokens=False)
        Total_TNUM += len(tok_trans.input_ids[0])

        elapsed_time += (end-start)

        out_trans = trans[0]

        outf.write(out_trans+"\n")
        NUM += 1
        if NUM % 50 == 0:
            print(f"###Processing {NUM} examples.")
            outf.flush()
            sys.stdout.flush()

    token_per_second = Total_TNUM / elapsed_time
    print(f"耗时：{elapsed_time}, 生成速度：{token_per_second:.2f} token/s")

    inf.close()
    outf.close()




