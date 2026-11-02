# tiny-gpt-armenian

A 10.8M-parameter GPT, written from scratch in PyTorch and trained on Armenian Wikipedia in under 8 minutes on a laptop GPU.

It is a character-level, decoder-only transformer. It sees one task, predicting the next character, and learns Armenian spelling, grammar and an encyclopedic tone from that alone. Two heavily commented Python files, no pretrained weights.

This is the code behind the YouTube video **"I Built a Tiny GPT From Scratch That Speaks Armenian"** on the **FromHardware** channel: AI built from scratch, on hardware you own.

## What it writes

Prompt `Հայաստանը`, same seed, at different training steps (verbatim from [`results/samples/`](results/samples), newlines shown as spaces, `…` marks cuts):

| Step | Output (first characters) |
|---|---|
| 0 | `Հայաստանը2քէեէ:թթԺԳ2 Ղգ-2թգծԳԸխքԳ խթ:խծ…` |
| 300 | `Հայաստանը այլ տեղբանական անդրապուն, որ հականագրապատվում է…` |
| 1,000 | `Հայաստանը շատ տեղի է կազմում, պահպանելով կարագրավորվում է «Նատանության» վրա։` |
| 5,000 | `Հայաստանը և այլն։ Հայկական լեռնաշխարհի բնակչությունը … գերազանցում է 300-1000 մմ։` |

Perfect grammar, complete nonsense: the population "exceeds 300-1000 mm". A language model learns what text looks like, not what is true.

## Results (run 2)

| | |
|---|---|
| Parameters | 10,779,264 (6 layers, 6 heads, 384 dims, 256-character context) |
| Data | 8,333 of 15,634 streamed articles kept, 50,031,803 characters (91.7 MB) |
| Vocabulary | 111 characters |
| Training | 5,000 steps × 64 × 256 = 81.9M characters seen |
| Time | 7.9 minutes on an RTX 4060 Laptop GPU (8 GB), bf16 |
| Validation loss | 4.87 → 1.16 |

Full logs: [`results/loss.csv`](results/loss.csv), [`results/run.json`](results/run.json), [`results/stats.json`](results/stats.json). [`results/attention.npz`](results/attention.npz) holds the attention weights (6 layers × 6 heads) for the sentence `Երևանը Հայաստանի մայրաքաղաքն է։` ("Yerevan is the capital of Armenia.").

## Run it

```bash
pip install -r requirements.txt     # pick the torch build that matches your CUDA driver
python prepare_data.py              # streams Armenian Wikipedia, ~50M characters → data/
python train.py                     # 5,000 steps → out/
```

- `prepare_data.py` streams `wikimedia/wikipedia` (`20231101.hy`) from Hugging Face, keeps lines that are at least 95% Armenian letters, builds the character vocabulary and writes `data/train.bin`, `data/val.bin`, `meta.json`, `stats.json`.
- `train.py` trains the model and writes `out/loss.csv`, `out/samples/step_*.txt`, `out/attention.npz`, `out/run.json` and `out/ckpt.pt`.
- Out of memory? `python train.py --batch-size 32`. On Linux, `--compile` makes training about 1.5–2× faster after a slower first step.
- Another language: change the snapshot (`python prepare_data.py --snapshot 20231101.<code>`) and adapt the allowed-character set at the top of `prepare_data.py`.

Every option is listed with `--help`.

## How it works

```
characters → ids (111-symbol vocabulary)
→ token embedding + position embedding          (384 numbers each)
→ 6 × [LayerNorm → causal self-attention → +residual
       LayerNorm → MLP 384→1536→384 (GELU) → +residual]
→ LayerNorm → linear head (tied to the token embedding) → next-character probabilities
```

Training: AdamW, learning rate 1e-3 → 1e-4 with 100 warmup steps and cosine decay, dropout 0.2, gradient clipping at 1.0.

## Credits

- The approach follows Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) (MIT).
- Training text: [Armenian Wikipedia](https://hy.wikipedia.org), CC BY-SA 4.0, via the Hugging Face dataset [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia). The dataset is not included in this repository; `prepare_data.py` downloads it. The sample outputs in `results/samples/` are model-generated text.

## License

MIT. See [LICENSE](LICENSE).
