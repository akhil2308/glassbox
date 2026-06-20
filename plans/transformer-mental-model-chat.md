# Understanding Transformers — a walkthrough chat

A conversation building up a mental model of how a transformer (GPT-2-small) generates
text, framed for someone who thinks in data flow / middleware pipelines. Covers: the
residual stream, the logit lens, the generation loop, the KV cache, the input matrix,
embeddings, and which row predicts the next word.

---

## 1. The core analogy: a request through a middleware chain

A transformer literally is a middleware pipeline.

Picture a FastAPI/Express request going through a stack of middleware:

```
ctx → [mw 1] → [mw 2] → [mw 3] → ... → [mw 12] → response
```

There's one shared context object (`ctx`) that flows straight through. Each middleware
reads `ctx`, computes something, and adds its contribution back onto it. It never
replaces it — it does `ctx += my_delta`. Then the next stage sees the enriched version.

That shared object flowing through is the **residual stream**. The middleware stages are
the model's **layers** (GPT-2-small has 12). That's the whole skeleton.

| Transformer thing | Middleware mental model |
|---|---|
| Residual stream | The `ctx` object passed down the chain |
| A layer | One middleware stage: `ctx += f(ctx)` |
| The token's vector | The current state/payload inside `ctx` |
| Unembedding (final step) | The handler that turns `ctx` into the actual response (next word) |
| Logit lens | A debug probe you `console.log` between every middleware |

### What the logit lens actually is

Normally you only see the response at the very end of the chain. The logit lens inserts a
probe after every middleware stage:

> "If we had to respond right now with whatever's in `ctx` at this point — what word
> would come out?"

You're not changing anything. You're tapping the pipe at each stage and decoding the
half-finished `ctx` early. That's why you can watch the answer get assembled:

```
hydrogen and → [grammar] then → therefore → hydrogen → OXYGEN
                 mw1..7          mw8         mw9-10      mw11
```

Early middleware has only done generic grammar work (`ctx` says "probably a connector
word like *then*"). By stage 9 the topic is in there (*hydrogen*). By stage 11 the actual
answer (*oxygen*) has been written in. The answer wasn't sitting there waiting — it got
assembled stage by stage, and the probe let you watch it appear.

### The one piece that breaks the pure-pipeline picture

In a normal request pipeline, requests are independent — request A doesn't read request
B's `ctx`. In a transformer, each token is its own parallel pipeline, but at certain
stages tokens are allowed to peek at each other's `ctx`. That cross-talk is **attention**
("the city of ___" peeks back at "Eiffel Tower" to know what it's about). Think of it as
a `Promise.all` step where parallel requests get to share state before continuing.

### A correction on the clean-climb intuition

People say early layers give generic guesses and later layers sharpen confidence. True
for the *guess*, but the **confidence often drops at the very last layer** (e.g. oxygen
63% at L11 → 48% at final). That's normal — the last stage spreads probability across
many plausible words. The lens shows you the journey, not a clean monotonic climb.

---

## 2. Generation is a loop — one full pass per word

Each new token is a full fresh pass through all 12 layers.

```
"Water is made of hydrogen and"
  → 12 layers → predicts "oxygen"

"Water is made of hydrogen and oxygen"
  → 12 layers → predicts "which"

"Water is made of hydrogen and oxygen which"
  → 12 layers → predicts "combines"

... and so on
```

The stopping condition is one of:
- The model outputs a special `<|endoftext|>` token (it learned when to stop in training)
- You hit a max token limit you set
- You pass a custom stop string (e.g. `\n`)

So the logit lens is a snapshot of **one single token's journey**. That same process
repeats for every word the model generates.

---

## 3. The KV cache — what "reprocessing" means

### The key fact: the model has no memory between words

When the model generates a word, it then **forgets everything**. To generate the next
word, you hand it the *entire sentence so far* as a brand-new input, like it's seeing it
for the first time.

Watch what gets fed in each time:

```
Pass 1:  feed in  [Water] [is] [made] [of] [hydrogen] [and]
                  → outputs "oxygen"

Pass 2:  feed in  [Water] [is] [made] [of] [hydrogen] [and] [oxygen]
                  → outputs "which"

Pass 3:  feed in  [Water] [is] [made] [of] [hydrogen] [and] [oxygen] [which]
                  → outputs "combines"
```

### The waste

In Pass 1, the model did a bunch of math on the first 6 words. In Pass 2, you feed those
**same 6 words in again** — so it does the **exact same math on them again**. Same input
→ same output. Pure repeated work. Then it does the math for the one genuinely new word,
`[oxygen]`. That redoing of the first 6 words is **"reprocessing."**

### What the cache does

The math the model does on each word produces a result (its **K** and **V** — Keys and
Values, its "contribution to the conversation"). That result **only depends on that word
and the words before it** — never on words that come *later*. So word 3's result is
identical in Pass 1, Pass 2, Pass 3... forever. It can never change.

So why recompute it? **Save it the first time. Reuse it.**

```
Pass 2 WITHOUT cache:  compute Water, is, made, of, hydrogen, and, oxygen   ← 7 words of math
Pass 2 WITH cache:     [reuse 6 saved results] + compute only oxygen        ← 1 word of math
```

The cache is just a drawer of saved results for words you've already seen, so each new
step only does math for **the single new word**.

The thing that makes this valid: a word's result can *never* depend on future words (the
model only ever looks backward). That guarantees the saved result stays valid forever.

---

## 4. The input as a matrix (rows = words)

A good instinct: the input is a **matrix where each row is a word**. It flows up through
12 layers, each layer's result **added back in place**, and at the top a new word is
produced.

But there are **two different "adds"** at two different scales — easy to merge by mistake:

**Add #1 — inside the pipeline (the residual add).** Shape never changes. 6 words in →
6 words out, every layer.

```
6×768 matrix  → [layer 1: result added back] → still 6×768
              → [layer 2: result added back] → still 6×768
              ... 12 times ...
              → 6×768 matrix
```

No new row here. The 12 layers **transform the rows in place**. Same shape in, same shape
out.

**Add #2 — between generations (the new word).** *This* is where a new row is born:

```
After all 12 layers, take the LAST row only  (the "and" position)
  → multiply by the unembedding matrix
  → get a word: "oxygen"
  → append "oxygen" as a new row

Now the sentence is 7 rows → feed back in → do the whole 12-layer pass again
```

| | what happens | shape effect |
|---|---|---|
| **Inside a pass** | result added back to ctx, ×12 layers | rows stay same count |
| **Between passes** | last row → one new word → new row | matrix grows by 1 row |

### Correction to "multiply by a fixed matrix"

In most of a layer, yes — each row gets multiplied by fixed learned matrices
independently. **But at the attention step the rows are not independent** — they look at
each other (the `Promise.all` cross-talk). That part isn't a fixed matrix; it's computed
*from the rows themselves* each time. That's the only spot where rows talk.

---

## 5. Which row predicts? And what is "layer 0"?

### Which row holds the prediction?

It's the **bottom row** (the last / most-recent word), not the top. Rows are just the
sentence in reading order:

```
row 1:  Water
row 2:  is
row 3:  made
row 4:  of
row 5:  hydrogen
row 6:  and      ← we read the prediction off THIS row (the last one)
```

**Why the last row?** Each row can only see itself and the rows above it (backward-looking
rule). So only the **last row has seen the whole sentence** — its final value (after 12
layers) is the one that holds enough information to guess the next word. The top row only
ever saw "Water"; it can't predict much.

### What is "layer 0" (the thing that builds the starting matrix)?

This is the **embedding**. Something has to *create* the starting matrix before the 12
layers can chew on it — but the surprise is:

**It's not computed. It's a lookup.**

The model has a giant dictionary — every word in its vocabulary (~50,000 words) has one
**stored row of numbers** (768 of them), learned during training. "Layer 0" just looks
each word up and stacks the rows:

```
"Water"     → dictionary → copy out its stored row  → row 1
"is"        → dictionary → copy out its stored row  → row 2
"hydrogen"  → dictionary → copy out its stored row  → row 5
...
```

No math, no thinking. *"hydrogen" always maps to this fixed list of 768 numbers.* Same
every time.

(One extra: it also adds a "position signal" to each row so the model knows row 5 came
before row 6 — otherwise the matrix would be an unordered bag of words.)

Full picture:

```
words → [layer 0: look up each word's stored row]  → starting matrix
      → [12 layers refine it in place]              → final matrix
      → read the LAST row → produces the next word
```

---

## 6. Next topic to tackle: attention

Left out deliberately so far. Attention's whole job is answering one question:

> "When the last row is trying to predict, how does it pull in info from the earlier rows?"

That's the entire job. (To be expanded next.)

---

## Appendix: the script that produced the logit-lens runs

`scripts/run_logit_lens.py` now takes a `--prompt` flag so you can poke at any sentence
from the terminal without editing the file:

```
uv run python scripts/run_logit_lens.py
uv run python scripts/run_logit_lens.py --prompt "Water is made of hydrogen and"
uv run python scripts/run_logit_lens.py --prompt "2 plus 2 equals" --prompt "The capital of France is"
```

No `--prompt` args = falls back to the built-in prompt set.
