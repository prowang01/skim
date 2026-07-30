# Skim — The Story Behind the Project

A personal account of what was built, why each decision was made, and what I learned. This is not the
README (that describes the current state for a reader). This is the journey: the trade-offs, the dead
ends, and the concepts I picked up along the way.

---

## 1. The pitch

I built Skim to solve a problem I kept hitting myself. I wanted to hand a video to an AI and talk about
it: summarize it, find a specific moment, ask what was said. Some tools are starting to do this. Gemini
and GPT can read short videos now, and Claude a little less so, but that support is recent, partial, and
not available everywhere I work. At my internship, the tool we use (Dust) does not accept `.mp4` files
at all, and models cannot open blocked YouTube links. Frontier LLMs read text and images, not raw video.

My first goal was modest: I just wanted a way to feed videos into an AI so I could prompt it better.
Along the way the goal shifted into something more valuable to me. It stopped being about shipping yet
another video-chat tool, since those are appearing anyway, and became about understanding, hands-on,
how you actually wire a model around a real need. The tool works, but the real result is that I can now
explain every part of it and why it is there. That is what this document captures, so I can retell the
project clearly and never lose the reasoning behind any choice.

In one line: Skim turns a video into something an LLM can query. Concretely, it is a local Streamlit app
where you upload a video and chat with it. You ask something like "how many eggs does she use?" and you
get back "four eggs" with the exact timestamp it was said or shown, or an honest "that is not in the
video" when the answer really is not there. Under the hood it splits the video into a timestamped
transcript and sampled visual frames, indexes those pieces, and retrieves only the relevant ones to
answer.

---

## 2. Architecture: the 4-station pipeline

The whole project makes sense once you hold one picture: a pipe with four stations. The mnemonic is
Split, Store, Search, Answer.

1. **Ingestion (split).** The video is broken into two streams. The audio becomes a timestamped
   transcript (faster-whisper, running locally), and the picture becomes a set of key frames (selected
   by ffmpeg scene-detection, described in words by GPT-4o vision). The output is many small pieces
   called chunks (a chunk is one small unit of content: a transcript segment or an image description).
   This step is deterministic per video, and it is the slow part.
2. **Index (store).** Each chunk is turned into an embedding, a vector that captures its meaning, and
   stored so it can be searched by meaning rather than by keyword. This is the library.
3. **Retrieval (search).** For a given question, the system fetches only the relevant chunks instead of
   dumping the whole video into the model. This is the R in RAG, the librarian.
4. **Answer (generate).** The LLM (gpt-4o-mini) answers from those chunks, cites timestamps, and
   declines when the information is absent. This is the AG in RAG, Augmented Generation.

Stations 2, 3, and 4 are literally a RAG system. The twist is that the knowledge base is not a set of
documents but a video, taken apart into audio and images. That is the whole idea: RAG applied to video.

---

## 3. How it was built: the levels

I built the project in stages, and each stage ended with something that worked. That was deliberate. I
never wanted to be stuck with a half-broken thing wondering whether any of it ran. Each level below adds
one capability, and together they build up the four stations one piece at a time.

**Level 1, audio.** Extract the audio, transcribe it, and chat about what was said. Validated on a short
TED talk: correct answers, correct timestamps, and an honest "not in the video" on an off-topic question.

**Level 2, vision.** Added the visual half. Scene-detected frames, described by GPT-4o. The system now
understood what was shown, not only what was said. It became multimodal.

**Level 3, retrieval and fusion.** Until here I dumped everything into the model. Now I added the Index
and Retrieval stations (embeddings, fetch only the good chunks). Fusion means that because audio and
image chunks share a timeline, a question can draw on both at once: the transcript might say "add the
sauce" while a frame taken at the same moment shows what is in the pan, and the answer can use the two
together.

**Level 4, maturity.** Two things that turn a prototype into a serious project. First, honesty about
blind spots: the system says "I cannot judge motion from still frames" instead of inventing. Second,
evals: measuring answer quality against a naive baseline that just dumps everything into the model.

**Level 5, retrieval hardening.** The richest chapter, and the whole of the next section.

---

## 4. Key decisions and trade-offs

This is the heart of the project. Almost every decision cost something: you gain one thing and give up
another, and the interesting question is always why you chose that particular trade. The clearest ones
are below, each tied to the concrete cost it carried.

**Scene-detection frames, not one frame per second.** I sample frames when the scene changes, capped at
about 25, instead of grabbing a frame every second. The cost of that choice showed up later. A recipe
video flashed the word "Bacon" on screen for a couple of seconds, between two scenes, and my sampler
never caught it, so the system guessed "minced meat" from the footage. That was the trade made visible.
But the trade was worth it: I keep cost bounded and I can handle long videos. I chose control over
completeness, and I wrote the gap down rather than hide it.

**Retrieval instead of dumping everything.** Rather than send the whole transcript and all frames to the
model on every question, I retrieve only the relevant chunks. The trade is more machinery (an index,
embeddings, a retriever) against lower cost and tighter focus. The lesson I discovered here reframed the
whole project: retrieval only helps when the content is too big to fit in the model's context at once.
On short videos, dumping everything works just as well.

**Adaptive top-k, then a relative cap.** Top-k is simply how many chunks I fetch for a question. A fixed
value of 6 missed answers on a 32-minute podcast of about 900 chunks. I made that number scale with the
size of the video, and the long-video score jumped. Then I noticed the opposite failure. On a 3-minute
video of about 90 chunks, the same rule was now pulling back roughly 80% of the video. At that point
retrieval is no longer selective, it is just burning tokens to hand the model almost everything. So I
added a ceiling: never fetch more than about a third of the chunks. Adaptive top-k fixes recall on long
content, and the cap fixes waste on short content. I validated both on item counts, a deterministic
signal, and confirmed the cap never even triggers on the podcast.

**Retrieve-then-rerank, built, tested, and deliberately turned off.** I added a reranker. The idea is to
retrieve a wide set of candidates with the fast embedding search (a bi-encoder, which sizes up the
question and each chunk separately), then re-score them with a slower, sharper model (a cross-encoder,
which reads the question and the chunk together and judges how well they match). On paper this catches
cases the bi-encoder misses. On my dataset it made scores slightly worse. So I kept the code as an
opt-in flag, disabled by default, with a note explaining why: on a small, clean dataset the extra
machinery does not pay, and it earns its place on larger, noisier corpora. Rejecting your own addition
because the numbers say so is a real engineering call.

**Adjacent-context retrieval.** When I retrieve a chunk, I also pull its neighbors in time. This fixed a
case the reranker could not. The passage that announces an answer and the passage that contains it are
sometimes different chunks. Announce a thing, and now you drag in the thing itself. The trade is that
pulling too many neighbors adds noise, so I keep the window small. This closed the last gap on the
podcast with no regressions elsewhere.

**LLM-as-judge, then calibrating it.** The evals are graded by an LLM acting as a judge. But an LLM
judge is non-deterministic. One borderline question kept flipping between "correct" and "partial" across
runs with no code change at all. I tightened the judge with an explicit rubric: exactly what counts as
correct, partial, or wrong, graded against the central fact rather than the exact wording. That steadied
it, but it introduced a side effect. An "honesty" clause over-triggered and mis-graded a decline-to-
answer as correct on a knowable fact. So I bounded that clause to only apply when the expected answer
itself asserts that something is absent. The lesson stuck: when you tune a judge, you are chasing
consistency, not leniency. A judge that inflates scores voids the whole eval.

**Caching ingestion to stabilize measurement.** Ingestion (transcript, frames, vision) is deterministic
per video but slow. I cache it, keyed by the video file, with a `--no-cache` override for when I change
the ingestion logic. There was a side benefit I did not expect. Whisper hallucinates slightly different
noise on near-silent audio each run, and caching froze that, because a cache hit returns a byte-identical
transcript. My experiments got cleaner because I had accidentally removed a source of random variance.
The underlying lesson: when you test one thing, freeze everything else, so the only thing moving is the
thing you are studying.

---

## 5. The three retrieval failure modes

Debugging retrieval on the long podcast surfaced three genuinely different reasons it can miss the right
passage, each with a different fix. Being able to tell them apart is the point, because "retrieval did
not find it" is really three separate problems wearing the same coat.

**One, the net was too small.** The right passage was findable, but I was not fetching enough chunks to
reach it. This is pure sizing, and it is fixed by scaling how many chunks I fetch.

**Two, the words do not match.** The answer is phrased with words that barely overlap the question. An
analogy about "tying a balloon to my wrist and never letting go" versus a question about "holding on too
tightly." They mean the same thing but share almost no vocabulary, so a similarity search ranks the
answer far down the list. This is where a cross-encoder, which weighs meaning rather than surface words,
can help in principle.

**Three, the answer is not where the topic is.** The most on-topic passage announces the answer, while
the answer's actual content sits a few chunks later. No single-passage scorer bridges that gap. The fix
is not a smarter ranker at all, it is reaching for the neighboring chunks.

The honest headline: after all of this, retrieval landed essentially tied with the naive
dump-everything baseline, a razor-thin edge on one long-video question that is well within noise. That
is not a disappointment, it is the lesson. Retrieval's advantage only appears when the content exceeds
the model's context window. On anything short, the simple approach is just as good. Knowing when an
architecture is justified is as valuable as knowing how to build it.

---

## 6. Concepts I learned

**RAG (Retrieval-Augmented Generation).** Fetch the relevant pieces of a knowledge base, then let the
model answer using them. Grounded answers instead of guesses.

**Chunk.** One small unit of content, a transcript segment or an image description, indexed and
retrieved as a single piece.

**Embedding.** A piece of text turned into numbers that capture its meaning, so things that mean the
same land near each other. It is what lets you search by sense instead of by keyword.

**Bi-encoder vs. cross-encoder.** A bi-encoder sizes up the question and each chunk separately, then
compares them. It is fast but coarse. A cross-encoder reads the question and chunk together and judges
the match. It is slow but sharp. The standard move is to cast a wide net with the bi-encoder and then
re-rank the finalists with the cross-encoder, so each does the job it is good at.

**Top-k.** How many chunks the retriever fetches for a given question. Too few and you miss the answer,
too many and you drown the model in noise and cost.

**LLM-as-judge.** Using an LLM to grade answers in an eval. It is powerful but wobbles from run to run,
so it needs an explicit rubric, and you tune it for consistency, not leniency.

**Deterministic vs. noisy signals.** Item counts and code proofs do not change between runs, so trust
them to validate a change. LLM-judge scores drift, so read them as a trend, not to the decimal. When in
doubt, validate on the signal that cannot wobble.

**Scene-detection frame sampling.** Choosing frames by how much the picture changes, instead of grabbing
every frame. Cheaper, but it can skip brief static moments, which is exactly what caused the bacon bug.

**VAD (voice activity detection).** Detecting where speech actually is, so you do not transcribe silence.
It is the real cure for Whisper inventing words over quiet audio.

---

## 7. Future work

Each of these is a door that a specific problem in the build opened, so I know where it comes from and
where it goes.

**Silence-proof the transcription.** The bug is concrete: on a near-silent cooking video, Whisper
invented small scraps of speech that were never said. I contained it (caching froze the noise, and the
judge stopped rewarding dodges), but the cause is still there. The next step is to detect where speech
actually is with voice activity detection, or to drop low-confidence segments, so the transcript never
contains ghosts in the first place.

**Tune retrieval per kind of video.** I learned that short and long videos want opposite things. One
needs a wider net, the other needs a ceiling. Right now I handle that with one blunt rule. I would like
the system to sense what kind of video it is looking at and adjust how much it pulls and how densely it
samples frames. This comes straight out of the short-versus-long fight.

**Add keyword search alongside meaning search (hybrid).** The "words do not match" failure mode has a
mirror image. Sometimes the exact word is what matters, a name, a number, a technical term, and a
meaning-based search glides right past it. Pairing the embedding search with a classic keyword search
(BM25) would cover both cases, which is why serious RAG systems run both.

**Ingest YouTube links directly.** Half of my original pain was that getting a video in was a chore.
Pulling a YouTube transcript straight from its link, with no download and no file juggling, would remove
that friction and bring the project back to the exact itch that started it.

**Make the judge vote.** The last sliver of eval variance lives on one genuinely subjective question. I
could have the judge grade it several times and take the majority, which is steadier at the cost of
triple the judge calls. I left it as a known, bounded bit of noise rather than pay that price, but it is
the obvious lever if the noise ever starts to matter.

**Give it a proper interface.** The current UI is functional, not beautiful. Chat on the right,
transcript top-left, frames bottom-left would show off what the tool can do. It is cosmetic, but a demo
lives or dies on it.

**Retry the reranker where it belongs.** It did not pay off on my small, clean dataset, but I only
proved it does not help here, not that it never helps. On a large, noisy corpus, retrieve-then-rerank is
supposed to shine. That is an experiment worth running on the terrain it was built for.

---

One honest note. I built this by directing an AI coding agent, decision by decision, rather than typing
every line by hand. What is mine is the architecture, every trade-off above, and the diagnoses. The
whole point was to understand the system deeply enough to stand behind any part of it.
