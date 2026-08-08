# Risk-Adaptive Verified Speculation (RAVS) v8

## Objective

RAVS accelerates AdaBlock without allowing a learned estimator to change the generated
sequence.  The estimator chooses the size of a speculative verification tree.  AdaBlock's
unchanged confidence-threshold transition verifies every traversed edge.

The design replaces the unsafe implication used by v1--v7,

\[
\text{low predicted risk}\Longrightarrow\text{commit every remaining token},
\]

with

\[
\text{low predicted risk}\Longrightarrow\text{verify more candidate transitions in parallel}.
\]

A wrong risk estimate can lower throughput by allocating an unhelpful tree, but it cannot approve
an unverified token.

## Reference transition

Let \(x\) be an active AdaBlock state and let \(z=f_\theta(x)\) be its logits.  The deterministic
greedy AdaBlock transition is

\[
F(x,z)=x\odot(1-M)+\operatorname{argmax}(z)\odot M,
\]

where \(M\) contains the still-masked positions whose top-1 probability is at least the frozen
transfer threshold.  If that set is empty, \(M\) additionally contains the highest-confidence
masked position.  This is exactly the existing Dream/LLaDA transfer rule at temperature zero.

## Draft graph and verification

From the latest logits RAVS ranks the unresolved positions and constructs a small graph of
hypothetical future active-block states.  A node stores its parent, token state, and speculative
depth.  Candidate children reveal a deterministic number of the parent's highest-ranked proposals.
All nodes are evaluated in one model batch.

Starting at the root, RAVS computes the real AdaBlock successor with the root node's verified
logits.  It traverses a child only when the entire child tensor equals that successor.  If no child
matches, the verified successor is returned.  At a verified leaf, its logits supply one final real
AdaBlock transition, guaranteeing progress.

### Sequence-equivalence invariant

Suppose the input to a RAVS batch equals the state reached by sequential AdaBlock.  Each traversed
edge is accepted only when its child equals \(F\) applied to the verified parent.  The returned state
therefore equals \(F^m(x)\) for some \(m\ge1\).  Induction over batches and blocks proves that RAVS
and AdaBlock finish with identical token sequences.  The proof is independent of estimator quality,
tree depth, and draft rejection rate.

## Risk-adaptive capacity

The existing calibrated local disagreement estimator supplies \(r(x)\in[0,1]\).  It controls only
the maximum linear depth:

\[
K(x)=\begin{cases}
K_{\max}, & r(x)\le q_{\mathrm{deep}},\\
K_{\mathrm{mid}}, & q_{\mathrm{deep}}<r(x)\le q_{\mathrm{mid}},\\
0, & r(x)>q_{\mathrm{mid}}.
\end{cases}
\]

The v8 screen compares conservative, balanced, and deep configurations.  A maximum node cap keeps
GPU memory bounded.  Verification statistics record proposed depth, evaluated nodes, accepted
transitions, rejection depth, equivalent sequential steps, and exactness checks.

## Runtime integration

- `pag.experiments.rc_pag_speculation` owns the model-independent graph, policy, cache batching,
  exact verifier, and serialization.
- The bundled Dream and LLaDA AdaBlock loops call the verifier only for `rc_pag_verified`.
- Native AdaBlock remains the confirmatory reference and uses its existing code path.
- v8 reuses compatible v4--v7 native traces but refits a fresh local
  histogram-gradient-boosting risk head. No old selection, certificate, harm/gain head, or
  remaining-NFE head is reused.
- If verification batching is unsupported or exceeds memory during the pilot, the run fails closed;
  it never silently force-commits or labels a sequential fallback as accelerated.

## Evaluation and gates

The pilot first requires token-for-token equality for both the no-op hook and a depth-4 batched
speculator. The screen then requires:

1. zero sequence disagreements against AdaBlock for both models;
2. at least 5% mean NFE reduction for both models;
3. lower NFE than the best fixed-depth verified speculator on each model; and
4. selection by NFE only after exactness eligibility.

Confirmation uses the existing workshop complement: 500 GSM8K, 200 MATH-500, 100 MBPP, and 64
HumanEval examples per model.  The paper reports accuracy, exact-match rate to AdaBlock, NFE,
latency, peak memory, verification acceptance, and batch-node work.  If the 5% compute gate fails,
the protocol stops before confirmation and reports the verified-speculation frontier rather than
weakening the gate.

## Novelty position

RAVS differs from fixed-width self-speculative decoding by learning state-dependent verification
capacity and by verifying AdaBlock's threshold-parallel, delimiter-adaptive macro-transition.  It
differs from S2D2-style routing because the router cannot select an unverified output.  It differs
from v1--v7 because risk determines verification effort rather than permission to make an
irreversible commitment.
