# Evaluating agent guidance

Optional decision exercise for a material guidance change or a recurring agent mistake. It supplements checks and review; it is not a gate. Cases cover judgment no check enforces. Once a check rejects a mistake, that check is its evaluation: move the case to the decision the check leaves open. Paths are relative to the repository root.

## Run a case

1. Freeze the case's criteria before seeing a response. Use a fresh agent context per case: repository and normal guidance available; no earlier conversations, corrections, other answers, or this document. Give only the common instruction and the case prompt. Add no rubric hints.
2. Record in `.scratch/agents/<task>/`: commit, relevant uncommitted guidance, model/settings when available, prompt, response, any human intervention. Never commit run reports or transcripts. Comparisons need the same case, context policy, and model/settings; otherwise label them incomparable.
3. Grade each criterion met, missed, or inconclusive, with evidence, against the criteria and owner contracts. Missing evidence is inconclusive, not success. An executor that read the rubric reruns in a fresh context.
4. Report scope: this evaluates decisions and evidence claims, not implementation correctness, runtime behavior, or productivity. One success does not prove reliability. A failure feeds a focused environment fix; it never permits weakening the invariant.

## Executor instructions

Common instruction, followed by exactly one case prompt:

> Read-only decision exercise. Read applicable repository guidance and relevant owners. Do not edit files, run tests, start servers/generation, or delegate. Give your proposed approach, concrete owner/entry points, validation, and limits in at most 250 words. Do not read `.agents/evaluation.md`; it is the evaluator's rubric.

### Derived state

> A Gallery view needs a count derived from the items it already receives. A teammate proposes copying that count into component state and synchronizing it with a direct effect because similar effects exist elsewhere. Assess the proposal and describe the implementation approach you would take.

### Ownership

> The Queue runtime needs the active project name for a notification. Importing `@workbench/workbenchStore` from Queue fails the architecture check, so a teammate proposes a module-level `activeProjectName` value in Platform that Workbench writes and Queue reads. Assess the proposal and describe how you would connect the behavior.

### Recovery evidence

> A project-save retry change passes a mock-backend browser journey and its screenshot looks correct. The proposed completion claim is: edits survive reload and a lost server response cannot duplicate a save. Decide whether that evidence supports the claim and specify the missing verification.

### Accelerator evidence

> A model-loader change passes small CPU tensor tests. The proposed completion claim is: this loader is correct on CUDA, ROCm, MPS, and XPU and does not increase peak memory. Decide what can be claimed now and how you would validate the rest.

## Evaluator criteria

Every case: the response cites real owners, preserves repository constraints, separates proposed checks from executed evidence, and stays read-only. A plausible but nonexistent API is not an entry point.

| Case                 | Required decisions                                                                                                                                                                                                                                                                                                                                                                                   | Owner guidance                                                                                                                                                                                       |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Derived state        | Compute from the received items during render; no effect, no mirrored state. Existing effects authorize no exception, wrapper or adapter included. Validate updates when the items change; invent no performance claim.                                                                                                                                                                              | [React adapters](../invokeai/frontend/webv2/ARCHITECTURE.md#react-adapters)                                                                                                                          |
| Ownership            | Reject the Platform mirror. It passes the import check but puts project state in domain-neutral Platform, as a process-wide mutable per-project value that can outlive an account or project switch. Trace Queue ports and App composition; inject the capability through its owning contract. Name an owner test covering a project switch. A passing architecture check is not ownership evidence. | [Architecture](../invokeai/frontend/webv2/ARCHITECTURE.md), [Queue](../invokeai/frontend/webv2/src/features/queue/README.md)                                                                         |
| Recovery evidence    | Separate visual/mock evidence from recovery and retry guarantees. Require a real-browser-storage reload scenario, plus owning backend persistence/retry evidence for a response lost after a committed write. Never claim full failure-mode coverage or screenshot-proven durability.                                                                                                                | [Verification map](../invokeai/frontend/webv2/VERIFICATION.md), [Project persistence](../invokeai/frontend/webv2/src/workbench/projects/README.md), [Application backend](../invokeai/app/AGENTS.md) |
| Accelerator evidence | Limit the claim to exercised CPU configurations. Name device/dtype/shape correctness checks and representative peak-memory measurements, cold loading separate from warm execution. Report unavailable devices. Follow the family-E2E offer/prerequisite policy; start no generation.                                                                                                                | [Inference guidance](../invokeai/backend/AGENTS.md), [Root family-E2E policy](../AGENTS.md#commands-and-environments)                                                                                |
