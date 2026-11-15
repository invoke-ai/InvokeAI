# Evaluating agent guidance

Use a small decision exercise when materially changing agent guidance or investigating a recurring agent mistake. This supplements code checks and review; it is not a mandatory gate for ordinary edits. Cases cover judgment that no check enforces: once a check rejects a mistake, that check is its evaluation and the case should move to the decision the check leaves open. Paths below are relative to the repository root.

## Run a case

1. Choose a case and freeze its acceptance criteria before seeing the response. Start a fresh agent context for each case, with the repository and its normal guidance available, but without earlier conversations, corrections, other case answers, or this evaluator document. Give only the common instruction and selected prompt below. Do not add hints from the rubric.
2. Record the repository commit and relevant uncommitted guidance, model/settings when available, case prompt, response, and any human intervention in `.scratch/agents/<task>/`. Never commit run reports or transcripts. Use the same case, context policy, and model/settings for a comparison; otherwise label the runs incomparable.
3. Assess the response against the evaluator criteria and relevant owner contracts. Record each criterion as met, missed, or inconclusive with supporting evidence. Missing evidence is inconclusive, not success. An executor that read the rubric must be rerun in a fresh context.
4. Report the scope: this read-only exercise evaluates decisions and evidence claims, not implementation correctness, runtime behavior, or a measured productivity improvement. One successful response does not establish reliability. A failure is input to a focused environment correction, not permission to weaken the invariant.

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

Apply to every case: the response cites relevant actual owners, preserves repository constraints, distinguishes proposed checks from executed evidence, and stays within the read-only task. A plausible-sounding but nonexistent API is not a supported entry point.

| Case                 | Required decisions                                                                                                                                                                                                                                                                                                                                                                                                            | Owner guidance                                                                                                                                                                                       |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Derived state        | Compute from the existing items during render; no effect or mirrored state. Existing effects do not authorize an exception, including a wrapper or adapter exemption. Validate updates when the input items change without inventing a performance claim.                                                                                                                                                                     | [React adapters](../invokeai/frontend/webv2/ARCHITECTURE.md#react-adapters)                                                                                                                          |
| Ownership            | Reject the Platform mirror: it passes the import check but puts project state in domain-neutral Platform and adds a process-wide mutable per-project value that can outlive an account or project switch. Trace the existing Queue ports and App composition; inject the capability through its owning contract. Identify an owner test that covers a project switch; a passing architecture check is not ownership evidence. | [Architecture](../invokeai/frontend/webv2/ARCHITECTURE.md), [Queue](../invokeai/frontend/webv2/src/features/queue/README.md)                                                                         |
| Recovery evidence    | Separate visual/mock evidence from actual recovery and retry guarantees. Require a relevant real-browser-storage reload scenario and owning backend persistence/retry evidence for a response lost after a committed write. Do not claim every failure mode is covered or that a screenshot proves durability.                                                                                                                | [Verification map](../invokeai/frontend/webv2/VERIFICATION.md), [Project persistence](../invokeai/frontend/webv2/src/workbench/projects/README.md), [Application backend](../invokeai/app/AGENTS.md) |
| Accelerator evidence | Limit the current claim to exercised CPU configurations. Identify relevant device/dtype/shape correctness and representative peak-memory measurements, distinguishing cold loading from warm execution. Report unavailable devices; follow the family-E2E offer/prerequisite policy without starting generation.                                                                                                              | [Inference guidance](../invokeai/backend/AGENTS.md), [Root family-E2E policy](../AGENTS.md#commands-and-environments)                                                                                |
