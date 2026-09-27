/**
 * After callers flush drafts and validate route/connection, send Canvas through async preparation and other
 * sources through reducer submission. Inject Canvas preparation to keep this decision independent of the engine.
 */

import type { GenerateSettings } from '@features/generation/contracts';
import type { ParseDynamicPromptsResponse } from '@features/generation/prompts';
import type { ModelConfig } from '@features/models';
import type { WorkflowGeneratorResolutions, WorkflowPendingGenerator } from '@features/workflow/utility';
import type { AccountScope } from '@platform/state/accountLifecycle';
import type { ResolvedInvocationRoute } from '@workbench/invocationContracts';
import type { Project } from '@workbench/projectContracts';

import { resolveDynamicPrompts } from '@features/generation/prompts';
import {
  getEffectivePrompts,
  hasDynamicPromptSyntax,
  normalizeGenerateSettings,
  sanitizeBatchCount,
} from '@features/generation/settings';
import { getWorkflowBatchCapReason } from '@features/workflow/graph';
import { getInvocationTemplatesSnapshot } from '@features/workflow/react';
import { planWorkflowBatch } from '@features/workflow/utility';
import { queryClient } from '@platform/query/client';
import { isAccountScopeCurrent } from '@platform/state/accountLifecycle';

import type { PrepareCanvasInvocationArgs } from './widgets/canvas/invoke/prepareCanvasInvocation';
import type { WorkbenchCommands } from './workbenchStore';

import { readCanvasCompositingSettings } from './widgets/canvas/invoke/canvasCompositing';
import { readCanvasDenoisingStrength } from './widgets/canvas/invoke/canvasStrength';
import { getProjectWidgetValues } from './widgetState';

/** Plain English, like the shell's other notices — this module has no i18n context. */
const EXPANSION_FAILED_TITLE = 'The prompt could not be expanded';
const GENERATOR_FAILED_TITLE = 'The batch generator could not be resolved';
const BATCH_NOT_READY_TITLE = 'The batch is not ready';

export interface SubmitResolvedInvocationDeps {
  /** The resolved route to submit — the caller has already checked it is valid. */
  route: ResolvedInvocationRoute;
  /** The project the canvas pipeline reads its generate/canvas widget values from. */
  project: Project;
  /** Loaded models (or `undefined` while loading), forwarded verbatim to both paths. */
  models: readonly ModelConfig[] | undefined;
  /** Identity lifetime that initiated this submission. */
  owner: AccountScope;
  commands: Pick<WorkbenchCommands, 'generation' | 'notifications'>;
  /**
   * Resolves after Canvas preparation queues or reports failure; Canvas never dispatches
   * submitResolvedInvocationSnapshot.
   */
  prepareCanvasInvocation: (args: PrepareCanvasInvocationArgs) => Promise<void> | void;
  /** Localizes a control-layer rejection notice; defaults to the English validation sentence. */
  formatControlLayerError?: PrepareCanvasInvocationArgs['formatControlLayerError'];
}

/**
 * Return merged Generate prompts only for dynamic routes; templates may introduce syntax and must consume their
 * own placeholders before expansion.
 */
const getExpandableSettings = (project: Project, route: ResolvedInvocationRoute): GenerateSettings | null => {
  if (route.sourceId === 'upscale' || route.sourceId === 'video' || route.sourceId === 'workflow') {
    return null;
  }

  const settings = normalizeGenerateSettings(getProjectWidgetValues(project, 'generate'));

  if (!settings) {
    return null;
  }

  const effectiveSettings = { ...settings, ...getEffectivePrompts(settings) };

  return hasDynamicPromptSyntax(effectiveSettings.positivePrompt) ? effectiveSettings : null;
};

/**
 * Expand here to avoid a gallery → queue → generation cycle. Treat null or response.error as failure even if
 * prompts are present.
 */
const resolveExpandedPrompts = async (settings: GenerateSettings): Promise<ParseDynamicPromptsResponse | null> => {
  try {
    return await resolveDynamicPrompts(queryClient, {
      combinatorial: settings.dynamicPromptsCombinatorial,
      max_prompts: settings.dynamicPromptsMaxPrompts,
      prompt: settings.positivePrompt,
      seed: settings.dynamicPromptsCombinatorial ? null : settings.dynamicPromptsSampleSeed,
    });
  } catch {
    return null;
  }
};

/** The async generators (dynamic prompts, board listings) a workflow submission still has to resolve. */
const getPendingWorkflowGenerators = (project: Project, route: ResolvedInvocationRoute): WorkflowPendingGenerator[] => {
  const templatesSnapshot = getInvocationTemplatesSnapshot();

  if (route.sourceId !== 'workflow' || templatesSnapshot.status !== 'loaded') {
    return [];
  }

  return planWorkflowBatch(project.projectGraph, templatesSnapshot.templates).pendingGenerators;
};

/**
 * Resolves the pending generators, the only asynchronous step before the reducer; their outputs travel with the
 * action and the planner re-checks them against the document it compiles.
 */
const resolveWorkflowGeneratorsForRoute = async (
  project: Project,
  pending: readonly WorkflowPendingGenerator[],
  owner: AccountScope,
  commands: Pick<WorkbenchCommands, 'notifications'>
): Promise<WorkflowGeneratorResolutions | null> => {
  const templatesSnapshot = getInvocationTemplatesSnapshot();

  if (templatesSnapshot.status !== 'loaded') {
    return null;
  }

  // Loaded on demand: the resolver and its gallery/prompt query dependencies stay out of the boot graph.
  const { resolveWorkflowGenerators } = await import('@features/workflow/generators');
  const { errors, resolutions } = await resolveWorkflowGenerators(queryClient, pending);

  if (!isAccountScopeCurrent(owner)) {
    return null;
  }

  if (errors.length > 0) {
    commands.notifications.add({ kind: 'error', message: errors[0], title: GENERATOR_FAILED_TITLE });
    return null;
  }

  // Sizes were unknown until now: an empty board or a mismatched group only shows once the lists exist.
  const resolvedPlan = planWorkflowBatch(project.projectGraph, templatesSnapshot.templates, {
    generators: resolutions,
  });

  const capReason =
    resolvedPlan.batchSize === null
      ? null
      : getWorkflowBatchCapReason(
          resolvedPlan.batchSize * sanitizeBatchCount(getProjectWidgetValues(project, 'workflow').batchCount)
        );
  const reason = resolvedPlan.reasons[0] ?? capReason;

  if (reason) {
    commands.notifications.add({ kind: 'error', message: reason, title: BATCH_NOT_READY_TITLE });
    return null;
  }

  return resolutions;
};

export const submitResolvedInvocation = async ({
  commands,
  formatControlLayerError,
  models,
  owner,
  prepareCanvasInvocation,
  project,
  route,
}: SubmitResolvedInvocationDeps): Promise<void> => {
  if (!isAccountScopeCurrent(owner)) {
    return;
  }

  // Only a workflow with unresolved generators needs this round trip; every other route dispatches synchronously.
  const pendingGenerators = getPendingWorkflowGenerators(project, route);
  let workflowGenerators: WorkflowGeneratorResolutions | undefined;

  if (pendingGenerators.length > 0) {
    const resolved = await resolveWorkflowGeneratorsForRoute(project, pendingGenerators, owner, commands);

    if (resolved === null) {
      return;
    }

    workflowGenerators = resolved;
  }

  const expandableSettings = getExpandableSettings(project, route);

  // Only dynamic prompts require a round trip.
  if (expandableSettings) {
    const expansion = await resolveExpandedPrompts(expandableSettings);
    if (!isAccountScopeCurrent(owner)) {
      return;
    }

    // Reject failed expansions for hotkey and preview callers too, so literal dynamic syntax never reaches
    // generation.
    if (expansion === null || expansion.error) {
      commands.notifications.add({
        kind: 'error',
        message: expansion?.error ?? undefined,
        title: EXPANSION_FAILED_TITLE,
      });
      return;
    }

    await dispatchResolvedInvocation(
      { commands, formatControlLayerError, models, owner, prepareCanvasInvocation, project, route },
      expansion.prompts.length > 0 ? expansion.prompts : undefined,
      workflowGenerators
    );
    return;
  }

  await dispatchResolvedInvocation(
    { commands, formatControlLayerError, models, owner, prepareCanvasInvocation, project, route },
    undefined,
    workflowGenerators
  );
};

const dispatchResolvedInvocation = async (
  {
    commands,
    formatControlLayerError,
    models,
    owner,
    prepareCanvasInvocation,
    project,
    route,
  }: SubmitResolvedInvocationDeps,
  positivePrompts: string[] | undefined,
  workflowGenerators: WorkflowGeneratorResolutions | undefined
): Promise<void> => {
  if (route.sourceId === 'canvas') {
    // Await Canvas preparation to retain the submission guard until completion; pass the resolved destination so
    // Canvas can output to Gallery.
    await prepareCanvasInvocation({
      compositing: readCanvasCompositingSettings(getProjectWidgetValues(project, 'canvas')),
      destination: route.destination,
      commands,
      formatControlLayerError,
      generateValues: getProjectWidgetValues(project, 'generate'),
      models,
      owner,
      positivePrompts,
      projectId: project.id,
      canvasValues: getProjectWidgetValues(project, 'canvas'),
      projectSettings: project.settings,
      strength: readCanvasDenoisingStrength(getProjectWidgetValues(project, 'canvas')),
    });
    return;
  }

  commands.generation.submitResolved({
    backendSupportsCancellation: true,
    models,
    positivePrompts,
    route,
    ...(workflowGenerators ? { workflowGenerators } : {}),
  });
};
