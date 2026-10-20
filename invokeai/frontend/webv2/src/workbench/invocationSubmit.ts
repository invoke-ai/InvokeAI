/**
 * After callers flush drafts and validate route/connection, send Canvas through async preparation and other
 * sources through reducer submission. Inject Canvas preparation to keep this decision independent of the engine.
 */

import type { GenerateSettings } from '@features/generation/contracts';
import type { ParseDynamicPromptsResponse } from '@features/generation/prompts';
import type { ModelConfig } from '@features/models';
import type { ProjectGraphState } from '@features/workflow/contracts';
import type { WorkflowGeneratorResolutions, WorkflowPendingGenerator } from '@features/workflow/utility';
import type { AccountScope } from '@platform/state/accountLifecycle';
import type { ExpandedPositivePrompts, ResolvedInvocationRoute } from '@workbench/invocationContracts';
import type { Project } from '@workbench/projectContracts';

import { ensureGalleryBoardDestinations } from '@features/gallery/queries';
import { resolveDynamicPrompts } from '@features/generation/prompts';
import {
  getEffectivePrompts,
  hasDynamicPromptSyntax,
  normalizeGenerateSettings,
  sanitizeBatchCount,
} from '@features/generation/settings';
import { normalizeUpscaleWidgetValues } from '@features/upscale';
import { getWorkflowBatchCapReason } from '@features/workflow/graph';
import { getInvocationTemplatesSnapshot } from '@features/workflow/react';
import { planWorkflowBatch } from '@features/workflow/utility';
import { queryClient } from '@platform/query/client';
import { isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';

import type { PrepareCanvasInvocationArgs } from './widgets/canvas/invoke/prepareCanvasInvocation';
import type { WorkbenchCommands } from './workbenchStore';

import { getActiveProjectWorkflow } from './projectWorkflows';
import { readCanvasCompositingSettings } from './widgets/canvas/invoke/canvasCompositing';
import { readCanvasDenoisingStrength } from './widgets/canvas/invoke/canvasStrength';
import { getProjectWidgetValues } from './widgetState';

/** Plain English, like the shell's other notices — this module has no i18n context. */
const EXPANSION_FAILED_TITLE = 'The prompt could not be expanded';
const GENERATOR_FAILED_TITLE = 'The batch generator could not be resolved';
const BATCH_NOT_READY_TITLE = 'The batch is not ready';

interface UpscaleSourceMetadata {
  imageName: string;
  metadata: Record<string, unknown> | null;
}

/** Settled, never rejected: it may fail while the board listing is still awaited, before anything awaits it. */
type UpscaleSourceMetadataRead = UpscaleSourceMetadata | { error: unknown; imageName: string };

/** Upscale in Model only mode copies its input's metadata into the result, so it is read before dispatch. */
const readModelOnlyUpscaleSourceMetadata = (
  project: Project,
  route: ResolvedInvocationRoute,
  owner: AccountScope
): Promise<UpscaleSourceMetadataRead> | null => {
  if (route.sourceId !== 'upscale') {
    return null;
  }

  const values = normalizeUpscaleWidgetValues(getProjectWidgetValues(project, 'upscale'));
  const imageName = values?.mode === 'model-only' ? values.inputImage?.image_name : undefined;

  // Loaded on demand: the reader's gallery transport stays out of the editor's startup graph.
  return imageName
    ? import('./image-actions/modelOnlyUpscale')
        .then(({ readUpscaleSourceMetadata }) => readUpscaleSourceMetadata(imageName, owner.signal))
        .then(
          (metadata) => ({ imageName, metadata }),
          (error: unknown) => ({ error, imageName })
        )
    : null;
};

export interface SubmitResolvedInvocationDeps {
  /** The resolved route to submit — the caller has already checked it is valid. */
  route: ResolvedInvocationRoute;
  /** The project the canvas pipeline reads its generate/canvas widget values from. */
  project: Project;
  /** Loaded models (or `undefined` while loading), forwarded verbatim to both paths. */
  models: readonly ModelConfig[] | undefined;
  /**
   * The project workflow a workflow route submits. Captured with the project snapshot so a switch during
   * generator resolution never submits another workflow; defaults to the snapshot's active workflow.
   */
  workflowId?: string;
  /** Identity lifetime that initiated this submission. */
  owner: AccountScope;
  /** Invoke Front: the backend runs the batch ahead of the user's other pending work. */
  prepend?: boolean;
  commands: Pick<WorkbenchCommands, 'generation' | 'notifications'>;
  /**
   * Resolves after Canvas preparation queues or reports failure; Canvas never dispatches
   * submitResolvedInvocationSnapshot.
   */
  prepareCanvasInvocation: (args: PrepareCanvasInvocationArgs) => Promise<void> | void;
  /** Words a control-layer rejection from the locale; its error message is for logs only. */
  formatControlLayerError: PrepareCanvasInvocationArgs['formatControlLayerError'];
  /** Words, from the locale, a Model only upscale whose input image's metadata could not be read. */
  formatUpscaleMetadataFailure: (imageName: string) => string;
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
const resolveExpandedPrompts = async (
  settings: GenerateSettings,
  sampleSeed: number | null
): Promise<ParseDynamicPromptsResponse | null> => {
  try {
    return await resolveDynamicPrompts(queryClient, {
      combinatorial: settings.dynamicPromptsCombinatorial,
      max_prompts: settings.dynamicPromptsMaxPrompts,
      prompt: settings.positivePrompt,
      seed: sampleSeed,
    });
  } catch {
    return null;
  }
};

const getSubmittedWorkflowDocument = (project: Project, workflowId: string | undefined): ProjectGraphState | null =>
  workflowId
    ? (project.workflows.entries.find((entry) => entry.document.id === workflowId)?.document ?? null)
    : getActiveProjectWorkflow(project).document;

/** The async generators (dynamic prompts, board listings) a workflow submission still has to resolve. */
const getPendingWorkflowGenerators = (
  document: ProjectGraphState | null,
  route: ResolvedInvocationRoute
): WorkflowPendingGenerator[] => {
  const templatesSnapshot = getInvocationTemplatesSnapshot();

  if (!document || route.sourceId !== 'workflow' || templatesSnapshot.status !== 'loaded') {
    return [];
  }

  return planWorkflowBatch(document, templatesSnapshot.templates).pendingGenerators;
};

/**
 * Resolves the pending generators, the only asynchronous step before the reducer; their outputs travel with the
 * action and the planner re-checks them against the document it compiles.
 */
const resolveWorkflowGeneratorsForRoute = async (
  project: Project,
  document: ProjectGraphState,
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
  const resolvedPlan = planWorkflowBatch(document, templatesSnapshot.templates, {
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
  formatUpscaleMetadataFailure,
  models,
  owner,
  prepareCanvasInvocation,
  prepend,
  project,
  route,
  workflowId,
}: SubmitResolvedInvocationDeps): Promise<void> => {
  if (!isAccountScopeCurrent(owner)) {
    return;
  }

  const workflowDocument = route.sourceId === 'workflow' ? getSubmittedWorkflowDocument(project, workflowId) : null;

  if (route.sourceId === 'workflow' && !workflowDocument) {
    return;
  }

  // The reducer resolves where results land (the auto-add board, a workflow node's board) from the cached board
  // listing, so a board is judged from a listing and never trusted unseen. Any cached listing decides at once, a
  // stale one refreshing in the background for the next run; only with none cached does a read start now, alongside
  // generator resolution and prompt expansion, to be awaited as the run is dispatched. So this path is synchronous
  // but for the first run after the listings were cleared, which the Invoke control shows as preparing. A run bound
  // for the canvas names no board, unless it is a workflow, whose nodes carry their boards whatever the destination.
  const pendingBoards =
    route.destination === 'gallery' || route.sourceId === 'workflow'
      ? ensureGalleryBoardDestinations(queryClient)
      : null;
  // Upscale routes have no other asynchronous step, so this is awaited at dispatch, alongside the board listing.
  const pendingUpscaleMetadata = readModelOnlyUpscaleSourceMetadata(project, route, owner);

  // Only a workflow with unresolved generators needs this round trip; every other route dispatches synchronously.
  const pendingGenerators = getPendingWorkflowGenerators(workflowDocument, route);
  let workflowGenerators: WorkflowGeneratorResolutions | undefined;

  if (pendingGenerators.length > 0 && workflowDocument) {
    const resolved = await resolveWorkflowGeneratorsForRoute(
      project,
      workflowDocument,
      pendingGenerators,
      owner,
      commands
    );

    if (resolved === null) {
      return;
    }

    workflowGenerators = resolved;
  }

  const expandableSettings = getExpandableSettings(project, route);

  // Only dynamic prompts require a round trip.
  if (expandableSettings) {
    const sampleSeed = expandableSettings.dynamicPromptsCombinatorial
      ? null
      : expandableSettings.dynamicPromptsSampleSeed;
    const expansion = await resolveExpandedPrompts(expandableSettings, sampleSeed);
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
      {
        commands,
        formatControlLayerError,
        formatUpscaleMetadataFailure,
        models,
        owner,
        prepareCanvasInvocation,
        prepend,
        project,
        route,
        workflowId,
      },
      expansion.prompts.length > 0
        ? { positivePrompts: expansion.prompts, positivePromptsSampleSeed: sampleSeed }
        : undefined,
      workflowGenerators,
      workflowDocument?.id,
      pendingBoards,
      pendingUpscaleMetadata
    );
    return;
  }

  await dispatchResolvedInvocation(
    {
      commands,
      formatControlLayerError,
      formatUpscaleMetadataFailure,
      models,
      owner,
      prepareCanvasInvocation,
      prepend,
      project,
      route,
      workflowId,
    },
    undefined,
    workflowGenerators,
    workflowDocument?.id,
    pendingBoards,
    pendingUpscaleMetadata
  );
};

const dispatchResolvedInvocation = async (
  {
    commands,
    formatControlLayerError,
    formatUpscaleMetadataFailure,
    models,
    owner,
    prepareCanvasInvocation,
    prepend,
    project,
    route,
  }: SubmitResolvedInvocationDeps,
  expansion: ExpandedPositivePrompts | undefined,
  workflowGenerators: WorkflowGeneratorResolutions | undefined,
  workflowId: string | undefined,
  pendingBoards: Promise<void> | null,
  pendingUpscaleMetadata: Promise<UpscaleSourceMetadataRead> | null = null
): Promise<void> => {
  // The board listing the destination is judged from, read while the rest was resolved when none was cached;
  // nothing to wait for keeps the dispatch synchronous.
  if (pendingBoards) {
    await pendingBoards;
    if (!isAccountScopeCurrent(owner)) {
      return;
    }
  }

  let upscaleSourceMetadata: UpscaleSourceMetadata | undefined;

  if (pendingUpscaleMetadata) {
    const read = await pendingUpscaleMetadata;

    if (!isAccountScopeCurrent(owner)) {
      return;
    }
    if ('error' in read) {
      commands.notifications.add({
        kind: 'error',
        message: getApiErrorMessage(read.error, String(read.error)),
        title: formatUpscaleMetadataFailure(read.imageName),
      });
      return;
    }
    upscaleSourceMetadata = read;
  }

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
      expansion,
      ...(prepend ? { prepend } : {}),
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
    ...expansion,
    ...(prepend ? { prepend } : {}),
    projectId: project.id,
    route,
    ...(upscaleSourceMetadata ? { upscaleSourceMetadata } : {}),
    ...(workflowGenerators ? { workflowGenerators } : {}),
    ...(workflowId ? { workflowId } : {}),
  });
};
