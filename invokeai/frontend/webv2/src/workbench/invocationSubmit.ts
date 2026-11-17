import type { ParseDynamicPromptsResponse } from '@features/generation/prompts';
import type { ModelConfig } from '@features/models';
import type { ProjectGraphState } from '@features/workflow/contracts';
import type { WorkflowGeneratorResolutions, WorkflowPendingGenerator } from '@features/workflow/utility';
import type { AccountScope } from '@platform/state/accountLifecycle';
import type {
  DynamicPromptsExpansionSource,
  ExpandedPositivePrompts,
  ResolvedInvocationRoute,
  UpscaleSourceMetadata,
} from '@workbench/invocationContracts';
import type { Project } from '@workbench/projectContracts';
import type { ScopedDefaults } from '@workbench/settings/contracts';

import { ensureGalleryBoardDestinations } from '@features/gallery/queries';
import { resolveDynamicPrompts } from '@features/generation/prompts';
import { sanitizeBatchCount } from '@features/generation/settings';
import { normalizeUpscaleWidgetValues } from '@features/upscale';
import { getWorkflowBatchCapReason } from '@features/workflow/graph';
import { getInvocationTemplatesSnapshot } from '@features/workflow/react';
import { planWorkflowBatch } from '@features/workflow/utility';
import { queryClient } from '@platform/query/client';
import { isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';

import type { PrepareCanvasInvocationArgs } from './widgets/canvas/invoke/prepareCanvasInvocation';
import type { WorkbenchCommands } from './workbenchStore';

import { getDynamicPromptsExpansionSource } from './invocation';
import { captureInvocationRuntimeInputs } from './invocationPreparation';
import { getActiveProjectWorkflow } from './projectWorkflows';
import { getEffectiveProjectSettings, PROJECT_SETTINGS_OWNER } from './settings/projectSettings';
import { readCanvasCompositingSettings } from './widgets/canvas/invoke/canvasCompositing';
import { readCanvasDenoisingStrength } from './widgets/canvas/invoke/canvasStrength';
import { getProjectWidgetValues, getResolvedProjectWidgetValues } from './widgetState';

/** Locale keys with their English, which notices keep for a language without them (`resolveNotificationText`). */
const EXPANSION_FAILED_TITLE = { title: 'The prompt could not be expanded', titleKey: 'topbar.invoke.expansionFailed' };
const EXPANSION_UNREACHABLE_MESSAGE = {
  message:
    'The server could not be reached to expand the prompt. Check the connection and try again, or turn dynamic prompts off to use the prompt as typed.',
  messageKey: 'topbar.invoke.expansionUnreachable',
};
const GENERATOR_FAILED_TITLE = {
  title: 'The batch generator could not be resolved',
  titleKey: 'topbar.invoke.generatorFailed',
};
const BATCH_NOT_READY_TITLE = { title: 'The batch is not ready', titleKey: 'topbar.invoke.batchNotReady' };

/** Settled, never rejected: it may fail while the board listing is still awaited, before anything awaits it. */
type UpscaleSourceMetadataRead = UpscaleSourceMetadata | { error: unknown; imageName: string };

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
    ? import('@features/upscale/sourceMetadata')
        .then(({ readUpscaleSourceMetadata }) => readUpscaleSourceMetadata(imageName, owner.signal))
        .then(
          (metadata) => ({ imageName, metadata }),
          (error: unknown) => ({ error, imageName })
        )
    : null;
};

export interface SubmitResolvedInvocationDeps {
  route: ResolvedInvocationRoute;
  project: Project;
  scopedDefaults: ScopedDefaults;
  models: readonly ModelConfig[] | undefined;
  /** Captured with the project snapshot so a switch during generator resolution never submits another workflow. */
  workflowId?: string;
  owner: AccountScope;
  prepend?: boolean;
  commands: Pick<WorkbenchCommands, 'generation' | 'notifications'>;
  prepareCanvasInvocation: (args: PrepareCanvasInvocationArgs) => Promise<void> | void;
  formatControlLayerError: PrepareCanvasInvocationArgs['formatControlLayerError'];
  formatUpscaleMetadataFailure: (imageName: string) => string;
}

/** Expanded here to avoid a gallery → queue → generation cycle; null or response.error is failure. */
const resolveExpandedPrompts = async (
  source: DynamicPromptsExpansionSource,
  sampleSeed: number | null
): Promise<ParseDynamicPromptsResponse | null> => {
  try {
    return await resolveDynamicPrompts(queryClient, {
      combinatorial: source.combinatorial,
      max_prompts: source.maxPrompts,
      prompt: source.positivePrompt,
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
    commands.notifications.add({ ...GENERATOR_FAILED_TITLE, kind: 'error', message: errors[0] });
    return null;
  }

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
    const { message, messageKey, messageValues } = reason.workflowIssue;

    commands.notifications.add({
      ...BATCH_NOT_READY_TITLE,
      kind: 'error',
      message,
      messageKey,
      ...(messageValues ? { messageValues } : {}),
    });
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
  scopedDefaults,
  workflowId,
}: SubmitResolvedInvocationDeps): Promise<void> => {
  if (!isAccountScopeCurrent(owner)) {
    return;
  }

  const workflowDocument = route.sourceId === 'workflow' ? getSubmittedWorkflowDocument(project, workflowId) : null;

  if (route.sourceId === 'workflow' && !workflowDocument) {
    return;
  }

  // Any cached board listing decides at once (a stale one refreshes for the next run); only with none cached does a
  // read start now, awaited at dispatch.
  const pendingBoards =
    route.destination === 'gallery' || route.sourceId === 'workflow'
      ? ensureGalleryBoardDestinations(queryClient)
      : null;
  const pendingUpscaleMetadata = readModelOnlyUpscaleSourceMetadata(project, route, owner);

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

  const expansionSource = getDynamicPromptsExpansionSource(project, route.sourceId);

  if (expansionSource) {
    const { sampleSeed: drawSeed, ...expandedFrom } = expansionSource;
    const sampleSeed = expandedFrom.combinatorial ? null : drawSeed;
    const expansion = await resolveExpandedPrompts(expandedFrom, sampleSeed);
    if (!isAccountScopeCurrent(owner)) {
      return;
    }

    // Reject failed expansions for hotkey and preview callers too, so literal dynamic syntax never reaches generation.
    if (expansion === null || expansion.error) {
      commands.notifications.add({
        ...EXPANSION_FAILED_TITLE,
        ...(expansion === null ? EXPANSION_UNREACHABLE_MESSAGE : { message: expansion.error ?? undefined }),
        kind: 'error',
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
        scopedDefaults,
        workflowId,
      },
      // Sent even when empty: the reducer checks the source against what it compiles either way.
      { expandedFrom, positivePrompts: expansion.prompts, positivePromptsSampleSeed: sampleSeed },
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
      scopedDefaults,
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
    scopedDefaults,
  }: SubmitResolvedInvocationDeps,
  expansion: ExpandedPositivePrompts | undefined,
  workflowGenerators: WorkflowGeneratorResolutions | undefined,
  workflowId: string | undefined,
  pendingBoards: Promise<void> | null,
  pendingUpscaleMetadata: Promise<UpscaleSourceMetadataRead> | null = null
): Promise<void> => {
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
    // Await Canvas preparation so the submission guard holds until it completes.
    const canvasValues = getResolvedProjectWidgetValues(project, 'canvas', scopedDefaults);
    await prepareCanvasInvocation({
      compositing: readCanvasCompositingSettings(canvasValues),
      destination: route.destination,
      commands,
      formatControlLayerError,
      generateValues: getProjectWidgetValues(project, 'generate'),
      models,
      owner,
      expansion,
      ...(prepend ? { prepend } : {}),
      projectId: project.id,
      canvasValues,
      projectSettings: getEffectiveProjectSettings(project.settings, scopedDefaults[PROJECT_SETTINGS_OWNER]),
      strength: readCanvasDenoisingStrength(canvasValues),
    });
    return;
  }

  commands.generation.submitResolved({
    runtime: captureInvocationRuntimeInputs(route.sourceId),
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
