/**
 * The single canvas-vs-generate submit decision, shared by the three Invoke
 * surfaces: the topbar Invoke button (`shell/topbar/InvokeControl`), the Invoke
 * hotkey command (`hotkeys/firstPartyCommands`), and the graph-preview dialog's
 * submit (`graph-preview/GraphPreviewDialog`).
 *
 * Both surfaces flush drafts, resolve the route, and gate on route validity +
 * backend connection themselves; this helper owns only what happens *after* that
 * gate — routing a `canvas` source through the async canvas pipeline
 * (`prepareCanvasInvocation`, honoring the resolved destination) versus
 * dispatching the reducer's `submitResolvedInvocationSnapshot` for every other
 * source. `prepareCanvasInvocation` is injected so the decision stays pure and
 * node-testable with a fake dispatch + a stubbed canvas pipeline.
 */

import type { GenerateSettings } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';
import type { AccountScope } from '@platform/state/accountLifecycle';
import type { ResolvedInvocationRoute } from '@workbench/invocationContracts';
import type { Project } from '@workbench/projectContracts';

import { resolveDynamicPrompts } from '@features/generation/prompts';
import { hasDynamicPromptSyntax, normalizeGenerateSettings } from '@features/generation/settings';
import { queryClient } from '@platform/query/client';
import { isAccountScopeCurrent } from '@platform/state/accountLifecycle';

import type { PrepareCanvasInvocationArgs } from './widgets/canvas/invoke/prepareCanvasInvocation';
import type { WorkbenchCommands } from './workbenchStore';

import { readCanvasCompositingSettings } from './widgets/canvas/invoke/canvasCompositing';
import { readCanvasDenoisingStrength } from './widgets/canvas/invoke/canvasStrength';
import { getProjectWidgetValues } from './widgetState';

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
   * The async canvas-invoke entry point, injected for testability. The real
   * implementation is fire-and-track (its returned promise is intentionally not
   * awaited); a canvas source never dispatches `submitResolvedInvocationSnapshot`.
   */
  prepareCanvasInvocation: (args: PrepareCanvasInvocationArgs) => unknown;
  /** Localizes a control-layer rejection notice; defaults to the English validation sentence. */
  formatControlLayerError?: PrepareCanvasInvocationArgs['formatControlLayerError'];
}

/**
 * The GenerateSettings behind a route that expands `{a|b}`, or `null` when the
 * route submits its prompt literally (Upscale, Workflow) or has no such syntax.
 */
const getExpandableSettings = (project: Project, route: ResolvedInvocationRoute): GenerateSettings | null => {
  if (route.sourceId === 'upscale' || route.sourceId === 'workflow') {
    return null;
  }

  const settings = normalizeGenerateSettings(getProjectWidgetValues(project, 'generate'));

  return settings && hasDynamicPromptSyntax(settings.positivePrompt) ? settings : null;
};

/**
 * Resolving the expansion here rather than inside Queue keeps Queue free of a
 * Generation import, which would close a `gallery -> queue -> generation`
 * dependency cycle. Reading the shared cache means invoking mid-edit still
 * submits the prompts the backend actually produces. A failed expansion falls
 * back to the literal prompt so an Invoke is never silently swallowed.
 */
const resolveExpandedPrompts = async (settings: GenerateSettings): Promise<string[] | undefined> => {
  try {
    const { prompts } = await resolveDynamicPrompts(queryClient, {
      combinatorial: settings.dynamicPromptsCombinatorial,
      max_prompts: settings.dynamicPromptsMaxPrompts,
      prompt: settings.positivePrompt,
      seed: settings.dynamicPromptsCombinatorial ? null : settings.dynamicPromptsSampleSeed,
    });

    return prompts.length > 0 ? prompts : undefined;
  } catch {
    return undefined;
  }
};

export const submitResolvedInvocation = ({
  commands,
  formatControlLayerError,
  models,
  owner,
  prepareCanvasInvocation,
  project,
  route,
}: SubmitResolvedInvocationDeps): void => {
  if (!isAccountScopeCurrent(owner)) {
    return;
  }

  const expandableSettings = getExpandableSettings(project, route);

  // Only a prompt with dynamic syntax pays for a round trip; every other Invoke
  // stays on the synchronous path it has always taken.
  if (expandableSettings) {
    void resolveExpandedPrompts(expandableSettings).then((positivePrompts) => {
      if (isAccountScopeCurrent(owner)) {
        dispatchResolvedInvocation(
          { commands, formatControlLayerError, models, owner, prepareCanvasInvocation, project, route },
          positivePrompts
        );
      }
    });
    return;
  }

  dispatchResolvedInvocation(
    { commands, formatControlLayerError, models, owner, prepareCanvasInvocation, project, route },
    undefined
  );
};

const dispatchResolvedInvocation = (
  {
    commands,
    formatControlLayerError,
    models,
    owner,
    prepareCanvasInvocation,
    project,
    route,
  }: SubmitResolvedInvocationDeps,
  positivePrompts: string[] | undefined
): void => {
  if (route.sourceId === 'canvas') {
    // The canvas graph is composited + compiled asynchronously outside the
    // reducer; fire-and-track (the orchestrator records any failure notice and
    // guards against concurrent invokes internally). The resolved destination is
    // threaded through so a Canvas source can still land its output in the
    // Gallery (durable, non-intermediate) instead of canvas staging.
    prepareCanvasInvocation({
      compositing: readCanvasCompositingSettings(getProjectWidgetValues(project, 'canvas')),
      destination: route.destination,
      commands,
      formatControlLayerError,
      generateValues: getProjectWidgetValues(project, 'generate'),
      models,
      owner,
      positivePrompts,
      projectId: project.id,
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
  });
};
