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
import type { ParseDynamicPromptsResponse } from '@features/generation/prompts';
import type { ModelConfig } from '@features/models';
import type { AccountScope } from '@platform/state/accountLifecycle';
import type { ResolvedInvocationRoute } from '@workbench/invocationContracts';
import type { Project } from '@workbench/projectContracts';

import { resolveDynamicPrompts } from '@features/generation/prompts';
import { getEffectivePrompts, hasDynamicPromptSyntax, normalizeGenerateSettings } from '@features/generation/settings';
import { queryClient } from '@platform/query/client';
import { isAccountScopeCurrent } from '@platform/state/accountLifecycle';

import type { PrepareCanvasInvocationArgs } from './widgets/canvas/invoke/prepareCanvasInvocation';
import type { WorkbenchCommands } from './workbenchStore';

import { readCanvasOutputOnlyMaskedRegions } from './widgets/canvas/canvasSettings';
import { readCanvasCompositingSettings } from './widgets/canvas/invoke/canvasCompositing';
import { readCanvasDenoisingStrength } from './widgets/canvas/invoke/canvasStrength';
import { getProjectWidgetValues } from './widgetState';

/** Plain English, like the shell's other notices — this module has no i18n context. */
const EXPANSION_FAILED_TITLE = 'The prompt could not be expanded';

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
 *
 * The returned settings carry the *merged* prompts, so both the gate and the
 * expansion below see what will actually generate. That matters in both
 * directions: an active template can introduce `{a|b}` a plain authored prompt
 * never had, and it always consumes its own `{prompt}` placeholder before the
 * expander could mistake it for a one-option group.
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
 * Resolving the expansion here rather than inside Queue keeps Queue free of a
 * Generation import, which would close a `gallery -> queue -> generation`
 * dependency cycle. Reading the shared cache means invoking mid-edit still
 * submits the prompts the backend actually produces.
 *
 * `null` means the request itself failed. A response may still carry an `error`
 * alongside a usable-looking `prompts` — the route is deliberately soft so the
 * preview can show the message, but the submit path must not act on it.
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
    void resolveExpandedPrompts(expandableSettings).then((expansion) => {
      if (!isAccountScopeCurrent(owner)) {
        return;
      }

      // Submitting the authored text would put the literal `{a|b}` or `__name__`
      // in front of the model, so a prompt that could not be expanded does not
      // generate at all. The Invoke button gates on the same state; this covers
      // the hotkey and graph-preview paths, which never see it.
      if (expansion === null || expansion.error) {
        commands.notifications.add({
          kind: 'error',
          message: expansion?.error ?? undefined,
          title: EXPANSION_FAILED_TITLE,
        });
        return;
      }

      dispatchResolvedInvocation(
        { commands, formatControlLayerError, models, owner, prepareCanvasInvocation, project, route },
        expansion.prompts.length > 0 ? expansion.prompts : undefined
      );
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
      outputOnlyMaskedRegions: readCanvasOutputOnlyMaskedRegions(getProjectWidgetValues(project, 'canvas')),
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
