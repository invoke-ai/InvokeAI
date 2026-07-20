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

import type { ModelConfig } from '@features/models';
import type { ResolvedInvocationRoute } from '@workbench/invocationContracts';
import type { Project } from '@workbench/projectContracts';

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
  commands: Pick<WorkbenchCommands, 'generation' | 'notifications'>;
  /**
   * The async canvas-invoke entry point, injected for testability. The real
   * implementation is fire-and-track (its returned promise is intentionally not
   * awaited); a canvas source never dispatches `submitResolvedInvocationSnapshot`.
   */
  prepareCanvasInvocation: (args: PrepareCanvasInvocationArgs) => unknown;
}

export const submitResolvedInvocation = ({
  commands,
  models,
  prepareCanvasInvocation,
  project,
  route,
}: SubmitResolvedInvocationDeps): void => {
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
      generateValues: getProjectWidgetValues(project, 'generate'),
      models,
      projectId: project.id,
      projectSettings: project.settings,
      strength: readCanvasDenoisingStrength(getProjectWidgetValues(project, 'canvas')),
    });
    return;
  }

  commands.generation.submitResolved({
    backendSupportsCancellation: true,
    models,
    route,
  });
};
