import type { SocketHub } from '@platform/transport/socketHub';
import type { WorkbenchCommands, WorkbenchQueries } from '@workbench/workbenchStore';
import type { TFunction } from 'i18next';

import { captureGalleryNavigation } from '@features/gallery/contracts';
import { createLogger } from '@platform/logging/logger';
import { captureAccountScope } from '@platform/state/accountLifecycle';
import { getProjectWidgetValues } from '@workbench/widgetState';

import type { PendingRecallEvent, RecallRevealContext, RecallRuntime } from './recallEventRuntime';
import type { createRecallParametersRuntime } from './recallParametersRuntime';
import type { ExternalGenerationRevealContext, revealExternalGeneration } from './revealExternalGeneration';
import type { createVideoRecallRuntime } from './videoRecallRuntime';

/*
 * Image-action socket bridges kept in the editor boot graph: each loads the runtime it attaches on demand.
 */

/**
 * Lazy-load a recall runtime on its first event; buffer events in arrival order with their original project. Report
 * load failures and retry on the next event. This module stays in the editor boot graph, so it imports the runtimes
 * by type only.
 */
const attachLazyRecallRuntime = <Module>({
  area,
  commands,
  create,
  eventName,
  hub,
  load,
  queries,
}: {
  area: string;
  commands: Pick<WorkbenchCommands, 'notifications'>;
  create: (module: Module, replay: PendingRecallEvent[]) => RecallRuntime;
  eventName: string;
  hub: Pick<SocketHub, 'on'>;
  load: () => Promise<Module>;
  queries: Pick<WorkbenchQueries, 'getSnapshot'>;
}): RecallRuntime => {
  let disposed = false;
  let loading: Promise<void> | null = null;
  let runtime: RecallRuntime | null = null;
  const pending: PendingRecallEvent[] = [];

  let detachBuffer: (() => void) | null = hub.on(eventName, (payload: unknown) => {
    pending.push({ payload, projectId: queries.getSnapshot().activeProject.id });
    loading ??= load()
      .then((module) => {
        if (disposed) {
          return;
        }

        // Hand-over and live subscription share one tick, so no event can
        // slip between the buffered ones and the runtime's own listener.
        detachBuffer?.();
        detachBuffer = null;
        runtime = create(module, pending.splice(0));
      })
      .catch((error: unknown) => {
        loading = null;
        pending.length = 0;

        if (!disposed) {
          commands.notifications.reportError({
            area,
            message: error instanceof Error ? error.message : String(error),
            namespace: 'generation',
            projectId: queries.getSnapshot().activeProject.id,
          });
        }
      });
  });

  return {
    dispose: () => {
      if (disposed) {
        return;
      }
      disposed = true;
      pending.length = 0;
      detachBuffer?.();
      detachBuffer = null;
      runtime?.dispose();
      runtime = null;
    },
  };
};

interface RecallParametersRuntimeModule {
  createRecallParametersRuntime: typeof createRecallParametersRuntime;
}

/** `recall_parameters_updated` → the Generate panel. */
export const attachRecallParametersRuntime = ({
  commands,
  getSessionUserId,
  hub,
  load = () => import('./recallParametersRuntime'),
  queries,
  reveal,
  t,
}: {
  commands: Pick<WorkbenchCommands, 'generation' | 'notifications' | 'widgets'>;
  getSessionUserId?: () => string | null;
  hub: Pick<SocketHub, 'on'>;
  load?: () => Promise<RecallParametersRuntimeModule>;
  queries: Pick<WorkbenchQueries, 'getProject' | 'getSnapshot'>;
  reveal: RecallRevealContext;
  t: TFunction;
}): RecallRuntime =>
  attachLazyRecallRuntime({
    area: 'recall-parameters',
    commands,
    create: (module, replay) =>
      module.createRecallParametersRuntime({ commands, getSessionUserId, hub, queries, replay, reveal, t }),
    eventName: 'recall_parameters_updated',
    hub,
    load,
    queries,
  });

interface VideoRecallRuntimeModule {
  createVideoRecallRuntime: typeof createVideoRecallRuntime;
}

/** `video_recall_requested` → the Video panel. */
export const attachVideoRecallRuntime = ({
  commands,
  getSessionUserId,
  hub,
  load = () => import('./videoRecallRuntime'),
  queries,
  reveal,
  t,
}: {
  commands: Pick<WorkbenchCommands, 'notifications' | 'widgets'>;
  getSessionUserId?: () => string | null;
  hub: Pick<SocketHub, 'on'>;
  load?: () => Promise<VideoRecallRuntimeModule>;
  queries: Pick<WorkbenchQueries, 'getProject' | 'getSnapshot'>;
  reveal: RecallRevealContext;
  t: TFunction;
}): RecallRuntime =>
  attachLazyRecallRuntime({
    area: 'video-recall',
    commands,
    create: (module, replay) =>
      module.createVideoRecallRuntime({ commands, getSessionUserId, hub, queries, replay, reveal, t }),
    eventName: 'video_recall_requested',
    hub,
    load,
    queries,
  });

interface ExternalGenerationRevealModule {
  revealExternalGeneration: typeof revealExternalGeneration;
}

export interface ExternalGenerationRevealer {
  dispose(): void;
  /** Reveal a backend queue item another client completed with the reveal destination. */
  request(queueItemId: number): void;
}

/**
 * Another client's completed reveal request → the Gallery. Capture the arrival state the reveal is fenced by
 * (account, project, Gallery navigation) before the reveal chunk loads, and let only the newest request reveal: each
 * arrival aborts the one before it. A request a hidden tab deferred waits, alone, until the tab is shown.
 */
export const attachExternalGenerationReveal = ({
  context,
  load = () => import('./revealExternalGeneration'),
}: {
  context: ExternalGenerationRevealContext;
  load?: () => Promise<ExternalGenerationRevealModule>;
}): ExternalGenerationRevealer => {
  type Request = Parameters<typeof revealExternalGeneration>[1];
  let disposed = false;
  let latest: AbortController | null = null;
  let deferred: { controller: AbortController; request: Request; sequence: number } | null = null;

  const run = (controller: AbortController, request: Request, sequence?: number): void => {
    void load()
      .then(({ revealExternalGeneration }) => revealExternalGeneration(context, request, sequence))
      .then((outcome) => {
        if (outcome.kind === 'hidden' && latest === controller && !disposed) {
          deferred = { controller, request, sequence: outcome.sequence };
          // The tab may have been shown again while this run was settling.
          resume();
        }
      })
      .catch((error: unknown) => {
        // Superseded and disposed reveals fail with their abort; nothing on screen asked for either.
        if (!controller.signal.aborted) {
          createLogger(
            { area: 'external-generation-reveal', namespace: 'gallery', projectId: request.projectId },
            { owner: request.accountScope }
          ).warn({
            context: { queueItemId: request.queueItemId },
            error,
            message: 'Could not reveal an external generation.',
          });
        }
      });
  };

  const resume = (): void => {
    if (!deferred || !context.isVisible()) {
      return;
    }

    const { controller, request, sequence } = deferred;

    deferred = null;
    run(controller, request, sequence);
  };

  const detachVisibility = context.onVisibilityChange(resume);

  return {
    dispose: () => {
      disposed = true;
      detachVisibility();
      deferred = null;
      latest?.abort();
      latest = null;
    },
    request: (queueItemId) => {
      if (disposed) {
        return;
      }

      latest?.abort();
      deferred = null;
      const controller = new AbortController();
      const project = context.queries.getSnapshot().activeProject;

      latest = controller;
      run(controller, {
        accountScope: captureAccountScope(),
        navigation: captureGalleryNavigation(getProjectWidgetValues(project, 'gallery')),
        projectId: project.id,
        queueItemId,
        signal: controller.signal,
      });
    },
  };
};
