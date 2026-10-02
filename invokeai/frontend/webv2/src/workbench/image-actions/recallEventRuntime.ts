import type { ModelConfig } from '@features/models';
import type { AccountScope } from '@platform/state/accountLifecycle';
import type { SocketHub } from '@platform/transport/socketHub';
import type { WidgetRegion } from '@workbench/layoutContracts';
import type { RegisteredWidget, WidgetTypeId } from '@workbench/widgetContracts';
import type { WorkbenchCommands, WorkbenchQueries } from '@workbench/workbenchStore';

import { ensureModelsLoaded, getModelsSnapshot } from '@features/models';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { openWidgetPlacement } from '@workbench/widgetPlacementCommands';

export interface RecallRuntime {
  dispose(): void;
}

/** A raw socket payload with the project that was active when it arrived. */
export interface PendingRecallEvent {
  payload: unknown;
  projectId: string;
}

const toErrorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

/**
 * Apply one kind of external recall event to its arrival-time project, strictly in arrival order so a later event
 * observes an earlier one's changes (an append after a replace). Buffered replay events run first. Each event waits
 * for the model catalog and is dropped when it belongs to another user or outlives the account it arrived under.
 */
export const createRecallEventRuntime = <Event extends { user_id: string }>({
  apply,
  area,
  commands,
  eventName,
  getSessionUserId = () => null,
  hub,
  isEvent,
  queries,
  replay = [],
}: {
  apply: (event: Event, context: { models: ModelConfig[]; owner: AccountScope; projectId: string }) => Promise<unknown>;
  /** The notification area failures are reported under. */
  area: string;
  commands: Pick<WorkbenchCommands, 'notifications'>;
  eventName: string;
  /**
   * The signed-in user in multi-user mode, or `null` to accept every event. Admin sockets also receive other users'
   * image recall events, which must not rewrite the admin's own panels.
   */
  getSessionUserId?: () => string | null;
  hub: Pick<SocketHub, 'on'>;
  isEvent: (payload: unknown) => payload is Event;
  queries: Pick<WorkbenchQueries, 'getSnapshot'>;
  replay?: readonly PendingRecallEvent[];
}): RecallRuntime => {
  let disposed = false;
  let chain: Promise<void> = Promise.resolve();

  const enqueue = ({ payload, projectId }: PendingRecallEvent) => {
    if (disposed || !isEvent(payload)) {
      return;
    }

    const sessionUserId = getSessionUserId();
    if (sessionUserId !== null && payload.user_id !== sessionUserId) {
      return;
    }

    const owner = captureAccountScope();
    const reportError = (error: unknown) => {
      if (!disposed && isAccountScopeCurrent(owner)) {
        commands.notifications.reportError({
          area,
          message: toErrorMessage(error),
          namespace: 'generation',
          projectId,
        });
      }
    };

    chain = chain
      .then(async () => {
        if (disposed || !isAccountScopeCurrent(owner)) {
          return;
        }

        // The models store never rejects; a failed catalog fetch is recorded as
        // its error status, which would otherwise read as "no model selected".
        await ensureModelsLoaded();
        if (disposed || !isAccountScopeCurrent(owner)) {
          return;
        }

        const snapshot = getModelsSnapshot();
        if (snapshot.status === 'error') {
          reportError(snapshot.error ?? 'Failed to load models.');
          return;
        }

        await apply(payload, { models: snapshot.models, owner, projectId });
      })
      // One failing event must not wedge the chain for every later one.
      .catch(reportError);
  };

  for (const pending of replay) {
    enqueue(pending);
  }

  const detach = hub.on(eventName, (payload: unknown) => {
    enqueue({ payload, projectId: queries.getSnapshot().activeProject.id });
  });

  return {
    dispose: () => {
      if (disposed) {
        return;
      }
      disposed = true;
      detach();
    },
  };
};

/** What bringing a recall's widget to the front needs from the app. */
export interface RecallRevealContext {
  getWidgetsForRegion: (region: WidgetRegion) => RegisteredWidget[];
  /** Whether the user is typing in a text field; a reveal must not switch tabs out from under them. */
  isEditingText: () => boolean;
}

const PANEL_OPEN_KEYS = { bottom: 'isBottomOpen', left: 'isLeftOpen', right: 'isRightOpen' } as const;

/**
 * Bring the widget a recall just changed to the front of its project, so the caller sees the result: select its
 * tab (expanding its region and opening its panel), raise (and unshade) it when it floats, or open it when the layout
 * has none. A
 * floating window stays floating. Nothing moves while the user is typing, or when the project is no longer the one
 * on screen; the notification reports the change either way.
 */
export const bringRecallWidgetToFront = ({
  commands,
  owner,
  projectId,
  queries,
  reveal,
  typeId,
}: {
  commands: Pick<WorkbenchCommands, 'widgets'>;
  owner: AccountScope;
  projectId: string;
  queries: Pick<WorkbenchQueries, 'getSnapshot'>;
  reveal: RecallRevealContext;
  typeId: WidgetTypeId;
}): void => {
  const project = queries.getSnapshot().activeProject;

  if (!isAccountScopeCurrent(owner) || project.id !== projectId || reveal.isEditingText()) {
    return;
  }

  const isTarget = (instanceId: string | null | undefined) =>
    instanceId !== null && instanceId !== undefined && project.widgetInstances[instanceId]?.typeId === typeId;
  const floating = Object.keys(project.floatingWidgets ?? {}).find(isTarget);

  if (floating) {
    // A shaded window is rolled up to its title bar; raising it alone would still hide the change.
    if (project.floatingWidgets?.[floating]?.mode === 'shaded') {
      commands.widgets.setFloatingMode(floating, 'windowed');
    }
    commands.widgets.focusFloating(floating);
    return;
  }

  const placed = (
    Object.entries(project.widgetRegions) as [WidgetRegion, (typeof project.widgetRegions)[WidgetRegion]][]
  ).filter(([, region]) => region.instanceIds.some(isTarget));
  const isShown = ([region, state]: (typeof placed)[number]) =>
    isTarget(state.activeInstanceId) &&
    !state.isCollapsed &&
    (region === 'center' || project.layout.panels[PANEL_OPEN_KEYS[region]]);

  if (placed.some(isShown)) {
    return;
  }
  if (placed.length > 0) {
    // Opening a type its region already holds activates that instance, expands the region and opens its panel.
    commands.widgets.open({ projectId, region: placed[0]![0], widgetId: typeId });
    return;
  }

  openWidgetPlacement({
    getWidgetsForRegion: reveal.getWidgetsForRegion,
    options: { preferredRegions: ['left'] },
    projectId,
    typeId,
    widgets: commands.widgets,
  });
};
