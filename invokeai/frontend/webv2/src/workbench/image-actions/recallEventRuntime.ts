import type { ModelConfig } from '@features/models';
import type { AccountScope } from '@platform/state/accountLifecycle';
import type { SocketHub } from '@platform/transport/socketHub';
import type { WidgetRegion } from '@workbench/layoutContracts';
import type { RegisteredWidget, WidgetTypeId } from '@workbench/widgetContracts';
import type { WorkbenchCommands, WorkbenchQueries } from '@workbench/workbenchStore';

import { ensureModelsLoaded, getModelsSnapshot } from '@features/models';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { getWorkspaceOffer, openInWorkspace } from '@workbench/workspace/workspaceNavigation';

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
 * observes an earlier one's changes (an append after a replace). Buffered replay events run first. Each event waits for
 * the model catalog and is dropped when it belongs to another user or outlives the account it arrived under.
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

        // The models store never rejects; a failed catalog fetch is recorded as its error status, which would otherwise
        // read as "no model selected".
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

export interface RecallRevealContext {
  getWidgetsForRegion: (region: WidgetRegion) => RegisteredWidget[];
  /** Whether the user is typing in a text field; a reveal must not switch tabs out from under them. */
  isEditingText: () => boolean;
}

/**
 * The commands a recall applies through. Once it writes the widget's values (Generate's settings, a widget patch), each
 * notice it records offers the way to that widget when this workspace cannot show it: a socket event never navigates,
 * so the recall's own notice says what changed and where. Notices before any write, such as "nothing applied", stay
 * plain.
 */
export const withRecallWidgetOffer = <
  Commands extends Pick<WorkbenchCommands, 'layout' | 'notifications' | 'widgets'> &
    Partial<Pick<WorkbenchCommands, 'generation'>>,
>(
  commands: Commands,
  {
    projectId,
    queries,
    reveal,
    typeId,
  }: {
    projectId: string;
    queries: Pick<WorkbenchQueries, 'getSnapshot'>;
    reveal: RecallRevealContext;
    typeId: WidgetTypeId;
  }
): Commands => {
  let hasWritten = false;
  const written =
    <Args extends unknown[]>(write: (...args: Args) => void) =>
    (...args: Args): void => {
      hasWritten = true;
      write(...args);
    };
  const { generation } = commands;

  return {
    ...commands,
    ...(generation
      ? {
          generation: {
            ...generation,
            patchSettings: written(generation.patchSettings),
            setSettings: written(generation.setSettings),
          },
        }
      : {}),
    notifications: {
      ...commands.notifications,
      add: (notice: Parameters<WorkbenchCommands['notifications']['add']>[0]) => {
        const offer =
          hasWritten && queries.getSnapshot().activeProject.id === projectId
            ? getWorkspaceOffer({ commands, getWidgetsForRegion: reveal.getWidgetsForRegion, queries }, { typeId })
            : [];

        commands.notifications.add(
          offer.length > 0 ? { ...notice, actions: [...(notice.actions ?? []), ...offer] } : notice
        );
      },
    },
    widgets: { ...commands.widgets, patchValues: written(commands.widgets.patchValues) },
  };
};

/**
 * Bring the widget a recall just changed to the front where this workspace places it, so the caller sees the result.
 * A socket event is no gesture: it never enters another workspace, adds a widget or moves focus, and leaves the offer
 * to its notice (`withRecallWidgetOffer`). Nothing moves while the user types or once the project is off screen.
 */
export const bringRecallWidgetToFront = ({
  commands,
  owner,
  projectId,
  queries,
  reveal,
  typeId,
}: {
  commands: Pick<WorkbenchCommands, 'layout' | 'notifications' | 'widgets'>;
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

  openInWorkspace(
    { commands, getWidgetsForRegion: reveal.getWidgetsForRegion, queries },
    { typeId },
    { background: true }
  );
};
