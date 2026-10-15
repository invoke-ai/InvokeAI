import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

/** The settings entry the manager lives in; entry points reveal it so focus lands in the manager. */
export const INTERMEDIATES_SETTING_ID = 'intermediatesManager';

/**
 * What an entry point wants the manager to start on: a project to preselect, or an account to filter by. The manager
 * reads it while rendering and consumes it once that render commits, so a discarded render cannot lose it and a
 * stale intent never resurfaces on a later visit.
 */
export interface IntermediatesFocus {
  projectId?: string;
  ownerId?: string;
  /** How to name `ownerId` before any of its rows load. */
  ownerLabel?: string;
}

/** `request` counts requests, so a manager that is already showing can restart on a new one. */
const focusStore = createExternalStore<{ focus: IntermediatesFocus | null; request: number }>({
  focus: null,
  request: 0,
});

registerAccountOwnedResource({
  clear: () => focusStore.setSnapshot({ focus: null, request: 0 }),
  name: 'intermediates-focus',
});

export const requestIntermediatesFocus = (focus: IntermediatesFocus): void => {
  focusStore.setSnapshot({ focus, request: focusStore.getSnapshot().request + 1 });
};

export const useIntermediatesFocusRequest = (): number => focusStore.useSelector((snapshot) => snapshot.request);

export const peekIntermediatesFocus = (): IntermediatesFocus | null => focusStore.getSnapshot().focus;

/** Clears `focus` unless a newer request has replaced it. */
export const consumeIntermediatesFocus = (focus: IntermediatesFocus | null): void => {
  if (focus !== null && focusStore.getSnapshot().focus === focus) {
    focusStore.patchSnapshot({ focus: null });
  }
};
