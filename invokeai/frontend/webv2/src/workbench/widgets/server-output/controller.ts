import type { WidgetInstanceId } from '@workbench/widgetContracts';

import { createKeyedTransientStore } from '@platform/state/externalStore';
import { createExternalStoreCore } from '@platform/state/externalStoreCore';

import type { ServerOutputReader, ServerOutputRefusalReason } from './reader';

export interface ServerOutputRefusal {
  readonly reason: ServerOutputRefusalReason;
}

export interface ServerOutputAttachment {
  readonly reader: ServerOutputReader;
  /** The selection inside the log as transcript text, or all of it when nothing there is selected. */
  readonly getCopyText: () => string;
}

export interface ServerOutputControllerSnapshot {
  readonly attachment: ServerOutputAttachment | null;
  /** A read the server refused; it stays until the grant is checked again, so its message does not change under it. */
  readonly refusal: ServerOutputRefusal | null;
  /** Counts grant re-checks that cleared a refusal: the reader mounts afresh for each. */
  readonly generation: number;
}

export interface ServerOutputController {
  getSnapshot(): ServerOutputControllerSnapshot;
  subscribe(listener: () => void): () => void;
  /** Called by the mounted panel; returns the detach. */
  attach(attachment: ServerOutputAttachment): () => void;
  refuse(reason: ServerOutputRefusalReason): void;
  /** The grant was checked again and holds: drop the refusal and read afresh. */
  renew(): void;
}

/**
 * What the expanded view, its header label and its header action share for one instance: the mounted reader and why
 * reading stopped. The view creates it and registers it under its instance while mounted.
 */
export const createServerOutputController = (): ServerOutputController => {
  const store = createExternalStoreCore<ServerOutputControllerSnapshot>({
    attachment: null,
    generation: 0,
    refusal: null,
  });

  return {
    attach: (attachment) => {
      store.patchSnapshot({ attachment });
      return () => {
        if (store.getSnapshot().attachment === attachment) {
          store.patchSnapshot({ attachment: null });
        }
      };
    },
    getSnapshot: store.getSnapshot,
    refuse: (reason) => {
      if (store.getSnapshot().refusal === null) {
        store.patchSnapshot({ refusal: { reason } });
      }
    },
    renew: () => {
      const { generation, refusal } = store.getSnapshot();
      if (refusal !== null) {
        store.patchSnapshot({ generation: generation + 1, refusal: null });
      }
    },
    subscribe: store.subscribe,
  };
};

/**
 * Header slots render apart from the view and know only the instance ID. Holds only mounted views' controllers: a
 * view removes its own on unmount, which an account change or hidden Activity also causes.
 */
export const serverOutputControllers = createKeyedTransientStore<WidgetInstanceId, ServerOutputController>();
