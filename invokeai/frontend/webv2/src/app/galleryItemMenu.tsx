import type { GalleryItem, GalleryItemRef } from '@features/gallery/contracts';
import type { GalleryItemContextMenuRequest, GalleryItemContextMenuTarget } from '@features/gallery/react';
import type { ComponentType } from 'react';

import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { createExternalStoreCore } from '@platform/state/externalStoreCore';
import { useSyncExternalStore } from 'react';

export interface GalleryItemMenuView {
  id: number;
  target: GalleryItemContextMenuTarget;
}

export interface GalleryItemMenuHostProps {
  generateValues: Record<string, unknown>;
  projectId: string;
  view: GalleryItemMenuView | null;
  /** A new view mounts a new menu, so a late close from the one it replaced names that one. */
  onClose(viewId: number): void;
}

export interface LoadedGalleryItemMenu {
  Host: ComponentType<GalleryItemMenuHostProps>;
  item: GalleryItem;
}

interface GalleryItemMenuSnapshot {
  /** Kept once loaded: a deletion confirmation the menu opened outlives the menu. */
  Host: ComponentType<GalleryItemMenuHostProps> | null;
  view: GalleryItemMenuView | null;
}

export type GalleryItemMenu = ReturnType<typeof createGalleryItemMenu>;

export interface GalleryItemMenuOpenOptions {
  /** Checked when the menu is ready: false (the project changed, say) drops the request. */
  isCurrent: () => boolean;
  onFailure: (error: unknown) => void;
}

/**
 * One item menu for media shown outside the Gallery widget; only the newest request in this account may open it. A
 * request still loading is dropped by the user's next click or Escape, so a menu never opens over what they did next.
 */
export const createGalleryItemMenu = ({
  interactions,
  load,
}: {
  /** Where the user's next click or Escape is heard; the window unless given. */
  interactions?: Pick<EventTarget, 'addEventListener'>;
  /** Fetches the item only when `item` is missing. */
  load: (ref: GalleryItemRef, item: GalleryItem | undefined, signal: AbortSignal) => Promise<LoadedGalleryItemMenu>;
}) => {
  const store = createExternalStoreCore<GalleryItemMenuSnapshot>({ Host: null, view: null });
  let pending: AbortController | null = null;
  let nextId = 1;
  const cancelPending = () => {
    pending?.abort();
    pending = null;
  };

  return {
    /** Closes `viewId` only: a menu dismissed by the right-click that opened the next one leaves that one be. */
    close: (viewId: number) => {
      const { view } = store.getSnapshot();

      if (view?.id !== viewId) {
        return;
      }
      // The menu opened at a point, with no trigger to take focus back; an action that moved focus on keeps it there.
      if (document.activeElement?.closest('[role="menu"]')) {
        view.target.returnFocus?.()?.focus({ preventScroll: true });
      }
      store.patchSnapshot({ view: null });
    },
    dispose: cancelPending,
    getSnapshot: store.getSnapshot,
    open: (request: GalleryItemContextMenuRequest, { isCurrent, onFailure }: GalleryItemMenuOpenOptions) => {
      cancelPending();
      const controller = new AbortController();
      const scope = captureAccountScope();
      const isLive = () => pending === controller && isAccountScopeCurrent(scope);
      const listenerOptions = { capture: true, signal: controller.signal };
      const target = interactions ?? window;
      pending = controller;
      // Registered after the pointerdown that opened this request, so only a later one cancels it.
      target.addEventListener('pointerdown', cancelPending, listenerOptions);
      target.addEventListener(
        'keydown',
        (event) => {
          if ((event as KeyboardEvent).key === 'Escape') {
            cancelPending();
          }
        },
        listenerOptions
      );

      load(request.ref, request.item, AbortSignal.any([controller.signal, scope.signal])).then(
        ({ Host, item }) => {
          if (!isLive()) {
            return;
          }
          cancelPending();
          if (!isCurrent()) {
            return;
          }
          store.setSnapshot({
            Host,
            view: {
              id: nextId++,
              target: {
                itemRefs: [request.ref],
                items: [item],
                returnFocus: request.returnFocus,
                x: request.x,
                y: request.y,
              },
            },
          });
        },
        (error: unknown) => {
          if (isLive()) {
            cancelPending();
            if (isCurrent()) {
              onFailure(error);
            }
          }
        }
      );
    },
    subscribe: store.subscribe,
  };
};

export const GalleryItemMenuSlot = ({
  generateValues,
  menu,
  projectId,
}: {
  generateValues: Record<string, unknown>;
  menu: GalleryItemMenu;
  projectId: string;
}) => {
  const { Host, view } = useSyncExternalStore(menu.subscribe, menu.getSnapshot, menu.getSnapshot);

  return Host ? <Host generateValues={generateValues} projectId={projectId} view={view} onClose={menu.close} /> : null;
};
