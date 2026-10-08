import type { GalleryImageItem, GalleryItem, GalleryItemKey, GalleryItemRef } from '@features/gallery/contracts';
import type { GallerySettings } from '@features/gallery/core/settings';
import type {
  GalleryBoard,
  GalleryBoardDeletionResult,
  GalleryImage,
  GalleryProjectRef,
  GalleryView,
} from '@features/gallery/core/types';
import type { QueueProgressSession } from '@features/queue/contracts';

import { createContext, use, useMemo, type ComponentType, type ReactNode } from 'react';

export interface GalleryItemActions {
  /**
   * `returnFocus` names where keyboard focus goes when a confirmation dialog closes, resolved then: the deletion can
   * remove the control that opened it.
   */
  deleteItems(items: GalleryItemRef[], options?: { returnFocus?: () => HTMLElement | null }): Promise<void>;
  downloadItem(item: GalleryItem): Promise<void>;
  downloadItems(items: GalleryItemRef[], loadedItems?: GalleryItem[]): Promise<void>;
  moveItemsToBoard(items: GalleryItemRef[], boardId: string): Promise<void>;
  openItemInNewTab(item: GalleryItem): void;
  openItemInPreview(item: GalleryItem): void;
  setItemsStarred(items: GalleryItemRef[], starred: boolean): Promise<void>;
}

export interface GalleryItemActionContext {
  filterIdentity: string;
  /**
   * Stamp selections with the host window's page; using the grid's unrelated page would break navigation from deep
   * Preview windows.
   */
  getItemSelectionPage?(item: GalleryItem): number;
  items: GalleryItem[];
  loadOrderedRefs(signal: AbortSignal): Promise<GalleryItemRef[]>;
  selectedItemKey: GalleryItemKey | null;
}

export interface GalleryItemActionsOptions {
  boards: GalleryBoard[];
  generateValues: Record<string, unknown>;
  getItemActionContext?(): GalleryItemActionContext | null;
  projectId: string;
}

export interface GalleryItemContextMenuTarget {
  itemRefs: GalleryItemRef[];
  items: GalleryItem[];
  x: number;
  y: number;
}

export interface GalleryItemContextMenuProps {
  boards: GalleryBoard[];
  target: GalleryItemContextMenuTarget | null;
  onClose(): void;
}

export interface GalleryCommandsPort {
  clearSelection(): void;
  reconcileDeletedBoardOutcome(outcome: GalleryBoardDeletionResult): void;
  selectBoard(boardId: string): void;
  selectItem(item: GalleryItem): void;
  selectImage(image: GalleryImage): void;
  setCompareItem(image: GalleryImageItem | null): void;
  setCompareImage(image: GalleryImage | null): void;
  setItemMultiSelection(itemKeys: GalleryItemKey[], primaryItem: GalleryItem): void;
  setPage(page: number): void;
  setPageInfo(totalImages: number): void;
  setSearchTerm(searchTerm: string): void;
  setStarredOnly(starredOnly: boolean): void;
  setSemanticSearchMode(enabled: boolean): void;
  setSemanticSearchText(text: string): void;
  commitSemanticSearch(text: string): void;
  clearSearch(): void;
  setView(view: GalleryView): void;
  toggleItemSelection(item: GalleryItem, nextPrimaryItem: GalleryItem | null): void;
  updateSettings(settings: Partial<GallerySettings>): void;
}

export interface GalleryNotificationsPort {
  add(notification: { kind: 'info' | 'success'; message?: string; title: string }): void;
  reportError(error: { area: string; message: string; namespace: 'gallery' }): void;
}

export interface GalleryWidgetRuntime {
  commands: {
    register(command: { handler: () => unknown; id: string; title: string }): () => void;
  };
  hotkeys: {
    register(hotkey: { commandId: string; defaultKeys: string[]; id: string; title: string }): () => void;
  };
}

export interface GalleryWidgetProps {
  presentation?: 'compact' | 'expanded' | 'tooltip';
  region: 'bottom' | 'center' | 'dialog' | 'floating' | 'left' | 'popover' | 'right';
  runtime: GalleryWidgetRuntime;
}

/** This UI port preserves dependency direction: Gallery cannot import Workbench. */
export interface GalleryUiAdapter {
  ItemActionsProvider: ComponentType<GalleryItemActionsOptions & { children: ReactNode }>;
  ImageContextMenu: ComponentType<GalleryItemContextMenuProps>;
  antialiasProgressImages: boolean;
  gallery: GalleryCommandsPort;
  galleryValues: Record<string, unknown>;
  generateValues: Record<string, unknown>;
  /** Resolves an item's best image-map vocabulary label, or null when it has none. */
  getItemLabel(item: GalleryItemRef): Promise<string | null>;
  notifications: GalleryNotificationsPort;
  projectId: string;
  projectName: string;
  /** The account's projects, for naming the boards that belong to each; the open project need not be among them. */
  projects: readonly GalleryProjectRef[];
  /**
   * Make the open project exist on the server before a board is created in or moved into it: a new project is
   * saved only after its first debounced flush. Absent for hosts without a workbench, which never create in one.
   */
  ensureProjectOnServer?: () => Promise<void>;
  /**
   * Export a project as an `.invk`, reporting progress itself. Keyed by project rather than
   * board because a board menu can offer this for any project's board, not only the open one.
   */
  exportProject(projectId: string, projectName: string): void;
  progressSessions: QueueProgressSession[];
  pinnedProgressSessionId: string | null;
  /** The session Preview is showing while it follows live; null otherwise. The arrow keys step from it. */
  followedProgressSessionId: string | null;
  /** Follow `sessionId` live; a tile click also reveals Preview, an arrow step must not move the layout. */
  followProgressSession(sessionId: string, options: { revealPreview: boolean }): void;
  liveFollowEnabled: boolean;
  widgets: {
    /** Open (or reveal) the Gallery widget; false when no region can host it. */
    openGallery(): boolean;
    patchGalleryValues(values: Record<string, unknown>): void;
  };
}

const GalleryUiContext = createContext<GalleryUiAdapter | null>(null);
const GalleryItemActionsContext = createContext<GalleryItemActions | null>(null);

export const GalleryItemActionsProvider = ({
  actions,
  children,
}: {
  actions: GalleryItemActions;
  children: ReactNode;
}) => <GalleryItemActionsContext value={actions}>{children}</GalleryItemActionsContext>;

export const useGalleryItemActions = (): GalleryItemActions => {
  const actions = use(GalleryItemActionsContext);

  if (!actions) {
    throw new Error('Gallery item actions require the App-owned action adapter.');
  }

  return actions;
};

export const GalleryUiProvider = ({ adapter, children }: { adapter: GalleryUiAdapter; children: ReactNode }) => (
  <GalleryUiContext value={adapter}>{children}</GalleryUiContext>
);

export const useGalleryUi = (): GalleryUiAdapter => {
  const adapter = use(GalleryUiContext);

  if (!adapter) {
    throw new Error('Gallery UI requires an App-composed GalleryUiProvider.');
  }

  return adapter;
};

/** For hooks shared with hosts that have no workbench; they use it only for Gallery widget side effects. */
export const useOptionalGalleryUi = (): GalleryUiAdapter | null => use(GalleryUiContext);

/**
 * What gallery surfaces outside the Gallery widget (the picker, media slots) need from their host. The workbench
 * derives it from its UI adapter; a host without a workbench (the Launchpad) supplies one with GalleryHostProvider.
 */
export interface GalleryHost {
  galleryValues: Record<string, unknown>;
  notifications: GalleryNotificationsPort;
  /** The open project, whose boards list first; absent where there is none (the Launchpad). */
  projectId?: string;
  projectName: string;
  /** For naming other projects' boards; absent hosts list them unnamed. */
  projects?: readonly GalleryProjectRef[];
  /** Shows the Gallery widget at a view, switching board when one is given; absent where there is no widget. */
  revealInGallery?: (location: { boardId: string | null; view: GalleryView }) => boolean;
}

const GalleryHostContext = createContext<GalleryHost | null>(null);

export const GalleryHostProvider = ({ children, host }: { children: ReactNode; host: GalleryHost }) => (
  <GalleryHostContext value={host}>{children}</GalleryHostContext>
);

export const useGalleryHost = (): GalleryHost => {
  const host = use(GalleryHostContext);
  const adapter = use(GalleryUiContext);
  const derived = useMemo<GalleryHost | null>(
    () =>
      adapter && {
        galleryValues: adapter.galleryValues,
        notifications: adapter.notifications,
        projectId: adapter.projectId,
        projectName: adapter.projectName,
        projects: adapter.projects,
        revealInGallery: ({ boardId, view }) => {
          if (boardId !== null) {
            adapter.gallery.selectBoard(boardId);
          }
          adapter.gallery.setView(view);
          return adapter.widgets.openGallery();
        },
      },
    [adapter]
  );
  const resolved = host ?? derived;

  if (!resolved) {
    throw new Error('Gallery surfaces require a GalleryHostProvider or an App-composed GalleryUiProvider.');
  }

  return resolved;
};
