import type { GalleryImageItem, GalleryItem, GalleryItemKey, GalleryItemRef } from '@features/gallery/contracts';
import type { GalleryMediaUsageReport } from '@features/gallery/core/mediaUsage';
import type { GallerySettings } from '@features/gallery/core/settings';
import type {
  GalleryBoard,
  GalleryBoardDeletionResult,
  GalleryImage,
  GalleryProjectRef,
  GalleryView,
} from '@features/gallery/core/types';
import type { QueueProgressSession } from '@features/queue/contracts';
import type { ResolvedValues } from '@platform/core/resolvedValues';

import { asSingleAccountBoards } from '@features/gallery/core/boardAccess';
import { createKeyedTransientStore, type KeyedTransientStore } from '@platform/state/externalStore';
import {
  createContext,
  use,
  useCallback,
  useMemo,
  useState,
  type ComponentType,
  type MouseEvent,
  type ReactNode,
} from 'react';

export interface GalleryItemActions {
  /**
   * `returnFocus` is resolved on close: the deletion can remove the opener. `targetsOffScreen` makes the deletion ask
   * even with routine confirmations off, as any multi-item deletion does.
   */
  deleteItems(
    items: GalleryItemRef[],
    options?: { returnFocus?: () => HTMLElement | null; targetsOffScreen?: boolean }
  ): Promise<void>;
  downloadItem(item: GalleryItem): Promise<void>;
  downloadItems(items: GalleryItemRef[], loadedItems?: GalleryItem[]): Promise<void>;
  moveItemsToBoard(items: GalleryItemRef[], boardId: string): Promise<void>;
  openItemInNewTab(item: GalleryItem): void;
  openItemInPreview(item: GalleryItem): void;
  setItemsStarred(items: GalleryItemRef[], starred: boolean): Promise<void>;
}

export interface GalleryItemActionContext {
  filterIdentity: string;
  /** The grid's unrelated page breaks deep Preview navigation. */
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
  /** `items` omits unloaded targets. */
  allStarred?: boolean;
  itemRefs: GalleryItemRef[];
  items: GalleryItem[];
  listingBoardId?: string;
  /** Resolved on close: the menu that had focus is gone by then. */
  returnFocus?: () => HTMLElement | null;
  x: number;
  y: number;
}

export interface GalleryItemContextMenuRequest {
  /** What the surface already holds; the menu fetches the item without it. */
  item?: GalleryItem;
  ref: GalleryItemRef;
  /** Resolved on close. */
  returnFocus?: () => HTMLElement | null;
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
  selectItem(item: GalleryItem, selectionPage?: number): void;
  selectImage(image: GalleryImage): void;
  setCompareItem(image: GalleryImageItem | null): void;
  setCompareImage(image: GalleryImage | null): void;
  setItemMultiSelection(itemKeys: GalleryItemKey[], primaryItem: GalleryItem, selectionPage?: number): void;
  setPage(page: number): void;
  setPageInfo(totalImages: number): void;
  setSearchTerm(searchTerm: string): void;
  setStarredOnly(starredOnly: boolean): void;
  setSemanticSearchMode(enabled: boolean): void;
  setSemanticSearchText(text: string): void;
  commitSemanticSearch(text: string): void;
  clearSearch(): void;
  setView(view: GalleryView): void;
  /** The grid page of whichever item becomes primary. */
  toggleItemSelection(item: GalleryItem, nextPrimaryItem: GalleryItem | null, selectionPage?: number): void;
  /** With `projectId`, writes to that project while open, for work that finishes after the user moved on (an Undo). */
  updateSettings(settings: Partial<GallerySettings>, projectId?: string): void;
}

export interface GalleryNotificationsPort {
  add(notification: { kind: 'info' | 'success'; message?: string; title: string }): void;
  reportError(error: { area: string; message: string; namespace: 'gallery' }): void;
  /** The Undo may run minutes later; a newer Undo with the same `supersedes` key withdraws the older one. */
  undoable?(title: string, notice: { message?: string; onUndo: () => void; supersedes?: string }): void;
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

/** Gallery cannot import Workbench. */
export interface GalleryUiAdapter {
  ItemActionsProvider: ComponentType<GalleryItemActionsOptions & { children: ReactNode }>;
  ImageContextMenu: ComponentType<GalleryItemContextMenuProps>;
  antialiasProgressImages: boolean;
  /** The server has more than one account. */
  boardSharing?: boolean;
  gallery: GalleryCommandsPort;
  galleryValues: ResolvedValues;
  generateValues: Record<string, unknown>;
  /** The host narrows the server's check with its projects' own scans; rejects when the check cannot run. */
  getBoardMediaUsage(boardId: string, signal: AbortSignal): Promise<GalleryMediaUsageReport>;
  /** Null once the project is closed. */
  getProjectGalleryValues?(projectId: string): ResolvedValues | null;
  /** The host may explain where new results still go. */
  noteBoardBrowsed?(boardId: string): void;
  getItemLabel(item: GalleryItemRef): Promise<string | null>;
  notifications: GalleryNotificationsPort;
  /**
   * The item menu for media shown outside the Gallery widget; a newer request replaces a pending or open one. Stable:
   * media surfaces read it alone, so adapter rebuilds do not re-render them.
   */
  openItemContextMenu(request: GalleryItemContextMenuRequest): void;
  projectId: string;
  projectName: string;
  /** The open project need not be among them. */
  projects: readonly GalleryProjectRef[];
  /** A new project is saved only after its first debounced flush; boards must not be created in it before. */
  ensureProjectOnServer?: () => Promise<void>;
  /** Keyed by project: a board menu offers it for any project's board. */
  exportProject(projectId: string, projectName: string): void;
  progressSessions: QueueProgressSession[];
  pinnedProgressSessionId: string | null;
  /** Null unless following live; the arrow keys step from it. */
  followedProgressSessionId: string | null;
  /** An arrow step must not move the layout, so only a tile click reveals Preview. */
  followProgressSession(sessionId: string, options: { revealPreview: boolean }): void;
  liveFollowEnabled: boolean;
  widgets: {
    /** False when no region can host it. */
    openGallery(): boolean;
    patchGalleryValues(values: Record<string, unknown>): void;
  };
}

const GalleryUiContext = createContext<GalleryUiAdapter | null>(null);
const GalleryItemContextMenuContext = createContext<GalleryUiAdapter['openItemContextMenu'] | undefined>(undefined);
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

export type GalleryLayoutMode = 'stacked' | 'wide';

/** The workbench renders the header label apart from the body, so the label reads the measured arrangement here. */
export const GalleryLayoutStoreContext = createContext<KeyedTransientStore<
  GalleryWidgetProps['region'],
  GalleryLayoutMode
> | null>(null);

export const GalleryUiProvider = ({ adapter, children }: { adapter: GalleryUiAdapter; children: ReactNode }) => {
  const [layoutStore] = useState(() => createKeyedTransientStore<GalleryWidgetProps['region'], GalleryLayoutMode>());

  return (
    <GalleryUiContext value={adapter}>
      <GalleryItemContextMenuContext value={adapter.openItemContextMenu}>
        <GalleryLayoutStoreContext value={layoutStore}>{children}</GalleryLayoutStoreContext>
      </GalleryItemContextMenuContext>
    </GalleryUiContext>
  );
};

export const useGalleryUi = (): GalleryUiAdapter => {
  const adapter = use(GalleryUiContext);

  if (!adapter) {
    throw new Error('Gallery UI requires an App-composed GalleryUiProvider.');
  }

  return adapter;
};

export const useOptionalGalleryUi = (): GalleryUiAdapter | null => use(GalleryUiContext);

const NO_BOARDS: GalleryBoard[] = [];

/**
 * Every surface that names boards reads them here, so a single-account host folds boards into the caller's own
 * consistently. Undefined reads as no boards, with a stable identity.
 */
export const useHostGalleryBoards = (boards: readonly GalleryBoard[] | undefined): GalleryBoard[] => {
  const isSingleAccount = useOptionalGalleryUi()?.boardSharing === false;

  return useMemo(
    () =>
      boards === undefined
        ? NO_BOARDS
        : isSingleAccount
          ? asSingleAccountBoards(boards)
          : // The listing's own identity: a folded list is a new one only when folding changes a board.
            (boards as GalleryBoard[]),
    [boards, isSingleAccount]
  );
};

/** A host without a workbench (the Launchpad) supplies one with GalleryHostProvider. */
export interface GalleryHost {
  galleryValues: ResolvedValues;
  notifications: GalleryNotificationsPort;
  projectId?: string;
  projectName: string;
  projects?: readonly GalleryProjectRef[];
  revealInGallery?: (location: { boardId: string | null; view: GalleryView }) => boolean;
}

const GalleryHostContext = createContext<GalleryHost | null>(null);

export const GalleryHostProvider = ({ children, host }: { children: ReactNode; host: GalleryHost }) => (
  <GalleryHostContext value={host}>{children}</GalleryHostContext>
);

export type GalleryHostProjectContext = Pick<GalleryHost, 'projectId' | 'projectName' | 'projects'>;

export const useOptionalGalleryHostProjectContext = (): GalleryHostProjectContext | null => {
  const source = use(GalleryHostContext) ?? use(GalleryUiContext);
  const hasSource = source !== null;
  const projectId = source?.projectId;
  const projectName = source?.projectName;
  const projects = source?.projects;

  return useMemo(
    () => (hasSource ? { projectId, projectName: projectName ?? '', projects } : null),
    [hasSource, projectId, projectName, projects]
  );
};

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

const getFocusedElement = (): HTMLElement | null => {
  const { activeElement } = document;

  return activeElement instanceof HTMLElement && activeElement !== document.body ? activeElement : null;
};

const isGalleryItem = (media: GalleryItem | GalleryItemRef): media is GalleryItem => 'fullUrl' in media;

/**
 * The item menu on right-click (or the context-menu key) for an element showing gallery media; pass the item itself
 * when the surface has it, so the menu opens without a fetch. Undefined without media or outside the workbench (a
 * `GalleryHostProvider` host), where the browser's own menu stays.
 */
export const useGalleryItemContextMenu = (
  media: GalleryItem | GalleryItemRef | null
): ((event: MouseEvent<HTMLElement>) => void) | undefined => {
  const host = use(GalleryHostContext);
  const open = use(GalleryItemContextMenuContext);
  const item = media && isGalleryItem(media) ? media : undefined;
  const kind = media?.kind;
  const name = media?.name;
  const handleContextMenu = useCallback(
    (event: MouseEvent<HTMLElement>) => {
      const { currentTarget, target } = event;

      // Handled already by media nested inside this one. React also bubbles events up from portals: a popover opened
      // from the media (its picker) keeps its own menus.
      if (
        !open ||
        kind === undefined ||
        name === undefined ||
        event.defaultPrevented ||
        !(target instanceof Node && currentTarget.contains(target))
      ) {
        return;
      }

      // Not stopped: owners around the media (a workflow node) read `defaultPrevented`, and window listeners (a touch
      // hold disarming its drag) must still hear the event.
      event.preventDefault();
      const opener = getFocusedElement();

      open({
        item,
        ref: { kind, name },
        returnFocus: () => (opener?.isConnected ? opener : null),
        x: event.clientX,
        y: event.clientY,
      });
    },
    [item, kind, name, open]
  );

  return host || !open || kind === undefined || name === undefined ? undefined : handleContextMenu;
};
