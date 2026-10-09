import type { AccountScope } from '@platform/state/accountLifecycle';
import type { QueryClient } from '@tanstack/react-query';
import type { Project } from '@workbench/projectContracts';
import type { TFunction } from 'i18next';

import { getGalleryBoardLabel, getGallerySettings, type GalleryBoard } from '@features/gallery/contracts';
import {
  galleryBoardsOptions,
  getGalleryListingBoardsQuery,
  invalidateGallery,
  type GalleryBoardsQuery,
} from '@features/gallery/queries';
import {
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { createKeyedTransientStore } from '@platform/state/externalStore';
import { createActionToast, toaster, type ActionToastOptions, type ToastAction } from '@platform/ui/toaster';
import { useQueryClient } from '@tanstack/react-query';
import {
  retryStagedResultBoard,
  saveStagedResultToGallery,
  type SaveStagedResultOutcome,
} from '@workbench/canvas-operations/api';
import { useFindGalleryItem } from '@workbench/image-actions/useFindGalleryItem';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

/** Staged images with a save or board retry in flight; the bar and the thumbnail menu share the gate. */
const savingImages = createKeyedTransientStore<string, true>();
registerAccountOwnedResource({ clear: () => savingImages.clear(), name: 'staged-result-gallery-saves' });

/** Actionable toasts stay long enough to reach the button; hovering or focusing them pauses the timer. */
const ACTION_TOAST_DURATION_MS = 8000;
/** How long a settled save waits for a boards listing it does not have before it reports without a name. */
const BOARD_LABEL_WAIT_MS = 600;

type BoardDescription = { kind: 'named'; label: string } | { kind: 'missing' } | { kind: 'unknown' };

/**
 * The board as the Gallery names it. Each listing is read from the cache at once, otherwise fetched, all within one
 * bound so feedback never waits on a slow request. Archiving a board keeps it the auto-add destination, so a board
 * absent from a listing that hides archived boards is looked for once more among them; `missing` only when a listing
 * that includes archived boards lacks it.
 */
const describeBoard = async (
  queryClient: QueryClient,
  project: Project,
  boardId: string,
  t: TFunction
): Promise<BoardDescription> => {
  if (boardId === 'none') {
    return { kind: 'named', label: t('widgets.gallery.uncategorized') };
  }

  const deadline = new Promise<null>((resolve) => {
    globalThis.setTimeout(() => resolve(null), BOARD_LABEL_WAIT_MS);
  });
  const readBoards = (query: GalleryBoardsQuery): Promise<GalleryBoard[] | null> => {
    const options = galleryBoardsOptions(query);
    const cached = queryClient.getQueryData<GalleryBoard[]>(options.queryKey);

    return cached
      ? Promise.resolve(cached)
      : Promise.race([queryClient.fetchQuery(options).catch(() => null), deadline]);
  };
  const describe = (boards: GalleryBoard[]): BoardDescription | null => {
    const board = boards.find((candidate) => candidate.id === boardId);

    return board ? { kind: 'named', label: getGalleryBoardLabel(board, t) } : null;
  };
  const listingQuery = getGalleryListingBoardsQuery(getGallerySettings(getProjectWidgetValues(project, 'gallery')));
  const listing = await readBoards(listingQuery);
  const listed = listing && describe(listing);

  if (listed) {
    return listed;
  }

  const everything = listingQuery.includeArchived
    ? listing
    : await readBoards({ ...listingQuery, includeArchived: true });

  if (!everything) {
    return { kind: 'unknown' };
  }

  return describe(everything) ?? { kind: 'missing' };
};

/** What a save, or a board retry, belongs to: the account and project current when the user asked for it. */
interface SaveOrigin {
  owner: AccountScope;
  project: Project;
}

/**
 * Report to the account that saved. The toast names that account's image and board, so it leaves with the account's
 * scope rather than outlive it with buttons that would act for nobody.
 */
const announce = (owner: AccountScope, options: ActionToastOptions): void => {
  const shown = new AbortController();
  const id = createActionToast({
    ...options,
    onStatusChange: ({ status }) => {
      if (status === 'unmounted') {
        shown.abort();
      }
    },
  });

  owner.signal.addEventListener('abort', () => toaster.dismiss(id), { once: true, signal: shown.signal });
};

/**
 * Save a staged canvas result to the Gallery and say where it went. The account and project are captured when the
 * save starts and every toast action is fenced to them: nothing runs for an account that has since signed out, and
 * "Show in Gallery" acts only while the project is still open. It reveals the item in a Gallery panel and leaves
 * the center (the Canvas under review) where it is.
 */
export const useStagedResultGallerySave = (
  imageName: string | null
): { isSaving: boolean; save: (imageName: string) => Promise<void> } => {
  const { t } = useTranslation();
  const queries = useWorkbenchQueries();
  const queryClient = useQueryClient();
  const findGalleryItem = useFindGalleryItem();
  const isSaving = savingImages.useValue(imageName ?? '') === true;

  const save = useMemo(() => {
    const run = async (
      name: string,
      { owner, project }: SaveOrigin,
      attempt: () => Promise<SaveStagedResultOutcome>
    ): Promise<void> => {
      if (!isAccountScopeCurrent(owner) || savingImages.get(name)) {
        return;
      }

      const showInGallery: ToastAction = {
        label: t('widgets.canvas.staging.showInGallery'),
        onClick: () => {
          if (isAccountScopeCurrent(owner) && queries.isActiveProject(project.id)) {
            findGalleryItem({ kind: 'image', name }, { revealPreview: false });
          }
        },
      };

      savingImages.set(name, true);

      try {
        const outcome = await attempt();

        if (outcome.status === 'stale' || !isAccountScopeCurrent(owner)) {
          return;
        }

        // The listing, its paging and the board counts are server-owned; refetch rather than guess where it lands.
        void invalidateGallery(queryClient, owner);
        const board = await describeBoard(queryClient, project, outcome.boardId, t);

        if (!isAccountScopeCurrent(owner)) {
          return;
        }

        const showActions = queries.isActiveProject(project.id) ? [showInGallery] : [];
        const description = t('widgets.canvas.staging.savedDescription', { name });

        if (outcome.status === 'saved') {
          const title =
            board.kind === 'named'
              ? t('widgets.canvas.staging.savedToBoard', { board: board.label })
              : t('widgets.canvas.staging.saved');

          announce(owner, {
            actions: showActions,
            description,
            ...(showActions.length > 0 ? { duration: ACTION_TOAST_DURATION_MS } : {}),
            title,
            type: 'success',
          });
          return;
        }

        if (board.kind === 'missing') {
          // Retrying cannot bring a deleted board back.
          announce(owner, {
            actions: showActions,
            description: t('widgets.canvas.staging.boardMissingDescription', { name }),
            ...(showActions.length > 0 ? { duration: ACTION_TOAST_DURATION_MS } : {}),
            title: t('widgets.canvas.staging.boardMissing'),
            type: 'warning',
          });
          return;
        }

        const retry: ToastAction = {
          label: t('common.retry'),
          onClick: () =>
            void run(name, { owner, project }, () =>
              retryStagedResultBoard({ boardId: outcome.boardId, imageName: name })
            ),
        };

        announce(owner, {
          actions: [retry, ...showActions],
          description: t('widgets.canvas.staging.boardFailedDescription', { name }),
          duration: ACTION_TOAST_DURATION_MS,
          title:
            board.kind === 'named'
              ? t('widgets.canvas.staging.boardFailed', { board: board.label })
              : t('widgets.canvas.staging.boardFailedUnnamed'),
          type: 'warning',
        });
      } catch {
        if (isAccountScopeCurrent(owner)) {
          announce(owner, { actions: [], title: t('widgets.canvas.staging.saveError'), type: 'error' });
        }
      } finally {
        savingImages.delete(name);
      }
    };

    return (name: string) => {
      const origin = { owner: captureAccountScope(), project: queries.getSnapshot().activeProject };

      return run(name, origin, () => saveStagedResultToGallery({ imageName: name, project: origin.project }));
    };
  }, [findGalleryItem, queries, queryClient, t]);

  return { isSaving, save };
};
