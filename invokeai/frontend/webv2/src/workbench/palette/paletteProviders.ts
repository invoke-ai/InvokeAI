import type { GalleryBoard, GalleryImage } from '@features/gallery/contracts';
import type { ModelConfig } from '@features/models';
import type { QueueReadModel } from '@features/queue/contracts';
import type { PromptHistoryItem } from '@workbench/projectContracts';
import type { TFunction } from 'i18next';

import { ALL_READABLE_BOARDS_ID, listGalleryImages } from '@features/gallery/paletteSearch';
import { galleryBoardsOptions } from '@features/gallery/queries';
import { focusPositivePrompt } from '@features/generation/react';
import { ensureModelsLoaded, getModelBaseLabel, getModelsSnapshot } from '@features/models';
import { extractGenerationMeta, getResultImageName } from '@features/queue/contracts';
import { listLibraryWorkflows } from '@features/workflow/paletteSearch';
import { requestLibraryWorkflowLoad } from '@features/workflow/react';
import { queryClient } from '@platform/query/client';
import { formatIsoDate, isTimestampInRange } from '@platform/search/dateTokens';
import { absolutizeApiUrl } from '@platform/transport/http';

import type { PaletteEntry, PaletteSearchProvider } from './entries';

import { getPaletteContributionKey } from './contributionKey';
import { getObjectIdentity } from './objectIdentity';

/**
 * First-party entity providers behind the palette's scoped search. Each is a
 * plain factory taking the workbench callbacks it needs — the host component
 * owns the hooks. Extension `search` contributions adapt into the same
 * PaletteSearchProvider shape, so all sections aggregate identically.
 */

const PROVIDER_PAGE_SIZE = 8;

/**
 * Boards back two providers on every debounced keystroke; read them through the
 * query cache (60s staleTime) instead of refetching per search.
 */
const loadActiveBoards = (): Promise<GalleryBoard[]> =>
  queryClient.fetchQuery(galleryBoardsOptions({ includeArchived: false }));

const matchesTerms = (haystack: string, query: string): boolean => {
  const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const lower = haystack.toLowerCase();

  return terms.every((term) => lower.includes(term));
};

export const createWorkflowsProvider = ({
  openWorkflowWidget,
  t,
}: {
  openWorkflowWidget: () => void;
  t: TFunction;
}): PaletteSearchProvider => ({
  contextKey: 'global',
  label: t('commandPalette.providers.workflows'),
  providerKey: getPaletteContributionKey('provider', 'workflows'),
  search: async (query, { signal }) => {
    const [user, defaults] = await Promise.all([
      listLibraryWorkflows({ category: 'user', page: 0, perPage: PROVIDER_PAGE_SIZE, query: query.text, signal }),
      listLibraryWorkflows({ category: 'default', page: 0, perPage: PROVIDER_PAGE_SIZE, query: query.text, signal }),
    ]);

    return [...user.items, ...defaults.items].map<PaletteEntry>((item) => ({
      group: 'Workflows',
      groupLabel: t('commandPalette.groups.workflows'),
      id: `workflow:${item.workflowId}`,
      isPersistentRecent: false,
      keywords: item.tags,
      run: () => {
        openWorkflowWidget();
        requestLibraryWorkflowLoad(item.workflowId);
      },
      subtitle:
        item.category === 'default'
          ? t('commandPalette.providers.defaultWorkflow')
          : t('commandPalette.providers.workflow'),
      title: item.name,
    }));
  },
});

export const createBoardsProvider = ({
  openGalleryWidget,
  selectBoard,
  t,
}: {
  openGalleryWidget: () => void;
  selectBoard: (boardId: string) => void;
  t: TFunction;
}): PaletteSearchProvider => ({
  contextKey: 'global',
  label: t('commandPalette.groups.boards'),
  providerKey: getPaletteContributionKey('provider', 'boards'),
  supportsCreatedAtRange: true,
  search: async (query, { signal }) => {
    const boards = await loadActiveBoards();

    signal.throwIfAborted();

    return boards
      .filter(
        (board: GalleryBoard) =>
          matchesTerms(board.name, query.text) &&
          // Boards without a creation date (uncategorized, date virtual
          // boards) are excluded only while a date range is active.
          (query.range === undefined || isTimestampInRange(board.createdAt ?? '', query.range))
      )
      .map<PaletteEntry>((board) => ({
        group: 'Boards',
        groupLabel: t('commandPalette.groups.boards'),
        id: `board:${board.id}`,
        isPersistentRecent: false,
        run: () => {
          openGalleryWidget();
          selectBoard(board.id);
        },
        subtitle: t('commandPalette.providers.boardSubtitle', { count: board.imageCount }),
        title: board.name,
      }));
  },
});

export const createPromptHistoryProvider = ({
  openGenerateWidget,
  projectId,
  promptHistory,
  recallContextKey,
  recallPrompt,
  t,
}: {
  openGenerateWidget: () => void;
  projectId: string;
  promptHistory: readonly PromptHistoryItem[];
  recallContextKey: string;
  recallPrompt: (item: PromptHistoryItem) => void;
  t: TFunction;
}): PaletteSearchProvider => ({
  contextKey: `${projectId}:${getObjectIdentity(promptHistory, 'history')}:${recallContextKey}`,
  label: t('commandPalette.providers.promptHistory'),
  providerKey: getPaletteContributionKey('provider', 'prompt-history'),
  search: (query, { signal }) => {
    signal.throwIfAborted();
    const seen = new Set<string>();
    const entries: PaletteEntry[] = [];

    for (const [index, item] of promptHistory.entries()) {
      const dedupeKey = `${item.positivePrompt}\n${item.negativePrompt ?? ''}`;

      // Prompt history items carry no timestamp, so this provider is text-only
      // (not range-capable); it never sees pure-date queries.
      if (seen.has(dedupeKey) || !matchesTerms(`${item.positivePrompt} ${item.negativePrompt ?? ''}`, query.text)) {
        continue;
      }

      seen.add(dedupeKey);
      entries.push({
        group: 'Prompt history',
        groupLabel: t('commandPalette.groups.promptHistory'),
        id: `prompt:${index}`,
        isPersistentRecent: false,
        run: () => {
          openGenerateWidget();
          recallPrompt(item);
          window.requestAnimationFrame(() => focusPositivePrompt());
        },
        subtitle: item.negativePrompt ? `− ${item.negativePrompt}` : undefined,
        title: item.positivePrompt,
      });
    }

    signal.throwIfAborted();
    return entries;
  },
});

const GENERATE_MODEL_TYPES = new Set(['external_image_generator', 'main']);

export const createModelsProvider = ({
  applyModel,
  openGenerateWidget,
  openModelManager,
  t,
}: {
  applyModel: (model: ModelConfig) => void;
  openGenerateWidget: () => void;
  openModelManager: () => void;
  t: TFunction;
}): PaletteSearchProvider => ({
  contextKey: 'global',
  label: t('commandPalette.providers.models'),
  providerKey: getPaletteContributionKey('provider', 'models'),
  search: async (query, { signal }) => {
    signal.throwIfAborted();
    await ensureModelsLoaded();
    signal.throwIfAborted();
    const snapshot = getModelsSnapshot();
    const models = snapshot.status === 'loaded' ? snapshot.models : [];

    return models
      .filter(
        (model) => GENERATE_MODEL_TYPES.has(model.type) && matchesTerms(`${model.name} ${model.base}`, query.text)
      )
      .map<PaletteEntry>((model) => ({
        group: 'Models',
        groupLabel: t('commandPalette.groups.models'),
        id: `model:${model.key}`,
        isPersistentRecent: false,
        keywords: model.base,
        run: () => {
          openGenerateWidget();
          applyModel(model);
        },
        secondary: { label: t('commandPalette.actions.openModelManager'), run: openModelManager },
        subtitle: getModelBaseLabel(model.base),
        title: model.name,
      }));
  },
});

export const createQueueItemsProvider = ({
  contextKey,
  loadQueue,
  openQueueWidget,
  revealItem,
  t,
}: {
  contextKey: string;
  loadQueue: () => Promise<QueueReadModel>;
  openQueueWidget: () => void;
  revealItem: (itemId: number) => void;
  t: TFunction;
}): PaletteSearchProvider => ({
  contextKey,
  label: t('commandPalette.providers.queueItems'),
  providerKey: getPaletteContributionKey('provider', 'queue-items'),
  supportsCreatedAtRange: true,
  search: async (query, { signal }) => {
    // The queue read model is a recent window with no server-side text search;
    // filter the fetched window client-side. Items without a timestamp fail
    // closed while a date range is active.
    signal.throwIfAborted();
    const model = await loadQueue();
    signal.throwIfAborted();

    return model.items
      .map((item) => ({ item, meta: extractGenerationMeta(item) }))
      .filter(
        ({ item, meta }) =>
          matchesTerms(
            `${meta.positivePrompt ?? ''} ${meta.negativePrompt ?? ''} ${t(`commandPalette.queueStatuses.${item.status}`)} ${item.userDisplayName ?? ''}`,
            query.text
          ) &&
          (query.range === undefined || isTimestampInRange(item.createdAt, query.range))
      )
      .map<PaletteEntry>(({ item, meta }) => {
        const resultImageName = getResultImageName(item);

        return {
          group: 'Queue items',
          groupLabel: t('commandPalette.groups.queueItems'),
          id: `queue-item:${item.id}`,
          isPersistentRecent: false,
          run: () => {
            openQueueWidget();
            revealItem(item.id);
          },
          subtitle: t(`commandPalette.queueStatuses.${item.status}`),
          thumbnailUrl: resultImageName
            ? absolutizeApiUrl(`/api/v1/images/i/${encodeURIComponent(resultImageName)}/thumbnail`)
            : undefined,
          title: meta.positivePrompt?.trim() || t('commandPalette.providers.noTitle'),
        };
      });
  },
});

export const createImagesProvider = ({
  openGalleryWidget,
  openPreviewWidget,
  selectBoard,
  selectImage,
  locale,
  t,
}: {
  openGalleryWidget: () => void;
  openPreviewWidget: () => void;
  selectBoard: (boardId: string) => void;
  selectImage: (image: GalleryImage) => void;
  locale?: string;
  t: TFunction;
}): PaletteSearchProvider => ({
  contextKey: 'global',
  label: t('commandPalette.providers.images'),
  providerKey: getPaletteContributionKey('provider', 'images'),
  supportsCreatedAtRange: true,
  search: async (query, { signal }) => {
    // Search the explicit all-readable scope. The endpoint excludes archived
    // and inaccessible boards; the active board list supplies display labels.
    const [page, boards] = await Promise.all([
      listGalleryImages({
        boardId: ALL_READABLE_BOARDS_ID,
        createdFrom: query.range?.from,
        createdTo: query.range?.to,
        galleryView: 'images',
        limit: 20,
        searchTerm: query.text,
        signal,
      }),
      loadActiveBoards(),
    ]);
    const boardNames = new Map(boards.map((board: GalleryBoard) => [board.id, board.name]));

    return page.images.map<PaletteEntry>((image) => ({
      group: 'Images',
      groupLabel: t('commandPalette.groups.images'),
      id: `image:${image.imageName}`,
      isPersistentRecent: false,
      run: () => {
        openPreviewWidget();
        selectImage(image);
      },
      secondary: {
        label: t('commandPalette.actions.revealInGallery'),
        run: () => {
          openGalleryWidget();
          selectBoard(image.boardId);
          selectImage(image);
        },
      },
      subtitle: `${boardNames.get(image.boardId) ?? t('commandPalette.providers.uncategorized')} · ${formatIsoDate(image.queuedAt.slice(0, 10), locale)} · ${image.width}×${image.height}`,
      thumbnailUrl: image.thumbnailUrl,
      title: image.imageName,
    }));
  },
});
