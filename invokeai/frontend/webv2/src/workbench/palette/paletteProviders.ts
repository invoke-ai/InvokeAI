import type { GenerateModelConfig } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';
import type { QueueReadModel } from '@features/queue/contracts';
import type { PromptHistoryItem } from '@workbench/projectContracts';
import type { TFunction } from 'i18next';

import { getGalleryBoardLabel, type GalleryBoard, type GalleryImage } from '@features/gallery/contracts';
import { ALL_READABLE_BOARDS_ID, listPaletteImages, listPaletteSemanticImages } from '@features/gallery/paletteSearch';
import { galleryBoardsOptions, imageIndexAvailabilityOptions } from '@features/gallery/queries';
import { focusPositivePrompt } from '@features/generation/react';
import { isGenerateModelSelectable } from '@features/generation/settings';
import { ensureModelsLoaded, getModelBaseLabel, getModelsSnapshot } from '@features/models';
import { extractGenerationMeta, getResultImageName } from '@features/queue/contracts';
import { listLibraryWorkflows } from '@features/workflow/paletteSearch';
import { requestLibraryWorkflowLoad } from '@features/workflow/react';
import { queryClient } from '@platform/query/client';
import { isTimestampInRange } from '@platform/search/dateTokens';
import { normalizeServerTimestamp } from '@platform/time/serverTimestamp';
import { absolutizeApiUrl, ApiError } from '@platform/transport/http';

import type { PaletteEntry, PaletteSearchProvider } from './entries';

import { getPaletteContributionKey } from './contributionKey';
import { PaletteSearchUnavailableError } from './entries';
import { getObjectIdentity } from './objectIdentity';

/** Factories receive host-owned workbench callbacks; extension searches adapt to the same provider contract. */

const PROVIDER_PAGE_SIZE = 8;

/**
 * Boards back two providers on every debounced keystroke; read them through the
 * query cache (60s staleTime) instead of refetching per search.
 */
const loadActiveBoards = (): Promise<GalleryBoard[]> =>
  queryClient.fetchQuery(galleryBoardsOptions({ includeArchived: false }));

const createTermsMatcher = (query: string): ((haystack: string) => boolean) => {
  const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);

  return (haystack) => {
    const lower = haystack.toLowerCase();

    return terms.every((term) => lower.includes(term));
  };
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
    const matchesQuery = createTermsMatcher(query.text);

    signal.throwIfAborted();

    return boards
      .filter(
        (board: GalleryBoard) =>
          matchesQuery(getGalleryBoardLabel(board, t)) &&
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
        title: getGalleryBoardLabel(board, t),
      }));
  },
});

export const createPromptHistoryProvider = ({
  openGenerateWidget,
  projectId,
  promptHistory,
  recallPrompt,
  t,
}: {
  openGenerateWidget: () => void;
  projectId: string;
  promptHistory: readonly PromptHistoryItem[];
  recallPrompt: (item: PromptHistoryItem) => void;
  t: TFunction;
}): PaletteSearchProvider => ({
  contextKey: `${projectId}:${getObjectIdentity(promptHistory, 'history')}`,
  label: t('commandPalette.providers.promptHistory'),
  providerKey: getPaletteContributionKey('provider', 'prompt-history'),
  search: (query, { signal }) => {
    signal.throwIfAborted();
    const seen = new Set<string>();
    const entries: PaletteEntry[] = [];
    const matchesQuery = createTermsMatcher(query.text);

    for (const [index, item] of promptHistory.entries()) {
      const dedupeKey = `${item.positivePrompt}\n${item.negativePrompt ?? ''}`;

      // Prompt history items carry no timestamp, so this provider is text-only
      // (not range-capable); it never sees pure-date queries.
      if (seen.has(dedupeKey) || !matchesQuery(`${item.positivePrompt} ${item.negativePrompt ?? ''}`)) {
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

export const createModelsProvider = ({
  applyModel,
  openGenerateWidget,
  openModelManager,
  t,
}: {
  applyModel: (model: GenerateModelConfig, models: readonly ModelConfig[]) => void;
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
    const matchesQuery = createTermsMatcher(query.text);

    return models
      .filter(isGenerateModelSelectable)
      .filter((model) => matchesQuery(`${model.name} ${model.base}`))
      .map<PaletteEntry>((model) => ({
        group: 'Models',
        groupLabel: t('commandPalette.groups.models'),
        id: `model:${model.key}`,
        isPersistentRecent: false,
        keywords: model.base,
        run: () => {
          applyModel(model, models);
          openGenerateWidget();
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
    const matchesQuery = createTermsMatcher(query.text);
    signal.throwIfAborted();

    return model.items
      .map((item) => ({ item, meta: extractGenerationMeta(item) }))
      .filter(
        ({ item, meta }) =>
          matchesQuery(
            `${meta.positivePrompt ?? ''} ${meta.negativePrompt ?? ''} ${t(`commandPalette.queueStatuses.${item.status}`)} ${item.userDisplayName ?? ''}`
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

interface ImageEntryDeps {
  openPreviewWidget: () => void;
  /** Shows the image in its own board with the gallery's filters cleared. */
  revealImage: (image: GalleryImage) => void;
  selectImage: (image: GalleryImage) => void;
  locale?: string;
  t: TFunction;
}

/** Opening previews the image; the secondary action reveals it in its board. */
const createImageEntryMapper = ({ locale, openPreviewWidget, revealImage, selectImage, t }: ImageEntryDeps) => {
  const titleFormatter = new Intl.DateTimeFormat(locale, {
    day: 'numeric',
    hour: 'numeric',
    minute: 'numeric',
    month: 'short',
  });

  return (images: readonly GalleryImage[], boards: readonly GalleryBoard[], idPrefix: string): PaletteEntry[] => {
    const boardNames = new Map(boards.map((board) => [board.id, getGalleryBoardLabel(board, t)] as const));

    // Use date and time to distinguish promptless images generated on the same day.
    return images.map<PaletteEntry>((image) => {
      const createdAt = new Date(normalizeServerTimestamp(image.createdAt ?? image.queuedAt));

      return {
        group: 'Images',
        groupLabel: t('commandPalette.groups.images'),
        id: `${idPrefix}:${image.imageName}`,
        isPersistentRecent: false,
        run: () => {
          openPreviewWidget();
          selectImage(image);
        },
        secondary: {
          label: t('commandPalette.actions.revealInGallery'),
          run: () => revealImage(image),
        },
        subtitle: `${boardNames.get(image.boardId) ?? t('commandPalette.providers.uncategorized')} · ${image.width}×${image.height}`,
        thumbnailUrl: image.thumbnailUrl,
        title: Number.isNaN(createdAt.getTime()) ? image.imageName : titleFormatter.format(createdAt),
      };
    });
  };
};

export const createImagesProvider = (deps: ImageEntryDeps): PaletteSearchProvider => {
  const toEntries = createImageEntryMapper(deps);

  return {
    contextKey: 'global',
    label: deps.t('commandPalette.providers.images'),
    providerKey: getPaletteContributionKey('provider', 'images'),
    supportsCreatedAtRange: true,
    search: async (query, { signal }) => {
      // Search the explicit all-readable scope. The endpoint excludes archived
      // and inaccessible boards; the active board list supplies display labels.
      const [page, boards] = await Promise.all([
        listPaletteImages({
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

      return toEntries(page.images, boards, 'image');
    },
  };
};

/**
 * Ranks every accessible image by meaning, unlike the gallery's search, which stays within its board. Scoped-only:
 * each search runs the server's text encoder, and cosine ranking matches any query, so root typing never triggers it.
 */
export const createSemanticImagesProvider = (deps: ImageEntryDeps): PaletteSearchProvider => {
  const toEntries = createImageEntryMapper(deps);

  return {
    contextKey: 'global',
    label: deps.t('commandPalette.providers.semanticImages'),
    providerKey: getPaletteContributionKey('provider', 'semantic-images'),
    scopedOnly: true,
    search: async (query, { signal }) => {
      const text = query.text.trim();

      if (!text || (await queryClient.fetchQuery(imageIndexAvailabilityOptions())).state !== 'ready') {
        return [];
      }

      const searched = listPaletteSemanticImages({ limit: 20, query: text, signal }).catch((error: unknown) => {
        // 409: the index can't embed text, typically a model without a usable text encoder.
        if (error instanceof ApiError && error.status === 409) {
          throw new PaletteSearchUnavailableError(deps.t('commandPalette.states.semanticTextUnavailable'));
        }

        throw error;
      });
      const [images, boards] = await Promise.all([searched, loadActiveBoards()]);

      return toEntries(images, boards, 'semantic-image');
    },
  };
};
