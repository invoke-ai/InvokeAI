import type { GalleryBoard, GalleryImage } from '@features/gallery/contracts';
import type { ModelConfig } from '@features/models';
import type { QueueItemStatus, QueueReadModel } from '@features/queue/contracts';
import type { PromptHistoryItem } from '@workbench/projectContracts';

import { galleryBoardsOptions, galleryImagesOptions } from '@features/gallery/queries';
import { focusPositivePrompt } from '@features/generation/react';
import { ensureModelsLoaded, getModelBaseLabel, getModelsSnapshot } from '@features/models';
import { extractGenerationMeta, getResultImageName } from '@features/queue/contracts';
import { listLibraryWorkflows } from '@features/workflow/queries';
import { requestLibraryWorkflowLoad } from '@features/workflow/react';
import { queryClient } from '@platform/query/client';
import { formatIsoDate, isTimestampInRange } from '@platform/search/dateTokens';
import { absolutizeApiUrl } from '@platform/transport/http';

import type { PaletteEntry, PaletteSearchProvider } from './entries';

import { getPaletteContributionKey } from './contributionKey';

/**
 * First-party entity providers behind the palette's scoped search. Each is a
 * plain factory taking the workbench callbacks it needs — the host component
 * owns the hooks. Extension `search` contributions adapt into the same
 * PaletteSearchProvider shape, so all sections aggregate identically.
 */

const PROVIDER_PAGE_SIZE = 8;

const matchesTerms = (haystack: string, query: string): boolean => {
  const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const lower = haystack.toLowerCase();

  return terms.every((term) => lower.includes(term));
};

export const createWorkflowsProvider = ({
  openWorkflowWidget,
}: {
  openWorkflowWidget: () => void;
}): PaletteSearchProvider => ({
  contextKey: 'global',
  label: 'Workflows',
  providerKey: getPaletteContributionKey('provider', 'workflows'),
  search: async (query) => {
    const [user, defaults] = await Promise.all([
      listLibraryWorkflows({ category: 'user', page: 0, perPage: PROVIDER_PAGE_SIZE, query: query.text }),
      listLibraryWorkflows({ category: 'default', page: 0, perPage: PROVIDER_PAGE_SIZE, query: query.text }),
    ]);

    return [...user.items, ...defaults.items].map<PaletteEntry>((item) => ({
      group: 'Workflows',
      id: `workflow:${item.workflow_id}`,
      isPersistentRecent: false,
      keywords: item.tags ?? undefined,
      run: () => {
        openWorkflowWidget();
        requestLibraryWorkflowLoad(item.workflow_id);
      },
      subtitle: item.category === 'default' ? 'Default workflow' : 'Workflow',
      title: item.name,
    }));
  },
});

export const createBoardsProvider = ({
  openGalleryWidget,
  selectBoard,
}: {
  openGalleryWidget: () => void;
  selectBoard: (boardId: string) => void;
}): PaletteSearchProvider => ({
  contextKey: 'global',
  label: 'Boards',
  providerKey: getPaletteContributionKey('provider', 'boards'),
  supportsCreatedAtRange: true,
  search: async (query) => {
    const boards = await queryClient.fetchQuery(galleryBoardsOptions({ includeArchived: false }));

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
        id: `board:${board.id}`,
        isPersistentRecent: false,
        run: () => {
          openGalleryWidget();
          selectBoard(board.id);
        },
        subtitle: `Board · ${board.imageCount} image${board.imageCount === 1 ? '' : 's'}`,
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
}: {
  openGenerateWidget: () => void;
  projectId: string;
  promptHistory: readonly PromptHistoryItem[];
  recallContextKey: string;
  recallPrompt: (item: PromptHistoryItem) => void;
}): PaletteSearchProvider => ({
  contextKey: JSON.stringify([projectId, promptHistory, recallContextKey]),
  label: 'Prompt history',
  providerKey: getPaletteContributionKey('provider', 'prompt-history'),
  search: (query) => {
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

    return entries;
  },
});

const GENERATE_MODEL_TYPES = new Set(['external_image_generator', 'main']);

export const createModelsProvider = ({
  applyModel,
  openGenerateWidget,
  openModelManager,
}: {
  applyModel: (model: ModelConfig) => void;
  openGenerateWidget: () => void;
  openModelManager: () => void;
}): PaletteSearchProvider => ({
  contextKey: 'global',
  label: 'Models',
  providerKey: getPaletteContributionKey('provider', 'models'),
  search: async (query) => {
    await ensureModelsLoaded();
    const snapshot = getModelsSnapshot();
    const models = snapshot.status === 'loaded' ? snapshot.models : [];

    return models
      .filter(
        (model) => GENERATE_MODEL_TYPES.has(model.type) && matchesTerms(`${model.name} ${model.base}`, query.text)
      )
      .map<PaletteEntry>((model) => ({
        group: 'Models',
        id: `model:${model.key}`,
        isPersistentRecent: false,
        keywords: model.base,
        run: () => {
          openGenerateWidget();
          applyModel(model);
        },
        secondary: { label: 'Open in Model Manager', run: openModelManager },
        subtitle: getModelBaseLabel(model.base),
        title: model.name,
      }));
  },
});

const QUEUE_STATUS_LABELS: Record<QueueItemStatus, string> = {
  canceled: 'Canceled',
  completed: 'Completed',
  failed: 'Failed',
  in_progress: 'In progress',
  pending: 'Pending',
  waiting: 'Waiting',
};

export const createQueueItemsProvider = ({
  contextKey,
  loadQueue,
  openQueueWidget,
  revealItem,
}: {
  contextKey: string;
  loadQueue: () => Promise<QueueReadModel>;
  openQueueWidget: () => void;
  revealItem: (itemId: number) => void;
}): PaletteSearchProvider => ({
  contextKey,
  label: 'Queue items',
  providerKey: getPaletteContributionKey('provider', 'queue-items'),
  supportsCreatedAtRange: true,
  search: async (query) => {
    // The queue read model is a recent window with no server-side text search;
    // filter the fetched window client-side. Items without a timestamp fail
    // closed while a date range is active.
    const model = await loadQueue();

    return model.items
      .map((item) => ({ item, meta: extractGenerationMeta(item) }))
      .filter(
        ({ item, meta }) =>
          matchesTerms(
            `${meta.positivePrompt ?? ''} ${meta.negativePrompt ?? ''} ${QUEUE_STATUS_LABELS[item.status]} ${item.userDisplayName ?? ''}`,
            query.text
          ) &&
          (query.range === undefined || isTimestampInRange(item.createdAt, query.range))
      )
      .map<PaletteEntry>(({ item, meta }) => {
        const resultImageName = getResultImageName(item);

        return {
          group: 'Queue items',
          id: `queue-item:${item.id}`,
          isPersistentRecent: false,
          run: () => {
            openQueueWidget();
            revealItem(item.id);
          },
          subtitle: QUEUE_STATUS_LABELS[item.status],
          thumbnailUrl: resultImageName
            ? absolutizeApiUrl(`/api/v1/images/i/${encodeURIComponent(resultImageName)}/thumbnail`)
            : undefined,
          title: meta.positivePrompt?.trim() || 'Untitled generation',
        };
      });
  },
});

export const createImagesProvider = ({
  openGalleryWidget,
  openPreviewWidget,
  selectBoard,
  selectImage,
}: {
  openGalleryWidget: () => void;
  openPreviewWidget: () => void;
  selectBoard: (boardId: string) => void;
  selectImage: (image: GalleryImage) => void;
}): PaletteSearchProvider => ({
  contextKey: 'global',
  label: 'Images',
  providerKey: getPaletteContributionKey('provider', 'images'),
  supportsCreatedAtRange: true,
  search: async (query) => {
    // Searches every board the user can see (no board_id filter); the cached
    // boards list labels each result. Text and date range filter server-side.
    const [page, boards] = await Promise.all([
      queryClient.fetchQuery(
        galleryImagesOptions({
          createdFrom: query.range?.from,
          createdTo: query.range?.to,
          galleryView: 'images',
          limit: 20,
          searchTerm: query.text,
        })
      ),
      queryClient.fetchQuery(galleryBoardsOptions({ includeArchived: false })),
    ]);
    const boardNames = new Map(boards.map((board: GalleryBoard) => [board.id, board.name]));

    return page.images.map<PaletteEntry>((image) => ({
      group: 'Images',
      id: `image:${image.imageName}`,
      isPersistentRecent: false,
      run: () => {
        openPreviewWidget();
        selectImage(image);
      },
      secondary: {
        label: 'Reveal in Gallery',
        run: () => {
          openGalleryWidget();
          selectBoard(image.boardId);
          selectImage(image);
        },
      },
      subtitle: `${boardNames.get(image.boardId) ?? 'Uncategorized'} · ${image.width}×${image.height}`,
      thumbnailUrl: image.thumbnailUrl,
      title: formatIsoDate(image.queuedAt.slice(0, 10)),
    }));
  },
});
