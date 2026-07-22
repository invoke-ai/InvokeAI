import type { GalleryBoard, GalleryImage } from '@features/gallery/contracts';
import type { ModelConfig } from '@features/models';
import type { QueueItemStatus, QueueQueryScope } from '@features/queue/contracts';
import type { PromptHistoryItem } from '@workbench/projectContracts';

import { galleryBoardsOptions, galleryImagesOptions } from '@features/gallery/queries';
import { focusPositivePrompt } from '@features/generation/react';
import { ensureModelsLoaded, getModelBaseLabel, getModelsSnapshot } from '@features/models';
import { extractGenerationMeta, getResultImageName } from '@features/queue/contracts';
import { getQueueReadModelOptions } from '@features/queue/queries';
import { listLibraryWorkflows } from '@features/workflow/queries';
import { requestLibraryWorkflowLoad } from '@features/workflow/react';
import { queryClient } from '@platform/query/client';
import { absolutizeApiUrl } from '@platform/transport/http';

import type { PaletteEntry, PaletteSearchProvider } from './entries';

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
  id: 'workflows',
  label: 'Workflows',
  search: async (query) => {
    const [user, defaults] = await Promise.all([
      listLibraryWorkflows({ category: 'user', page: 0, perPage: PROVIDER_PAGE_SIZE, query }),
      listLibraryWorkflows({ category: 'default', page: 0, perPage: PROVIDER_PAGE_SIZE, query }),
    ]);

    return [...user.items, ...defaults.items].map<PaletteEntry>((item) => ({
      group: 'Workflows',
      id: `workflow:${item.workflow_id}`,
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
  id: 'boards',
  label: 'Boards',
  search: async (query) => {
    const boards = await queryClient.fetchQuery(galleryBoardsOptions({ includeArchived: false }));

    return boards
      .filter((board: GalleryBoard) => matchesTerms(board.name, query))
      .map<PaletteEntry>((board) => ({
        group: 'Boards',
        id: `board:${board.id}`,
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
  getPromptHistory,
  openGenerateWidget,
  recallPrompt,
}: {
  getPromptHistory: () => readonly PromptHistoryItem[];
  openGenerateWidget: () => void;
  recallPrompt: (item: PromptHistoryItem) => void;
}): PaletteSearchProvider => ({
  id: 'prompt-history',
  label: 'Prompt history',
  search: (query) => {
    const seen = new Set<string>();
    const entries: PaletteEntry[] = [];

    for (const [index, item] of getPromptHistory().entries()) {
      const dedupeKey = `${item.positivePrompt}\n${item.negativePrompt ?? ''}`;

      if (seen.has(dedupeKey) || !matchesTerms(`${item.positivePrompt} ${item.negativePrompt ?? ''}`, query)) {
        continue;
      }

      seen.add(dedupeKey);
      entries.push({
        group: 'Prompt history',
        id: `prompt:${index}`,
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
  id: 'models',
  label: 'Models',
  search: async (query) => {
    await ensureModelsLoaded();
    const snapshot = getModelsSnapshot();
    const models = snapshot.status === 'loaded' ? snapshot.models : [];

    return models
      .filter((model) => GENERATE_MODEL_TYPES.has(model.type) && matchesTerms(`${model.name} ${model.base}`, query))
      .map<PaletteEntry>((model) => ({
        group: 'Models',
        id: `model:${model.key}`,
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
  getScope,
  openQueueWidget,
  revealItem,
}: {
  getScope: () => QueueQueryScope;
  openQueueWidget: () => void;
  revealItem: (itemId: number) => void;
}): PaletteSearchProvider => ({
  id: 'queue-items',
  label: 'Queue items',
  search: async (query) => {
    // The queue read model is a recent window with no server-side text search;
    // filter the fetched window client-side.
    const model = await queryClient.fetchQuery(getQueueReadModelOptions(getScope()));

    return model.items
      .map((item) => ({ item, meta: extractGenerationMeta(item) }))
      .filter(({ item, meta }) =>
        matchesTerms(
          `${meta.positivePrompt ?? ''} ${meta.negativePrompt ?? ''} ${QUEUE_STATUS_LABELS[item.status]} ${item.userDisplayName ?? ''}`,
          query
        )
      )
      .map<PaletteEntry>(({ item, meta }) => {
        const resultImageName = getResultImageName(item);

        return {
          group: 'Queue items',
          id: `queue-item:${item.id}`,
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
  getSelectedBoardId,
  openGalleryWidget,
  openPreviewWidget,
  selectBoard,
  selectImage,
}: {
  getSelectedBoardId: () => string;
  openGalleryWidget: () => void;
  openPreviewWidget: () => void;
  selectBoard: (boardId: string) => void;
  selectImage: (image: GalleryImage) => void;
}): PaletteSearchProvider => ({
  id: 'images',
  label: 'Images',
  search: async (query) => {
    // The images endpoint is board-scoped; the palette searches the board the
    // gallery is currently on.
    const page = await queryClient.fetchQuery(
      galleryImagesOptions({
        boardId: getSelectedBoardId(),
        galleryView: 'images',
        limit: 20,
        searchTerm: query,
      })
    );

    return page.images.map<PaletteEntry>((image) => ({
      group: 'Images',
      id: `image:${image.imageName}`,
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
      subtitle: `${image.width}×${image.height}`,
      thumbnailUrl: image.thumbnailUrl,
      title: image.imageName,
    }));
  },
});
