import type { GalleryBoard } from '@features/gallery/contracts';
import type { PromptHistoryItem } from '@workbench/projectContracts';

import { galleryBoardsOptions } from '@features/gallery/queries';
import { focusPositivePrompt } from '@features/generation/react';
import { listLibraryWorkflows } from '@features/workflow/queries';
import { requestLibraryWorkflowLoad } from '@features/workflow/react';
import { queryClient } from '@platform/query/client';

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
