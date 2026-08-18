import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { createStore } from 'app/store/store';
import { boardIdSelected, searchTermChanged } from 'features/gallery/store/gallerySlice';
import { galleryApi } from 'services/api/endpoints/gallery';
import { describe, expect, it } from 'vitest';

import { selectCachedGalleryItemNames } from './selectCachedGalleryItemNames';

const readSource = (relativePath: string) =>
  readFileSync(fileURLToPath(new URL(relativePath, import.meta.url)), 'utf8');

describe('virtual board gallery consumers', () => {
  it('invalidates virtual-board and uncategorized caches after deleting a board with its media', () => {
    const source = readSource('../../../services/api/endpoints/images.ts');
    const deleteBoardAndImages = source.slice(
      source.indexOf('deleteBoardAndImages:'),
      source.indexOf('addImageToBoard:')
    );

    expect(deleteBoardAndImages).toContain("'VirtualBoards'");
    expect(deleteBoardAndImages).toContain("getTagsToInvalidateForBoardAffectingMutation(['none'])");
  });

  it.each([
    ['range selection', './selectCachedGalleryItemNames.ts'],
    ['board auto-selection', '../../../app/store/middleware/listenerMiddleware/listeners/boardIdSelected.ts'],
    ['initial board auto-selection', '../../../app/store/middleware/listenerMiddleware/listeners/appStarted.ts'],
  ])('%s reads the polymorphic item-name cache', (_label, relativePath) => {
    const source = readSource(relativePath);

    // Regular boards and virtual dates share one endpoint; a consumer that reached for a
    // different cache would silently see an empty list on virtual dates.
    expect(source).toContain('listGalleryItemNames');
  });

  it('keeps the prior virtual-board cache available during filter debounce', async () => {
    const store = createStore();
    store.dispatch(boardIdSelected({ boardId: 'by_date:2026-07-26' }));
    await store.dispatch(
      galleryApi.util.upsertQueryData(
        'listGalleryItemNames',
        {
          created_date: '2026-07-26',
          categories: ['general'],
          order_dir: 'DESC',
          starred_first: true,
          is_intermediate: false,
        },
        { item_names: ['cached.mp4'], starred_count: 0, total_count: 1 }
      )
    );

    store.dispatch(searchTermChanged('new filter'));

    expect(selectCachedGalleryItemNames(store.getState())).toEqual(['cached.mp4']);
  });

  it('prefers the most recently active same-date cache during filter debounce', async () => {
    const store = createStore();
    store.dispatch(boardIdSelected({ boardId: 'by_date:2026-07-26' }));
    await store.dispatch(
      galleryApi.util.upsertQueryData(
        'listGalleryItemNames',
        {
          created_date: '2026-07-26',
          categories: ['general'],
          search_term: 'older filter',
          order_dir: 'DESC',
          starred_first: true,
          is_intermediate: false,
        },
        { item_names: ['older.mp4'], starred_count: 0, total_count: 1 }
      )
    );
    await new Promise<void>((resolve) => {
      setTimeout(resolve, 5);
    });
    await store.dispatch(
      galleryApi.util.upsertQueryData(
        'listGalleryItemNames',
        {
          created_date: '2026-07-26',
          categories: ['general'],
          order_dir: 'DESC',
          starred_first: true,
          is_intermediate: false,
        },
        { item_names: ['active.mp4'], starred_count: 0, total_count: 1 }
      )
    );

    store.dispatch(searchTermChanged('new filter'));

    expect(selectCachedGalleryItemNames(store.getState())).toEqual(['active.mp4']);
  });
});
