/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GalleryItem, GalleryItemRef } from '@features/gallery/core/items';

import { ChakraProvider } from '@chakra-ui/react';
import { EMPTY_GALLERY_MARKS_FILTER, getGalleryItemMarks, type GalleryMarksFilter } from '@features/gallery/core/marks';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { GalleryCullCommands, type GalleryCullStepper, type GalleryCullTarget } from './GalleryCullCommands';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, unknown>) => (values ? `${key}:${JSON.stringify(values)}` : key),
  }),
}));

const createItem = (name: string, overrides: Partial<GalleryItem> = {}): GalleryItem =>
  ({
    boardId: 'none',
    colorLabel: null,
    kind: 'image',
    name,
    rating: 0,
    rejected: false,
    starred: false,
    ...overrides,
  }) as GalleryItem;

const toTarget = (...items: GalleryItem[]): GalleryCullTarget => ({
  item: items.length === 1 ? items[0]! : null,
  marks: items.map(getGalleryItemMarks),
  primary: getGalleryItemMarks(items.at(-1)!),
  refs: items.map(({ kind, name }) => ({ kind, name })),
});

const handlers = new Map<string, (press?: unknown) => unknown>();
const runtime = {
  commands: {
    register: ({ handler, id }: { handler: (press?: unknown) => unknown; id: string }) => {
      handlers.set(id, handler);
      return () => handlers.delete(id);
    },
  },
  hotkeys: { activateMode: vi.fn(() => () => undefined) },
};
const onMark = vi.fn<(refs: GalleryItemRef[], patch: unknown) => void>();
const go = vi.fn();
let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

/** A listing of `names` in order; the stepper reads it at the press, as the surfaces do. */
const render = async ({
  autoAdvance = false,
  marksFilter = EMPTY_GALLERY_MARKS_FILTER,
  names = ['a', 'b', 'c'],
  target,
}: {
  autoAdvance?: boolean;
  marksFilter?: GalleryMarksFilter;
  names?: string[];
  target: GalleryCullTarget | null;
}) => {
  const stepper: GalleryCullStepper<string> = {
    go,
    peek: (fromKey, direction) => {
      const index = names.findIndex((name) => `image:${name}` === fromKey);

      return index < 0 ? null : (names[index + (direction === 'right' ? 1 : -1)] ?? null);
    },
  };

  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <GalleryCullCommands
          autoAdvance={autoAdvance}
          enabled
          marksFilter={marksFilter}
          runtime={runtime}
          stepper={stepper}
          target={target}
          onMark={onMark}
          onToggle={vi.fn()}
        />
      </ChakraProvider>
    )
  );
};

const press = (id: string, shiftKey = false) => act(() => void handlers.get(id)?.({ shiftKey }));
const announcements = () => [...(host?.querySelectorAll('[role="status"]') ?? [])].map((node) => node.textContent);

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  handlers.clear();
  onMark.mockClear();
  go.mockClear();
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('Cull keys', () => {
  it('turn the mode on and mark the whole selection, announcing how many items', async () => {
    const a = createItem('a', { rating: 4 });
    const b = createItem('b', { rating: 1 });

    await render({ autoAdvance: true, target: toTarget(a, b) });
    await press('cull.raiseRating');

    expect(runtime.hotkeys.activateMode).toHaveBeenCalledWith({ mode: 'cull' });
    // `]` steps from the primary item, the last added; several targets never advance.
    expect(onMark).toHaveBeenCalledExactlyOnceWith(
      [
        { kind: 'image', name: 'a' },
        { kind: 'image', name: 'b' },
      ],
      { rating: 2 }
    );
    expect(go).not.toHaveBeenCalled();
    expect(announcements()).toContain(
      'widgets.gallery.cull.feedback.forItems:{"count":2,"mark":"widgets.gallery.marks.ratingSummary:{\\"count\\":2}"}'
    );
  });

  it('moves on from the item shown at the press, before marking it', async () => {
    await render({ autoAdvance: true, target: toTarget(createItem('a')) });
    await render({ autoAdvance: true, target: toTarget(createItem('b')) });
    await press('cull.rate3');

    expect(go).toHaveBeenCalledExactlyOnceWith('c', 'right');
    expect(onMark).toHaveBeenCalledExactlyOnceWith([{ kind: 'image', name: 'b' }], { rating: 3 });
    expect(go.mock.invocationCallOrder[0]).toBeLessThan(onMark.mock.invocationCallOrder[0] ?? 0);
  });

  it('hands the selection to the replacement of an item leaving the listing, or at its end to the one before', async () => {
    const unflagged = { ...EMPTY_GALLERY_MARKS_FILTER, flags: ['none' as const] };

    await render({ marksFilter: unflagged, target: toTarget(createItem('b')) });
    await press('cull.reject');
    expect(go).toHaveBeenLastCalledWith('c', 'right');

    await render({ marksFilter: unflagged, target: toTarget(createItem('c')) });
    await press('cull.reject');
    expect(go).toHaveBeenLastCalledWith('b', 'left');
  });

  it('keeps the selection on an item that only moves between the Bookmarked strip and the grid', async () => {
    await render({ target: toTarget(createItem('b')) });
    await press('cull.bookmark');

    expect(onMark).toHaveBeenCalledExactlyOnceWith([{ kind: 'image', name: 'b' }], { flag: 'pick' });
    expect(go).not.toHaveBeenCalled();
  });

  it('say there is nothing to mark without a target', async () => {
    await render({ target: null });
    await press('cull.rate1');

    expect(onMark).not.toHaveBeenCalled();
    expect(announcements()).toContain('widgets.gallery.cull.feedback.nothingToMark');
  });
});
