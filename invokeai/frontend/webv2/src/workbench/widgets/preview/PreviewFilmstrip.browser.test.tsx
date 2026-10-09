import type * as dndKitModule from '@dnd-kit/core';
import type { GalleryImageItem, GalleryItem, GalleryItemKey } from '@features/gallery';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { PreviewFilmstrip } from './PreviewFilmstrip';

const renderedThumbs = vi.hoisted(() => [] as string[]);

// Every thumb render calls the drag hook once with its item's drag id, which names the thumb that rendered.
vi.mock('@dnd-kit/core', async (importOriginal) => {
  const actual = await importOriginal<typeof dndKitModule>();

  return {
    ...actual,
    useDraggable: (args: Parameters<typeof actual.useDraggable>[0]) => {
      renderedThumbs.push(String(args.id));

      return actual.useDraggable(args);
    },
  };
});

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const i18n = createInstance();
void i18n.use(initReactI18next).init({ fallbackLng: 'en', initAsync: false, lng: 'en', resources: {} });

const image = (index: number): GalleryImageItem => ({
  boardId: 'none',
  category: 'general',
  createdAt: '2026-07-30T12:00:00Z',
  fullUrl: `/images/${String(index)}/full`,
  height: 512,
  isIntermediate: false,
  kind: 'image',
  name: `image-${String(index)}`,
  starred: false,
  thumbnailUrl: `/images/${String(index)}/thumbnail`,
  width: 512,
});

const items = Array.from({ length: 24 }, (_, index) => image(index));
const onSelect = (_item: GalleryItem) => undefined;
const onCompare = (_item: GalleryItem) => undefined;
const onContextMenu = (_item: GalleryItem, _x: number, _y: number) => undefined;

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
  host = document.createElement('div');
  host.style.cssText = 'height:120px;width:640px;';
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  renderedThumbs.length = 0;
});

const renderStrip = (selectedItemKey: GalleryItemKey) =>
  act(() =>
    root.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <DndContext>
            <PreviewFilmstrip
              density="full"
              items={items}
              selectedItemKey={selectedItemKey}
              onCompare={onCompare}
              onContextMenu={onContextMenu}
              onSelect={onSelect}
            />
          </DndContext>
        </ChakraProvider>
      </I18nextProvider>
    )
  );

describe('PreviewFilmstrip selection moves', () => {
  it('re-renders only the thumbs that lose and gain the selection', async () => {
    await renderStrip('image:image-3' as GalleryItemKey);
    renderedThumbs.length = 0;

    await renderStrip('image:image-4' as GalleryItemKey);

    expect(new Set(renderedThumbs).size).toBe(2);
    expect(renderedThumbs.every((id) => id.includes('image-3') || id.includes('image-4'))).toBe(true);
    expect(host.querySelector('[aria-current="true"]')?.getAttribute('aria-label')).toBe('image-4');
  });
});
