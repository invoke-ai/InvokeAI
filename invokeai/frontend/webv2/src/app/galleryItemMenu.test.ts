import type { GalleryItem, GalleryItemRef } from '@features/gallery/contracts';

import { describe, expect, it, vi } from 'vitest';

import { createGalleryItemMenu, type LoadedGalleryItemMenu } from './galleryItemMenu';

const Host = () => null;
const first = { kind: 'image', name: 'first.png' } as const;
const second = { kind: 'video', name: 'second.mp4' } as const;
const third = { kind: 'image', name: 'third.png' } as const;

const createMenu = () => {
  const interactions = new EventTarget();
  const loads = new Map<string, { promise: Promise<LoadedGalleryItemMenu>; resolve: () => void }>();
  const menu = createGalleryItemMenu({
    interactions,
    load: (ref) => {
      let resolve = () => {};
      const promise = new Promise<LoadedGalleryItemMenu>((settle) => {
        resolve = () => settle({ Host, item: ref as unknown as GalleryItem });
      });
      loads.set(ref.name, { promise, resolve });
      return promise;
    },
  });
  const settle = async (ref: GalleryItemRef) => {
    const load = loads.get(ref.name)!;
    load.resolve();
    await load.promise;
  };

  return { interactions, menu, settle };
};

const options = (isCurrent = () => true) => ({ isCurrent, onFailure: vi.fn() });

describe('gallery item menu', () => {
  it('shows only the newest request, and ignores a late close from the menu it replaced', async () => {
    const { menu, settle } = createMenu();
    const onFailure = vi.fn();

    menu.open({ ref: first, x: 1, y: 1 }, { isCurrent: () => true, onFailure });
    menu.open({ ref: second, x: 2, y: 2 }, { isCurrent: () => true, onFailure });
    await settle(second);
    await settle(first);

    expect(menu.getSnapshot().view?.target).toMatchObject({ itemRefs: [second], items: [second], x: 2, y: 2 });

    const secondViewId = menu.getSnapshot().view!.id;
    menu.open({ ref: third, x: 3, y: 3 }, { isCurrent: () => true, onFailure });
    await settle(third);
    menu.close(secondViewId);

    expect(menu.getSnapshot().view?.target.itemRefs).toEqual([third]);
    expect(onFailure).not.toHaveBeenCalled();
  });

  it.each([
    ['a click', new Event('pointerdown')],
    ['Escape', Object.assign(new Event('keydown'), { key: 'Escape' })],
  ])('drops a request still loading when the user moves on with %s', async (_, interaction) => {
    const { interactions, menu, settle } = createMenu();

    menu.open({ ref: first, x: 1, y: 1 }, options());
    interactions.dispatchEvent(interaction);
    await settle(first);

    expect(menu.getSnapshot().view).toBeNull();
  });

  it('drops a request whose project is no longer the active one when it is ready', async () => {
    const { menu, settle } = createMenu();

    menu.open(
      { ref: first, x: 1, y: 1 },
      options(() => false)
    );
    await settle(first);

    expect(menu.getSnapshot().view).toBeNull();
  });
});
