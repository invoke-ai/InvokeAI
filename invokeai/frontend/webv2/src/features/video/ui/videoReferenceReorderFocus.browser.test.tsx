import type { VideoReferenceItem } from '@features/video/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { system } from '@theme/system';
import i18next from 'i18next';
import { act, useCallback, useEffect, useMemo, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { VideoReferenceListField } from './VideoReferenceListField';
import { VideoUiProvider, type VideoUiAdapter } from './VideoUiContext';

/** Reordering must preserve card identity and transfer focus when the pressed arrow becomes disabled. */
const i18n = i18next.createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  lng: 'en',
  resources: {
    en: {
      translation: {
        widgets: {
          gallery: { picker: { dropHint: 'Drop', upload: 'Upload' } },
          video: {
            addReference: 'Add reference',
            chooseReference: 'Choose from Gallery',
            dropReference: 'Drop Reference',
            moveReferenceDown: 'Move reference down',
            moveReferenceUp: 'Move reference up',
            referenceDetailMatch: 'Match generation size',
            referenceDetailMax: 'Max detail',
            referencesHelp: 'help',
            removeReference: 'Remove reference',
            uploadImageReference: 'Upload image',
            uploadVideoReference: 'Upload video',
          },
        },
      },
    },
  },
});

let host: HTMLDivElement;
let root: Root;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const adapter = {
  getUploadBoardId: () => 'none',
  patchValues: vi.fn(),
  reportError: vi.fn(),
  touchGalleryImages: vi.fn(),
  videoSpanPlayback: { getState: () => null, subscribe: () => () => undefined },
} as unknown as VideoUiAdapter;

const imageReference = (name: string): VideoReferenceItem => ({
  detail: 'match',
  image: { height: 512, image_name: name, width: 512 },
  kind: 'image',
});

const Harness = ({ names }: { names: string[] }) => {
  const [references, setReferences] = useState(() => names.map(imageReference));
  const handleChange = useCallback((update: (current: VideoReferenceItem[]) => VideoReferenceItem[]) => {
    setReferences((current) => update(current));
  }, []);

  return (
    <>
      <input aria-label="Prompt" />
      <VideoReferenceListField
        maxImages={9}
        maxVideos={3}
        references={references}
        targetArea={null}
        onChange={handleChange}
      />
    </>
  );
};

const render = async (names: string[]) => {
  await act(() =>
    root.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <DndContext>
            <VideoUiProvider adapter={adapter}>
              <Harness names={names} />
            </VideoUiProvider>
          </DndContext>
        </ChakraProvider>
      </I18nextProvider>
    )
  );
};

const buttons = (label: string): HTMLButtonElement[] =>
  [...document.querySelectorAll(`button[aria-label="${label}"]`)] as HTMLButtonElement[];

/** The thumbnails ARE the rendered order. */
const order = (): string[] => [...document.querySelectorAll('img')].map((image) => image.src.split('/').at(-2) ?? '');

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

describe('video reference reorder focus', () => {
  it('keeps focus on the pressed arrow through an ordinary move', async () => {
    await render(['a.png', 'b.png', 'c.png']);
    expect(order()).toEqual(['a.png', 'b.png', 'c.png']);

    const up = buttons('Move reference up')[2];

    await act(() => up?.focus());
    await act(() => up?.click());

    expect(order()).toEqual(['a.png', 'c.png', 'b.png']);

    expect(document.activeElement).toBe(up);
    expect(buttons('Move reference up')[1]).toBe(up);
  });

  it('hands focus to the sibling arrow when the move lands on an end', async () => {
    await render(['a.png', 'b.png', 'c.png']);

    const up = buttons('Move reference up')[1];

    await act(() => up?.focus());
    await act(() => up?.click());

    expect(order()).toEqual(['b.png', 'a.png', 'c.png']);
    expect(up?.disabled).toBe(true);
    expect(document.activeElement).toBe(buttons('Move reference down')[0]);
    expect((document.activeElement as HTMLButtonElement).disabled).toBe(false);
  });

  it('leaves focus alone when the arrow was pressed without holding focus', async () => {
    // Some browsers dispatch clicks without focusing the button; preserve focus in the prompt.
    await render(['a.png', 'b.png', 'c.png']);

    const field = document.querySelector('[aria-label="Prompt"]') as HTMLInputElement;
    const up = buttons('Move reference up')[1];

    await act(() => field.focus());
    await act(() => up?.click());

    expect(order()).toEqual(['b.png', 'a.png', 'c.png']);
    expect(document.activeElement).toBe(field);
  });

  it('never fires a stale handoff after a dropped write', async () => {
    // A dropped write must not leave an arm that steals focus when a later change disables the arrow.
    let drop = true;
    let mutate: (next: string[]) => void = () => undefined;

    const StaleHarness = () => {
      const [names, setNames] = useState(['a.png', 'b.png', 'c.png']);

      useEffect(() => {
        mutate = setNames;
      }, []);

      const references = useMemo(() => names.map(imageReference), [names]);
      const handleChange = useCallback((update: (current: VideoReferenceItem[]) => VideoReferenceItem[]) => {
        if (drop) {
          return;
        }
        setNames((current) =>
          update(current.map(imageReference)).map((reference) =>
            reference.kind === 'image' ? reference.image.image_name : ''
          )
        );
      }, []);

      return (
        <>
          <input aria-label="Prompt" />
          <VideoReferenceListField
            maxImages={9}
            maxVideos={3}
            references={references}
            targetArea={null}
            onChange={handleChange}
          />
        </>
      );
    };

    await act(() =>
      root.render(
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <DndContext>
              <VideoUiProvider adapter={adapter}>
                <StaleHarness />
              </VideoUiProvider>
            </DndContext>
          </ChakraProvider>
        </I18nextProvider>
      )
    );

    const up = buttons('Move reference up')[1];

    await act(() => up?.focus());
    await act(() => up?.click());

    // Dropped: nothing moved, and the arm is still sitting on the card.
    expect(order()).toEqual(['a.png', 'b.png', 'c.png']);

    const field = document.querySelector('[aria-label="Prompt"]') as HTMLInputElement;

    await act(() => field.focus());

    drop = false;
    await act(() => mutate(['b.png', 'c.png']));

    expect(order()).toEqual(['b.png', 'c.png']);
    expect(buttons('Move reference up')[0]?.disabled).toBe(true);
    expect(document.activeElement).toBe(field);
  });

  it('keeps focus when a card moves PAST a duplicate of another reference', async () => {
    // Repeated clips use occurrence keys; moving tail past the second duplicate must preserve tail's identity.
    await render(['dup.png', 'tail.png', 'dup.png', 'end.png']);

    const down = buttons('Move reference down')[1];

    await act(() => down?.focus());
    await act(() => down?.click());

    expect(order()).toEqual(['dup.png', 'dup.png', 'tail.png', 'end.png']);
    expect(document.activeElement).toBe(down);
    expect(buttons('Move reference down')[2]).toBe(down);
  });
});
