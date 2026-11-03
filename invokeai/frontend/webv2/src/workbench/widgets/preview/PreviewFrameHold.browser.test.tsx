/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop */
import type { StreamingImageSource } from '@platform/ui/streaming-image/streamingImageSource';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { PreviewFrame } from './PreviewFrame';

const i18n = createInstance();
void i18n.use(initReactI18next).init({ fallbackLng: 'en', initAsync: false, lng: 'en', resources: {} });

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;
let uniqueSource = 0;

/** A fresh URL each time: a data URL the page has already decoded loads synchronously. */
const createSource = (alt: string, kind: StreamingImageSource['kind']): StreamingImageSource => {
  uniqueSource += 1;

  return {
    alt,
    height: 8,
    kind,
    src: `data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8" data-n="${uniqueSource}"/>`,
    width: 8,
  };
};

const render = (element: ReactNode): void => {
  act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <DndContext>{element}</DndContext>
        </I18nextProvider>
      </ChakraProvider>
    );
  });
};

const settle = (): Promise<void> =>
  act(async () => {
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 100);
    });
  });

const frame = (source: StreamingImageSource, holdSource: StreamingImageSource | null, onSourceLoaded: () => void) => (
  <PreviewFrame
    frameHeight={8}
    frameWidth={8}
    holdSource={holdSource}
    isLive={false}
    shouldAntialiasLiveImage={false}
    source={{ itemKey: 'image:finished', kind: 'image', source }}
    variant="framed"
    onSourceLoaded={onSourceLoaded}
  />
);

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(async () => {
    root?.unmount();
    await Promise.resolve();
  });
  host?.remove();
  host = null;
  root = null;
});

describe('PreviewFrame hold', () => {
  it('paints the held frame over the finished image until it has decoded', async () => {
    const onSourceLoaded = vi.fn();
    const finished = createSource('finished', 'fallback');
    const held = createSource('held', 'live');

    render(frame(finished, held, onSourceLoaded));

    // Synchronously after the commit the finished image has not loaded: the
    // held frame is painted and the finished image only holds the geometry.
    const hold = host?.querySelector<HTMLImageElement>('img[aria-hidden="true"]');
    expect(hold?.getAttribute('src')).toBe(held.src);
    expect(host?.querySelector<HTMLImageElement>('img[alt="finished"]')?.style.visibility).toBe('hidden');
    expect(onSourceLoaded).not.toHaveBeenCalled();

    await settle();

    expect(host?.querySelector('img[aria-hidden="true"]')).toBeNull();
    expect(host?.querySelector<HTMLImageElement>('img[alt="finished"]')?.style.visibility).toBe('');
    expect(onSourceLoaded).toHaveBeenCalledWith(finished.src);
  });

  it('does not hold over an image that had already decoded', async () => {
    const onSourceLoaded = vi.fn();
    const finished = createSource('finished', 'fallback');

    render(frame(finished, null, onSourceLoaded));
    await settle();
    render(frame(finished, createSource('held', 'live'), onSourceLoaded));

    expect(host?.querySelector('img[aria-hidden="true"]')).toBeNull();
    expect(onSourceLoaded).toHaveBeenCalledWith(finished.src);
  });

  it('drops the hold when the finished image fails to load', async () => {
    const onSourceLoaded = vi.fn();
    const broken: StreamingImageSource = { alt: 'finished', height: 8, kind: 'fallback', src: 'data:,', width: 8 };

    render(frame(broken, createSource('held', 'live'), onSourceLoaded));
    await settle();

    expect(host?.querySelector('img[aria-hidden="true"]')).toBeNull();
  });
});
