import type { StreamingImageSource } from '@platform/ui/streaming-image/streamingImageSource';
import type { CSSProperties } from 'react';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it } from 'vitest';

import { PreviewFrame } from './PreviewFrame';

const i18n = createInstance();
void i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  lng: 'en',
  resources: {
    en: { translation: { widgets: { preview: { dropToCompare: 'Drop to compare', resetZoom: 'Reset zoom' } } } },
  },
});

/**
 * What `CenterArea` publishes: the island band's own 8px inset, its 32px
 * height, and an 8px gap under it.
 */
const CENTER_CHROME_INSET_PX = 48;
const STAGE_PADDING = '6';

/** The right panel, where nothing floats and the variable is never published. */
const PLAIN_REGION_STYLE: CSSProperties = { display: 'flex', height: '320px', width: '320px' };
/** The centre region, standing in for what `CenterArea` puts on its root. */
const CHROME_REGION_STYLE = { ...PLAIN_REGION_STYLE, '--wb-center-chrome-inset': '3rem' } as CSSProperties;

const source: StreamingImageSource = {
  alt: 'preview.png',
  height: 128,
  kind: 'fallback',
  src: 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128"/>',
  width: 128,
};

const MEDIA_SOURCE = { itemKey: 'image:preview.png', kind: 'image', source } as const;

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const settle = (action: () => void): Promise<void> =>
  act(async () => {
    action();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 50);
    });
  });

/** Renders the framed stage, optionally under an ancestor that publishes the inset. */
const render = async (hasFloatingChrome: boolean) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await settle(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <DndContext>
            <div data-testid="region" style={hasFloatingChrome ? CHROME_REGION_STYLE : PLAIN_REGION_STYLE}>
              <PreviewFrame
                frameHeight={128}
                frameWidth={128}
                isLive={false}
                liveBadgeLabel="Generating"
                padding={STAGE_PADDING}
                shouldAntialiasLiveImage
                source={MEDIA_SOURCE}
                variant="framed"
              />
            </div>
          </DndContext>
        </ChakraProvider>
      </I18nextProvider>
    );
  });
};

/** How far the fitted media sits below the top of the grid surface. */
const mediaOffsetFromStageTop = (): number => {
  const stage = host!.querySelector<HTMLElement>('[data-testid="region"]')!.firstElementChild!;
  const image = host!.querySelector<HTMLImageElement>('img[alt="preview.png"]')!;

  return image.getBoundingClientRect().top - stage.getBoundingClientRect().top;
};

afterEach(async () => {
  await settle(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('preview frame chrome clearance', () => {
  it('drops the fitted media clear of the centre region chrome islands', async () => {
    // The islands float over the work surface rather than reserving a row, so
    // a height-constrained image used to run under them at fit zoom.
    await render(true);

    expect(mediaOffsetFromStageTop()).toBeGreaterThanOrEqual(CENTER_CHROME_INSET_PX);
  });

  it('reserves nothing where no chrome floats, so the panel keeps its own padding', async () => {
    // The other half of the contract: outside the centre region the variable is
    // undefined and the widget gets a real header row instead, so the `0px`
    // fallback has to leave the stage on its plain padding.
    await render(false);

    const offset = mediaOffsetFromStageTop();

    expect(offset).toBeGreaterThan(0);
    expect(offset).toBeLessThan(CENTER_CHROME_INSET_PX);
  });

  it('keeps the dot grid running to the top edge behind the islands', async () => {
    // The clearance is padding *inside* the grid surface, not a gap above it —
    // the grid is the widget floor and still passes behind the chrome.
    await render(true);

    const region = host!.querySelector<HTMLElement>('[data-testid="region"]')!;
    const stage = region.firstElementChild!;

    expect(stage.getBoundingClientRect().top).toBeCloseTo(region.getBoundingClientRect().top, 1);
  });
});
