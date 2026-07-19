import type { GeneratedImageContract } from '@features/gallery';
import type { WidgetRuntimeApi } from '@workbench/widgetContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act, useCallback, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { PreviewComparisonMode } from './previewSettings';

import { PreviewCompare } from './PreviewCompare';

const i18n = createInstance();
const commandHandlers = new Map<string, () => void>();
const previewRuntime = {
  commands: {
    register: ({ handler, id }: { handler: () => void; id: string }) => {
      commandHandlers.set(id, handler);
      return () => commandHandlers.delete(id);
    },
  },
  hotkeys: { register: () => () => undefined },
} as unknown as WidgetRuntimeApi;
void i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initImmediate: false,
  lng: 'en',
  resources: {
    en: {
      translation: {
        common: { swap: 'Swap' },
        widgets: {
          preview: {
            commands: { nextComparisonMode: 'Next comparison mode' },
            compare: 'Compare',
            exitCompare: 'Exit Compare',
            hover: 'Hover',
            hoverComparisonAriaLabel: 'Reveal comparison image on hover, focus, or touch',
            sideBySide: 'Side by Side',
            slider: 'Slider',
            viewing: 'Viewing',
          },
        },
      },
    },
  },
});

const createImage = (name: string, width: number, height: number): GeneratedImageContract => ({
  height,
  imageName: name,
  imageUrl: `data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}"/>`,
  queuedAt: '2026-07-16T00:00:00.000Z',
  sourceQueueItemId: 'queue-1',
  thumbnailUrl: '',
  width,
});

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const interact = (action: () => void): Promise<void> =>
  act(async () => {
    action();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 50);
    });
  });

const waitForMotion = (): Promise<void> =>
  act(async () => {
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 350);
    });
  });

const renderComparison = async ({
  baseImage = createImage('base', 1200, 800),
  initialMode = 'hover',
}: {
  baseImage?: GeneratedImageContract;
  initialMode?: PreviewComparisonMode;
} = {}) => {
  commandHandlers.clear();
  const onExit = vi.fn();
  const onModeChange = vi.fn();
  const onSwap = vi.fn();
  const compareImage = createImage('compare', 800, 1200);
  const Harness = () => {
    const [mode, setMode] = useState(initialMode);
    const handleModeChange = useCallback((nextMode: PreviewComparisonMode) => {
      setMode(nextMode);
      onModeChange(nextMode);
    }, []);

    return (
      <PreviewCompare
        baseImage={baseImage}
        compareImage={compareImage}
        mode={mode}
        runtime={previewRuntime}
        onExit={onExit}
        onModeChange={handleModeChange}
        onSwap={onSwap}
      />
    );
  };

  host = document.createElement('div');
  host.style.height = '500px';
  host.style.width = '800px';
  document.body.append(host);
  root = createRoot(host);
  await interact(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <Harness />
        </ChakraProvider>
      </I18nextProvider>
    );
  });

  return { commandHandlers, onExit, onModeChange, onSwap };
};

afterEach(async () => {
  await interact(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('PreviewCompare', () => {
  it('reveals the comparison on focus and touch press, then restores the selected image', async () => {
    await renderComparison();
    const frame = host!.querySelector<HTMLElement>('[aria-label*="Reveal comparison"]')!;
    const compareOverlay = host!.querySelector<HTMLImageElement>('img[alt="compare"]')?.parentElement as HTMLElement;

    expect(getComputedStyle(compareOverlay).opacity).toBe('0');
    await interact(() => frame.focus());
    await waitForMotion();
    expect(getComputedStyle(compareOverlay).opacity).toBe('1');
    await interact(() => frame.blur());
    await waitForMotion();
    expect(getComputedStyle(compareOverlay).opacity).toBe('0');

    Object.defineProperties(frame, {
      hasPointerCapture: { value: () => true },
      releasePointerCapture: { value: vi.fn() },
      setPointerCapture: { value: vi.fn() },
    });
    await interact(() =>
      frame.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, pointerId: 7, pointerType: 'touch' }))
    );
    await waitForMotion();
    expect(getComputedStyle(compareOverlay).opacity).toBe('1');
    await interact(() =>
      frame.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, pointerId: 7, pointerType: 'touch' }))
    );
    await waitForMotion();
    expect(getComputedStyle(compareOverlay).opacity).toBe('0');
    expect(getComputedStyle(compareOverlay).transitionProperty).toContain('opacity');
  });

  it('cycles all modes through the M command and keeps swap and exit separate', async () => {
    const { commandHandlers, onExit, onModeChange, onSwap } = await renderComparison({ initialMode: 'slider' });
    const nextMode = commandHandlers.get('viewer.nextComparisonMode')!;

    await interact(nextMode);
    expect(onModeChange).toHaveBeenLastCalledWith('side-by-side');
    await interact(nextMode);
    expect(onModeChange).toHaveBeenLastCalledWith('hover');
    await interact(nextMode);
    expect(onModeChange).toHaveBeenLastCalledWith('slider');

    const buttons = Array.from(host?.querySelectorAll<HTMLButtonElement>('button') ?? []);
    await interact(() => buttons.find((button) => button.textContent?.includes('Swap'))?.click());
    await interact(() => buttons.find((button) => button.textContent?.includes('Exit Compare'))?.click());
    expect(onSwap).toHaveBeenCalledOnce();
    expect(onExit).toHaveBeenCalledOnce();
  });

  it.each([
    ['portrait', 400, 1600],
    ['panorama', 1600, 400],
    ['square', 900, 900],
  ])('fits a %s image frame without overflow', async (_name, width, height) => {
    await renderComparison({ baseImage: createImage('base', width, height), initialMode: 'slider' });
    const frame = host?.querySelector<HTMLImageElement>('img[alt="base"]')?.parentElement as HTMLElement;
    const stage = frame.parentElement!;
    const frameRect = frame.getBoundingClientRect();
    const stageRect = stage.getBoundingClientRect();

    expect(frameRect.width).toBeLessThanOrEqual(stageRect.width);
    expect(frameRect.height).toBeLessThanOrEqual(stageRect.height);
    expect(frameRect.width / frameRect.height).toBeCloseTo(width / height, 1);
  });
});
