import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { STALE_AFTER_MS } from '@workbench/image-map/indexProgress';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ImageIndexProgressInline, ImageIndexProgressPanel } from './ImageIndexProgress';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const settle = (action: () => void): Promise<void> =>
  act(async () => {
    action();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 40);
    });
  });

type Counts = { embedded: number; failed: number; pending: number; total: number };

// Hoisted: an object or function literal in a JSX attribute is what
// react-perf/jsx-no-new-*-as-prop forbids, tests included.
const noop = () => {};
const QUARTER: Counts = { embedded: 25, failed: 0, pending: 75, total: 100 };
const PART_WAY: Counts = { embedded: 1204, failed: 0, pending: 3108, total: 4312 };
const SIX_DIGITS: Counts = { embedded: 123456, failed: 0, pending: 60544, total: 184000 };
const NARROW_HOST = { display: 'flex', width: '120px' } as const;
const NARROW_HOST_280 = { display: 'flex', width: '280px' } as const;
const LONG_ERROR =
  'sqlite3.OperationalError: no such column: image_index_embeddings.model_key_normalized_v2_with_a_long_suffix';

const mount = async (element: React.ReactElement, width?: string) => {
  host = document.createElement('div');

  if (width) {
    host.style.width = width;
  }

  document.body.append(host);
  root = createRoot(host);

  await settle(() => {
    root?.render(<ChakraProvider value={system}>{element}</ChakraProvider>);
  });
};

const render = (counts: Counts, updatedAt: number | null = null) =>
  mount(<ImageIndexProgressPanel counts={counts} updatedAt={updatedAt} onRetry={noop} />);

afterEach(async () => {
  await settle(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('ImageIndexProgressPanel', () => {
  it('exposes the share done to assistive tech, not just as a painted width', async () => {
    await render({ embedded: 25, failed: 0, pending: 75, total: 100 });

    const bar = host!.querySelector('[role="progressbar"]')!;

    expect(bar.getAttribute('aria-valuenow')).toBe('25');
    expect(bar.getAttribute('aria-valuemax')).toBe('100');
    expect(bar.getAttribute('aria-label')).toBe('Image indexing progress');
  });

  it('rounds the announced value instead of reading out sixteen digits', async () => {
    await render(PART_WAY);

    expect(host!.querySelector('[role="progressbar"]')!.getAttribute('aria-valuenow')).toBe('28');
  });

  it('shows the counts and the percentage', async () => {
    await render({ embedded: 25, failed: 0, pending: 75, total: 100 });

    const text = host!.textContent ?? '';

    expect(text).toContain('25 of 100 images');
    expect(text).toContain('25%');
  });

  it('says nothing about staleness while reports are still arriving', async () => {
    await render({ embedded: 25, failed: 0, pending: 75, total: 100 }, Date.now());

    expect(host!.textContent).not.toContain('No progress reported');
  });

  it('notices the silence itself, without waiting for an event', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });

    try {
      await render({ embedded: 25, failed: 0, pending: 75, total: 100 }, Date.now());
      expect(host!.textContent).not.toContain('No progress reported');

      await act(async () => {
        await vi.advanceTimersByTimeAsync(STALE_AFTER_MS + 10_000);
      });

      expect(host!.textContent).toContain('No progress reported for');
    } finally {
      vi.useRealTimers();
    }
  });

  it('states the fact without claiming to know the cause', async () => {
    await render({ embedded: 25, failed: 0, pending: 75, total: 100 }, Date.now() - 10 * 60_000);

    const text = host!.textContent ?? '';

    // An indexer waiting out a generation and one that has died look
    // identical from here, and the first is routine.
    expect(text).toContain('No progress reported for 10m 00s');
    expect(text).not.toContain('queue');
    expect(text).not.toContain('Paused');
  });

  it('accounts for images that were given up on', async () => {
    await render({ embedded: 25, failed: 4, pending: 71, total: 100 });

    expect(host!.textContent).toContain('4 skipped after repeated failures');
  });

  it('states a failed refresh instead of letting the branch it precedes swallow it', async () => {
    const onRetry = vi.fn();

    await mount(
      <ImageIndexProgressPanel
        counts={QUARTER}
        error="Failed to load the image map"
        updatedAt={null}
        onRetry={onRetry}
      />
    );

    const text = host!.textContent ?? '';

    // Both, not one or the other: the backfill is still worth watching.
    expect(text).toContain('Failed to load the image map');
    expect(text).toContain('25 of 100 images');

    const retry = [...host!.querySelectorAll('button')].find((element) => element.textContent === 'Retry')!;

    await settle(() => retry.click());
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it('keeps a server error message inside the panel at the minimum widget width', async () => {
    await mount(
      <div style={NARROW_HOST_280}>
        <ImageIndexProgressPanel counts={QUARTER} error={LONG_ERROR} updatedAt={null} onRetry={noop} />
      </div>,
      '280px'
    );

    // A URL or a dotted identifier has no break opportunity, and the widget
    // body is clipped: without `overflowWrap` it runs off both edges.
    expect(host!.scrollWidth).toBeLessThanOrEqual(host!.clientWidth);
  });

  it('offers a way out when the counts themselves are what is stuck', async () => {
    const onRetry = vi.fn();

    await mount(<ImageIndexProgressPanel counts={QUARTER} updatedAt={null} onRetry={onRetry} />);

    const button = [...host!.querySelectorAll('button')].find((element) => element.textContent === 'Check again')!;

    await settle(() => button.click());
    expect(onRetry).toHaveBeenCalledTimes(1);
  });
});

describe('ImageIndexProgressInline', () => {
  it('stays inside a footer narrower than its content', async () => {
    // The widget resizes down to 280px and the counts can be six digits each.
    // `<Text truncate>` is what kept the line this replaced from pushing past
    // the panel edge and under the refresh button.
    await mount(
      <div style={NARROW_HOST}>
        <ImageIndexProgressInline counts={SIX_DIGITS} updatedAt={Date.now()} />
      </div>,
      '120px'
    );

    expect(host!.scrollWidth).toBeLessThanOrEqual(host!.clientWidth);
  });

  it('labels its bar and keeps the counts compact', async () => {
    await mount(<ImageIndexProgressInline counts={PART_WAY} updatedAt={Date.now()} />);

    expect(host!.textContent).toContain(`indexing ${(1204).toLocaleString()}/${(4312).toLocaleString()}`);
    expect(host!.querySelector('[role="progressbar"]')!.getAttribute('aria-valuenow')).toBe('28');
  });

  it('carries the counts in the accessible name, which truncation cannot take away', async () => {
    // At the minimum widget width the visible label is down to a couple of
    // characters and the tooltip that carries the rest is hover-only.
    await mount(
      <div style={NARROW_HOST}>
        <ImageIndexProgressInline counts={SIX_DIGITS} updatedAt={Date.now()} />
      </div>,
      '120px'
    );

    const name = host!.querySelector('[role="progressbar"]')!.getAttribute('aria-label') ?? '';

    expect(name).toContain('Image indexing progress');
    expect(name).toContain(`${(123456).toLocaleString()} of ${(184000).toLocaleString()} images`);
  });

  it('keeps the bar at full size when the label has to shrink', async () => {
    await mount(
      <div style={NARROW_HOST}>
        <ImageIndexProgressInline counts={SIX_DIGITS} updatedAt={Date.now()} />
      </div>,
      '120px'
    );

    // The bar is the glanceable half; letting the flex squash it would leave
    // the row saying nothing at all.
    expect(host!.querySelector('[role="progressbar"]')!.getBoundingClientRect().width).toBeGreaterThanOrEqual(38);
  });
});
