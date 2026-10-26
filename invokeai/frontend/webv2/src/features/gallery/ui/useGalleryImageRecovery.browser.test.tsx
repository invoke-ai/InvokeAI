import { accountLifecycle } from '@platform/state/accountLifecycle';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { commands } from 'vitest/browser';

import { useGalleryImageRecovery } from './useGalleryImageRecovery';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const imageRouteCommands = commands as typeof commands & {
  endThumbnailRouteFailures: () => Promise<void>;
  startThumbnailRoute: (pathnames: string[], options?: { failures?: number; status?: number }) => Promise<void>;
  stopThumbnailRoute: () => Promise<void>;
};

/** An index-keyed slot: the element stays, the image it shows changes. */
const Slot = ({ url }: { url: string }) => {
  const { onError, src } = useGalleryImageRecovery(url);

  return <img alt="slot" src={src} onError={onError} />;
};

const FIRST = '/api/v1/images/i/first.png/thumbnail';
const SECOND = '/api/v1/images/i/second.png/thumbnail';
const LOADED = `data:image/svg+xml,${encodeURIComponent(
  '<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8"><rect width="8" height="8" fill="#547c83"/></svg>'
)}`;

let host: HTMLDivElement;
let root: Root;
const slot = () => host.querySelector<HTMLImageElement>('img[alt="slot"]')!;

beforeEach(() => {
  accountLifecycle.activate('image-recovery-slot');
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  vi.useFakeTimers({ shouldAdvanceTime: true, toFake: ['setTimeout', 'clearTimeout'] });
});

afterEach(async () => {
  vi.useRealTimers();
  await act(() => root.unmount());
  host.remove();
  await imageRouteCommands.stopThumbnailRoute();
  accountLifecycle.invalidate();
});

describe('useGalleryImageRecovery', () => {
  it('does not carry a recovered image’s reload into the next image the slot shows', async () => {
    await imageRouteCommands.startThumbnailRoute([FIRST, SECOND], { failures: 1, status: 409 });
    await act(() => root.render(<Slot url={FIRST} />));
    await expect.poll(() => slot().complete && slot().naturalWidth === 0).toBe(true);

    await act(() => vi.advanceTimersByTimeAsync(30_000));
    await expect.poll(() => slot().naturalWidth).toBe(64);
    expect(slot().getAttribute('src')).toContain('gallery_image_recovery=');

    // The slot now shows another image: it asks for that image's own URL, which the HTTP cache can answer.
    await act(() => root.render(<Slot url={SECOND} />));
    expect(slot().getAttribute('src')).toBe(SECOND);
  });

  it('leaves an image that loaded alone when a recovery reloads the failed ones', async () => {
    await imageRouteCommands.startThumbnailRoute([FIRST], { failures: 1, status: 409 });
    await act(() =>
      root.render(
        <>
          <Slot url={FIRST} />
          <Slot url={LOADED} />
        </>
      )
    );
    const [failing] = host.querySelectorAll('img');
    await expect.poll(() => failing!.complete && failing!.naturalWidth === 0).toBe(true);

    await act(() => vi.advanceTimersByTimeAsync(30_000));
    await expect.poll(() => failing!.naturalWidth).toBe(64);
    // The image that never failed keeps its plain URL through the recovery.
    expect(host.querySelectorAll('img')[1]!.getAttribute('src')).toBe(LOADED);
  });
});
