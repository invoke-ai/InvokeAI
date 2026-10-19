import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({ getImageLabels: vi.fn() }));

vi.mock('@workbench/image-map/imageLabelCache', () => ({ getImageLabels: mocks.getImageLabels }));

import { MapHoverCard } from './MapHoverCard';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;

const HOVER_CLUSTER = { cluster: 0, clusterSize: 4 } as const;

const render = async (preview: Parameters<typeof MapHoverCard>[0]['preview']) => {
  await act(async () => {
    root?.render(
      <ChakraProvider value={system}>
        <MapHoverCard clusterLabel={null} hoverCluster={HOVER_CLUSTER} preview={preview} />
      </ChakraProvider>
    );
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });
  });

  return host?.textContent ?? '';
};

beforeEach(() => {
  mocks.getImageLabels.mockReset();
  mocks.getImageLabels.mockResolvedValue(null);
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
  root = null;
  host = null;
});

describe('MapHoverCard', () => {
  it('marks a video with the gallery’s play badge and its duration', async () => {
    // Identify videos because their still thumbnails otherwise look like images.
    const text = await render({
      clientX: 10,
      clientY: 10,
      key: 'video:clip.mp4',
      thumbnail: { durationSeconds: 75, url: '/thumb.webp' },
    });

    expect(text).toContain('1:15');
    expect(text).toContain('clip.mp4');
    expect(host?.querySelector('img')?.getAttribute('alt')).toBe('Video: clip.mp4');
  });

  it('leaves an image unmarked and names it plainly', async () => {
    const text = await render({
      clientX: 10,
      clientY: 10,
      key: 'image:a.png',
      thumbnail: { durationSeconds: null, url: '/thumb.webp' },
    });

    expect(text).toContain('a.png');
    expect(text).not.toContain(':');
    expect(host?.querySelector('img')?.getAttribute('alt')).toBe('a.png');
  });

  it('asks for the hovered item’s tags in its own namespace', async () => {
    mocks.getImageLabels.mockResolvedValue({ alternates: ['surf'], label: 'clip' });

    const text = await render({
      clientX: 10,
      clientY: 10,
      key: 'video:clip.mp4',
      thumbnail: { durationSeconds: 2, url: '/thumb.webp' },
    });

    expect(mocks.getImageLabels).toHaveBeenCalledWith({ kind: 'video', name: 'clip.mp4' });
    // Named apart from the cluster's tags.
    expect(text).toContain('Image tags: clip, surf');
  });
});
