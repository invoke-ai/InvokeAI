import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

import { ProjectCover } from './ProjectCover';

/**
 * The three states a cover has: an image, no image, and an image the browser
 * could not load. The last is the one worth pinning — a cover names a server
 * image that may since have been deleted, and a broken `<img>` reads worse than
 * the glyph. All three reserve the same box, so a grid does not reflow as
 * covers resolve.
 */

// A 1x1 transparent GIF, so the success path needs no network.
const PIXEL = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
const BROKEN = 'data:image/gif;base64,not-an-image';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const render = async (coverUrl?: string) => {
  host = document.createElement('div');
  host.style.width = '320px';
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <ProjectCover coverUrl={coverUrl} />
      </ChakraProvider>
    );
  });

  return host;
};

const waitFor = async (predicate: () => boolean) => {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (predicate()) {
      return;
    }

    await act(async () => {
      await new Promise((resolve) => {
        setTimeout(resolve, 20);
      });
    });
  }

  throw new Error('condition never became true');
};

describe('ProjectCover', () => {
  it('shows the folder glyph and no image when there is no cover', async () => {
    const container = await render();

    expect(container.querySelector('img')).toBeNull();
    expect(container.querySelector('svg')).not.toBeNull();
  });

  it('renders the cover as a decorative image when one is supplied', async () => {
    const container = await render(PIXEL);
    const image = container.querySelector('img');

    expect(image).not.toBeNull();
    // The name is always rendered beside the cover, so announcing it twice
    // would only add noise.
    expect(image?.getAttribute('alt')).toBe('');
    expect(image?.getAttribute('src')).toBe(PIXEL);
  });

  it('falls back to the glyph when the cover image fails to load', async () => {
    const container = await render(BROKEN);

    await waitFor(() => container.querySelector('img') === null);
    expect(container.querySelector('svg')).not.toBeNull();
  });

  it('reserves the same box whether or not a cover is present', async () => {
    const withoutCover = (await render()).firstElementChild?.getBoundingClientRect();

    await act(() => root?.unmount());
    host?.remove();

    const withCover = (await render(PIXEL)).firstElementChild?.getBoundingClientRect();

    expect(withoutCover?.height).toBeGreaterThan(0);
    expect(withCover?.height).toBe(withoutCover?.height);
  });
});
