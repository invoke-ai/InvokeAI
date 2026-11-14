import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { DropZone } from './DropZone';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const CALLER_CSS = { outlineStyle: 'solid', outlineWidth: '3px' } as const;

describe('DropZone', () => {
  it('stays in its hover look while the trigger filling it is pointed, keeping a caller css', async () => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() =>
      root?.render(
        <ChakraProvider value={system}>
          <DropZone css={CALLER_CSS} data-testid="zone" h="16" w="16">
            <button style={{ height: '100%', width: '100%' }} type="button">
              Choose
            </button>
          </DropZone>
        </ChakraProvider>
      )
    );
    const zone = host.querySelector<HTMLElement>('[data-testid="zone"]')!;
    const trigger = zone.querySelector('button')!;
    // Read once the zone's border transition has finished.
    const border = async () => {
      await vi.waitFor(() => expect(zone.getAnimations()).toHaveLength(0));
      return getComputedStyle(zone).borderColor;
    };
    const resting = await border();

    await userEvent.hover(zone);
    const hovered = await border();
    expect(hovered).not.toBe(resting);
    await userEvent.unhover(zone);
    expect(await border()).toBe(resting);

    // The gallery picker marks its trigger `data-hover` while it is open.
    trigger.setAttribute('data-hover', '');
    expect(await border()).toBe(hovered);
    trigger.removeAttribute('data-hover');
    expect(await border()).toBe(resting);

    expect(getComputedStyle(zone).outlineWidth).toBe('3px');
  });
});
