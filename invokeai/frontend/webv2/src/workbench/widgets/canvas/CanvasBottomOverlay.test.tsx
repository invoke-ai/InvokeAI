import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import {
  BOTTOM_OVERLAY_LAYOUT,
  BOTTOM_OVERLAY_STACK_LAYOUT,
  BOTTOM_STAGING_SLOT_LAYOUT,
  CanvasBottomOverlay,
} from './CanvasBottomOverlay';

describe('CanvasBottomOverlay', () => {
  it('allocates only the canvas widget inset and allows the stack to shrink', () => {
    expect(BOTTOM_OVERLAY_LAYOUT).toMatchObject({ bottom: '2', minH: '0', overflow: 'hidden', top: '2' });
    expect(BOTTOM_OVERLAY_STACK_LAYOUT).toMatchObject({ h: 'full', minH: '0', overflow: 'hidden' });
  });

  it('spans the staging slot across the canvas so its thumbnail strip can scroll', () => {
    expect(BOTTOM_STAGING_SLOT_LAYOUT).toMatchObject({ flexShrink: '0', minW: '0', w: 'full' });
  });

  it('forwards root props and ref-compatible attributes', () => {
    const markup = renderToStaticMarkup(
      <ChakraProvider value={system}>
        <CanvasBottomOverlay.Root aria-label="Bottom chrome" data-forwarded="yes" id="bottom-overlay" />
      </ChakraProvider>
    );

    expect(markup).toContain('aria-label="Bottom chrome"');
    expect(markup).toContain('data-forwarded="yes"');
    expect(markup).toContain('id="bottom-overlay"');
  });
});
