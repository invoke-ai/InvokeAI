import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import {
  BOTTOM_CONTROLS_SLOT_LAYOUT,
  BOTTOM_OVERLAY_LAYOUT,
  BOTTOM_OVERLAY_STACK_LAYOUT,
  CanvasBottomOverlay,
} from './CanvasBottomOverlay';

describe('CanvasBottomOverlay', () => {
  it('allocates only the canvas widget inset and allows the stack to shrink', () => {
    expect(BOTTOM_OVERLAY_LAYOUT).toMatchObject({ bottom: '2', minH: '0', overflow: 'hidden', top: '2' });
    expect(BOTTOM_OVERLAY_STACK_LAYOUT).toMatchObject({ h: 'full', minH: '0', overflow: 'hidden' });
    expect(BOTTOM_CONTROLS_SLOT_LAYOUT).toMatchObject({ flex: '0 1 auto', minH: '0', overflow: 'hidden' });
  });

  it('server-renders staging above controls inside the bounded layout', () => {
    const markup = renderToStaticMarkup(
      <ChakraProvider value={system}>
        <CanvasBottomOverlay.Root>
          <CanvasBottomOverlay.Staging>Staging</CanvasBottomOverlay.Staging>
          <CanvasBottomOverlay.Controls>Operation</CanvasBottomOverlay.Controls>
        </CanvasBottomOverlay.Root>
      </ChakraProvider>
    );

    expect(markup.indexOf('Staging')).toBeLessThan(markup.indexOf('Operation'));
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
