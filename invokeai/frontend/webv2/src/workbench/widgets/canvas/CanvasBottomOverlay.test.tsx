import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { CanvasBottomOverlay } from './CanvasBottomOverlay';

describe('CanvasBottomOverlay', () => {
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
