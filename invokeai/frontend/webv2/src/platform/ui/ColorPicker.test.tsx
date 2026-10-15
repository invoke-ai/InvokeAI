import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';

import { ColorPicker } from './ColorPicker';

describe('ColorPicker', () => {
  it('does not mount its overlay until it is opened', () => {
    const markup = renderToStaticMarkup(
      <ChakraProvider value={system}>
        <ColorPicker aria-label="Fill color" value="#ff0000" onValueChange={vi.fn()} />
      </ChakraProvider>
    );

    expect(markup).not.toContain('data-part="content"');
  });
});
