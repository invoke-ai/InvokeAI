import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act, StrictMode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { AddNodeDialog } from './AddNodeDialog';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe('AddNodeDialog lifecycle', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  it('does not mount the virtualized result content while closed', async () => {
    const errorSpy = vi.spyOn(console, 'error');

    try {
      await act(() => {
        root.render(
          <StrictMode>
            <ChakraProvider value={system}>
              <AddNodeDialog
                connectionFilter={null}
                isOpen={false}
                onAddConnector={vi.fn()}
                onAddCurrentImage={vi.fn()}
                onAddNode={vi.fn()}
                onAddNote={vi.fn()}
                onOpenChange={vi.fn()}
              />
            </ChakraProvider>
          </StrictMode>
        );
      });

      expect(host.querySelector('[aria-label="Node search results"]')).toBeNull();
      expect(errorSpy.mock.calls.flat().join(' ')).not.toContain('getSnapshot');
    } finally {
      errorSpy.mockRestore();
    }
  });
});
