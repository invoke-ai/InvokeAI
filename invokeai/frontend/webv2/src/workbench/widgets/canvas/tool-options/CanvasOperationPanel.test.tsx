import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { CanvasOperationPanel } from './CanvasOperationPanel';

describe('CanvasOperationPanel', () => {
  it('renders the operation panel slots as semantic regions in stable order', () => {
    const markup = renderToStaticMarkup(
      <ChakraProvider value={system}>
        <CanvasOperationPanel.Root aria-labelledby="operation-title" operation="select-object">
          <CanvasOperationPanel.Header>
            <h2 id="operation-title">Select Object</h2>
          </CanvasOperationPanel.Header>
          <CanvasOperationPanel.Body>Inputs</CanvasOperationPanel.Body>
          <CanvasOperationPanel.Feedback>Status</CanvasOperationPanel.Feedback>
          <CanvasOperationPanel.Footer>
            <button>Process</button>
            <button>Cancel</button>
          </CanvasOperationPanel.Footer>
        </CanvasOperationPanel.Root>
      </ChakraProvider>
    );

    expect(markup).toContain('role="region"');
    expect(markup).toContain('aria-labelledby="operation-title"');
    expect(markup).toContain('data-operation="select-object"');
    expect(markup).toContain('data-slot="header"');
    expect(markup).toContain('data-slot="body"');
    expect(markup).toContain('data-slot="feedback"');
    expect(markup).toContain('data-slot="footer"');
    expect(markup.indexOf('data-slot="header"')).toBeLessThan(markup.indexOf('data-slot="body"'));
    expect(markup.indexOf('data-slot="body"')).toBeLessThan(markup.indexOf('data-slot="feedback"'));
    expect(markup.indexOf('data-slot="feedback"')).toBeLessThan(markup.indexOf('data-slot="footer"'));
    expect(markup.indexOf('Process')).toBeLessThan(markup.indexOf('Cancel'));
  });

  it('keeps responsive width on the canvas container and scroll policy on the body only', () => {
    const markup = renderToStaticMarkup(
      <ChakraProvider value={system}>
        <CanvasOperationPanel.Root aria-label="Filter" operation="filter">
          <CanvasOperationPanel.Body>Parameters</CanvasOperationPanel.Body>
        </CanvasOperationPanel.Root>
      </ChakraProvider>
    );

    expect(markup).toContain('data-operation-panel-width="responsive"');
    expect(markup).toContain('data-operation-panel-max-width="container"');
    expect(markup).toContain('data-scroll-container="body"');
  });
});
