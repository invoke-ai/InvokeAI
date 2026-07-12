import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import {
  CANVAS_OPERATION_BODY_LAYOUT,
  CANVAS_OPERATION_FIXED_SECTION_LAYOUT,
  CANVAS_OPERATION_FOOTER_LAYOUT,
  CANVAS_OPERATION_PANEL_LAYOUT,
  CANVAS_OPERATION_SLOT_LAYOUT,
  CanvasOperationPanel,
} from './CanvasOperationPanel';

describe('CanvasOperationPanel', () => {
  it('renders the operation panel slots as semantic regions in stable order', () => {
    const markup = renderToStaticMarkup(
      <ChakraProvider value={system}>
        <CanvasOperationPanel.Root aria-labelledby="operation-title">
          <CanvasOperationPanel.Header>
            <h2 id="operation-title">Filter</h2>
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
    expect(markup).toContain('data-operation="filter"');
    expect(markup).toContain('data-slot="header"');
    expect(markup).toContain('data-slot="body"');
    expect(markup).toContain('data-slot="feedback"');
    expect(markup).toContain('data-slot="footer"');
    expect(markup.indexOf('data-slot="header"')).toBeLessThan(markup.indexOf('data-slot="body"'));
    expect(markup.indexOf('data-slot="body"')).toBeLessThan(markup.indexOf('data-slot="feedback"'));
    expect(markup.indexOf('data-slot="feedback"')).toBeLessThan(markup.indexOf('data-slot="footer"'));
    expect(markup.indexOf('Process')).toBeLessThan(markup.indexOf('Cancel'));
  });

  it('uses a widget-sized flex policy with a 30rem ideal width', () => {
    expect(CANVAS_OPERATION_PANEL_LAYOUT).toEqual({
      flex: '0 1 30rem',
      maxH: 'full',
      maxW: 'full',
      minH: '0',
      minW: '0',
      overflow: 'hidden',
      w: '30rem',
    });
    expect(CANVAS_OPERATION_FOOTER_LAYOUT).toMatchObject({ flexWrap: 'wrap', minW: '0' });
    expect(CANVAS_OPERATION_SLOT_LAYOUT).toEqual({ px: '4', py: '3' });
  });

  it('keeps vertical overflow on the body while fixed sections retain their order', () => {
    expect(CANVAS_OPERATION_BODY_LAYOUT).toEqual({
      flex: '1',
      minH: '0',
      overflowX: 'hidden',
      overflowY: 'auto',
    });
    expect(CANVAS_OPERATION_FIXED_SECTION_LAYOUT).toEqual({ flexShrink: '0' });
    const markup = renderToStaticMarkup(
      <ChakraProvider value={system}>
        <CanvasOperationPanel.Root aria-label="Filter">
          <CanvasOperationPanel.Header>Filter</CanvasOperationPanel.Header>
          <CanvasOperationPanel.Body>Parameters</CanvasOperationPanel.Body>
          <CanvasOperationPanel.Feedback>Ready</CanvasOperationPanel.Feedback>
          <CanvasOperationPanel.Footer>Process</CanvasOperationPanel.Footer>
        </CanvasOperationPanel.Root>
      </ChakraProvider>
    );

    const body = markup.indexOf('data-slot="body"');
    const feedback = markup.indexOf('data-slot="feedback"');
    expect(body).toBeGreaterThan(-1);
    expect(body).toBeLessThan(feedback);
  });
});
