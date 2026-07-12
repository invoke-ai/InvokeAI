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
import {
  CANVAS_OPERATION_BODY_LAYOUT,
  CANVAS_OPERATION_FIXED_SECTION_LAYOUT,
  CANVAS_OPERATION_PANEL_LAYOUT,
  CanvasOperationPanel,
} from './tool-options/CanvasOperationPanel';

describe('CanvasBottomOverlay', () => {
  it('allocates only the canvas widget inset and allows the stack to shrink', () => {
    expect(BOTTOM_OVERLAY_LAYOUT).toMatchObject({ bottom: '2', minH: '0', overflow: 'hidden', top: '2' });
    expect(BOTTOM_OVERLAY_STACK_LAYOUT).toMatchObject({ h: 'full', minH: '0', overflow: 'hidden' });
    expect(BOTTOM_CONTROLS_SLOT_LAYOUT).toMatchObject({ flex: '1', minH: '0', overflow: 'hidden' });
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

  it('server-renders the complete staging and operation flex chain with body-only scrolling', () => {
    const markup = renderToStaticMarkup(
      <ChakraProvider value={system}>
        <CanvasBottomOverlay.Root data-layout="overlay">
          <CanvasBottomOverlay.Staging data-layout="staging">Staging</CanvasBottomOverlay.Staging>
          <CanvasBottomOverlay.Controls data-layout="controls">
            <CanvasOperationPanel.Root aria-label="Filter" operation="filter">
              <CanvasOperationPanel.Header>Header</CanvasOperationPanel.Header>
              <CanvasOperationPanel.Body>{'Large body '.repeat(500)}</CanvasOperationPanel.Body>
              <CanvasOperationPanel.Feedback>Ready</CanvasOperationPanel.Feedback>
              <CanvasOperationPanel.Footer>Footer</CanvasOperationPanel.Footer>
            </CanvasOperationPanel.Root>
          </CanvasBottomOverlay.Controls>
        </CanvasBottomOverlay.Root>
      </ChakraProvider>
    );

    expect(markup.indexOf('data-layout="overlay"')).toBeLessThan(markup.indexOf('data-layout="staging"'));
    expect(markup.indexOf('data-layout="staging"')).toBeLessThan(markup.indexOf('data-layout="controls"'));
    expect(markup.indexOf('data-slot="header"')).toBeLessThan(markup.indexOf('data-slot="body"'));
    expect(markup.indexOf('data-slot="body"')).toBeLessThan(markup.indexOf('data-slot="feedback"'));
    expect(markup.indexOf('data-slot="feedback"')).toBeLessThan(markup.indexOf('data-slot="footer"'));
    expect(BOTTOM_OVERLAY_LAYOUT).toMatchObject({ bottom: '2', overflow: 'hidden', top: '2' });
    expect(BOTTOM_CONTROLS_SLOT_LAYOUT).toMatchObject({ flex: '1', minH: '0', overflow: 'hidden' });
    expect(CANVAS_OPERATION_PANEL_LAYOUT).toMatchObject({ maxH: 'full', minH: '0', overflow: 'hidden' });
    expect(CANVAS_OPERATION_BODY_LAYOUT).toMatchObject({ minH: '0', overflowY: 'auto' });
    expect(CANVAS_OPERATION_FIXED_SECTION_LAYOUT).toEqual({ flexShrink: '0' });
  });
});
