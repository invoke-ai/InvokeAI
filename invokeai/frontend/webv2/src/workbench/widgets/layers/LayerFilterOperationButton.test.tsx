import type { CanvasEngine, StartFilterOperationResult } from '@workbench/canvas-engine/engine';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createControlLayer, createEmptyPaintLayer } from '@workbench/widgets/layers/layerOps';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';

import { LayerFilterOperationButton } from './LayerFilterOperationButton';

const { notifyError } = vi.hoisted(() => ({ notifyError: vi.fn() }));

vi.mock('@workbench/useNotify', () => ({
  useNotify: () => ({ error: notifyError, info: vi.fn(), success: vi.fn() }),
}));
vi.mock('@workbench/widgets/canvas/engineStoreHooks', () => ({
  useLayerThumbnailVersion: () => 1,
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

const createEngine = (hasExportableContent: boolean, result: StartFilterOperationResult = 'started') =>
  ({
    hasExportableLayerContent: vi.fn(() => hasExportableContent),
    startFilterOperation: vi.fn(() => result),
  }) as unknown as CanvasEngine;

const renderButton = (
  layer: ReturnType<typeof createControlLayer> | ReturnType<typeof createEmptyPaintLayer>,
  hasExportableContent: boolean
) =>
  renderToStaticMarkup(
    <ChakraProvider value={system}>
      <LayerFilterOperationButton
        engine={createEngine(hasExportableContent)}
        layer={layer}
        onOperationStarted={vi.fn()}
      />
    </ChakraProvider>
  );

describe('LayerFilterOperationButton', () => {
  it.each([
    ['raster', createEmptyPaintLayer('Raster', 'raster')],
    ['control', createControlLayer('Control', 'control')],
  ] as const)('disables empty %s layers', (_type, layer) => {
    const markup = renderButton(layer, false);

    expect(markup).toContain('widgets.layers.control.filter');
    expect(markup).toContain('disabled');
  });

  it.each([
    ['raster', createEmptyPaintLayer('Raster', 'raster')],
    ['control', createControlLayer('Control', 'control')],
  ] as const)('enables filterable %s layers', (_type, layer) => {
    const markup = renderButton(layer, true);

    expect(markup).toContain('widgets.layers.control.filter');
    expect(markup).not.toContain('disabled=""');
  });
});
