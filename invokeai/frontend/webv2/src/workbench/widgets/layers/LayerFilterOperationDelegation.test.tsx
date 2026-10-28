import type { CanvasEngine, CanvasRasterLayerContractV2 } from '@workbench/canvas-engine/api';

import { createEmptyPaintLayer } from '@workbench/widgets/layers/layerOps';
import { renderToStaticMarkup } from 'react-dom/server';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { RasterLayerFilterSection } from './RasterLayerFilterSection';

interface SharedLaunchProps {
  engine: CanvasEngine | null;
  layer: CanvasRasterLayerContractV2;
  onOperationStarted(): void;
  operations: unknown;
}

const { operations, renderSharedLaunch } = vi.hoisted(() => ({
  operations: { startFilterOperation: vi.fn() },
  renderSharedLaunch: vi.fn(),
}));
const ENGINE = {
  document: { getDocument: vi.fn(() => ({ layers: [] })), model: () => null },
  exports: { hasExportableLayerContent: vi.fn(() => false) },
  interaction: { get: () => false, subscribe: () => () => undefined },
} as unknown as CanvasEngine;

vi.mock('./LayerFilterOperationButton', () => ({
  LayerFilterOperationButton: (props: SharedLaunchProps) => {
    renderSharedLaunch(props);
    return <span>shared filter launch</span>;
  },
}));
vi.mock('@workbench/canvas-operations/api', () => ({
  getCanvasOperations: () => operations,
}));

beforeEach(() => {
  renderSharedLaunch.mockClear();
});

describe('layer filter launch delegation', () => {
  it('delegates raster launch rendering to the shared control with the original props', () => {
    const layer = createEmptyPaintLayer('Raster', 'raster');
    const onOperationStarted = vi.fn();

    renderToStaticMarkup(
      <RasterLayerFilterSection engine={ENGINE} layer={layer} onOperationStarted={onOperationStarted} />
    );

    expect(renderSharedLaunch).toHaveBeenCalledOnce();
    expect(renderSharedLaunch).toHaveBeenCalledWith(
      expect.objectContaining({ engine: ENGINE, layer, onOperationStarted, operations: expect.any(Object) })
    );
  });
});
