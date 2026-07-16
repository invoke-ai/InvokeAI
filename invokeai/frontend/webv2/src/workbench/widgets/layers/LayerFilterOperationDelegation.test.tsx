import type { CanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import type { CanvasControlLayerContract, CanvasRasterLayerContractV2 } from '@workbench/types';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createControlLayer, createEmptyPaintLayer } from '@workbench/widgets/layers/layerOps';
import { renderToStaticMarkup } from 'react-dom/server';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { ControlLayerSettings } from './ControlLayerSettings';
import { RasterLayerFilterSection } from './RasterLayerFilterSection';

interface SharedLaunchProps {
  engine: CanvasEngine | null;
  layer: CanvasControlLayerContract | CanvasRasterLayerContractV2;
  onOperationStarted(): void;
  operations: unknown;
}

const { dispatch, operations, renderSharedLaunch } = vi.hoisted(() => ({
  dispatch: vi.fn(),
  operations: { startFilterOperation: vi.fn() },
  renderSharedLaunch: vi.fn(),
}));
const ENGINE = {
  document: { getDocument: vi.fn(() => ({ layers: [] })) },
  exports: { hasExportableLayerContent: vi.fn(() => false) },
} as unknown as CanvasEngine;

vi.mock('./LayerFilterOperationButton', () => ({
  LayerFilterOperationButton: (props: SharedLaunchProps) => {
    renderSharedLaunch(props);
    return <span>shared filter launch</span>;
  },
}));
vi.mock('@workbench/canvas-operations/createCanvasEngine', () => ({
  getCanvasOperations: () => operations,
}));
vi.mock('@workbench/models/modelsStore', () => ({
  useModelsSelector: (selector: (snapshot: { models: never[] }) => unknown) => selector({ models: [] }),
}));
vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectId: () => 'project',
  useActiveProjectSelector: () => null,
  useWorkbenchDispatch: () => dispatch,
  useWorkbenchStore: () => ({ dispatch, getState: vi.fn() }),
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string) => key }),
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

  it('delegates control-layer launch rendering to the shared control with the original props', () => {
    const layer = createControlLayer('Control', 'control');
    const onOperationStarted = vi.fn();

    renderToStaticMarkup(
      <ChakraProvider value={system}>
        <ControlLayerSettings engine={ENGINE} layer={layer} onOperationStarted={onOperationStarted} />
      </ChakraProvider>
    );

    expect(renderSharedLaunch).toHaveBeenCalledOnce();
    expect(renderSharedLaunch).toHaveBeenCalledWith(
      expect.objectContaining({ engine: ENGINE, layer, onOperationStarted, operations: expect.any(Object) })
    );
  });
});
