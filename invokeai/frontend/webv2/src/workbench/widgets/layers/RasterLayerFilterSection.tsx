import type { CanvasEngine } from '@workbench/canvas-engine/engine';
import type { CanvasRasterLayerContractV2 } from '@workbench/types';

import { LayerFilterOperationButton } from './LayerFilterOperationButton';

interface RasterLayerFilterSectionProps {
  engine: CanvasEngine | null;
  layer: CanvasRasterLayerContractV2;
  onOperationStarted(): void;
}

export const RasterLayerFilterSection = ({ engine, layer, onOperationStarted }: RasterLayerFilterSectionProps) => (
  <LayerFilterOperationButton engine={engine} layer={layer} onOperationStarted={onOperationStarted} />
);
