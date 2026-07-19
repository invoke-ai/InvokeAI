import type { CanvasRasterLayerContractV2 } from '@workbench/canvas-engine/api';
import type { LayerFilterOperationEngine } from '@workbench/widgets/layers/LayerFilterOperationButton';

import { getCanvasOperations } from '@workbench/canvas-operations/api';

import { LayerFilterOperationButton } from './LayerFilterOperationButton';

interface RasterLayerFilterSectionProps {
  engine: LayerFilterOperationEngine | null;
  layer: CanvasRasterLayerContractV2;
  onOperationStarted(): void;
}

export const RasterLayerFilterSection = ({ engine, layer, onOperationStarted }: RasterLayerFilterSectionProps) => (
  <LayerFilterOperationButton
    engine={engine}
    layer={layer}
    onOperationStarted={onOperationStarted}
    operations={engine ? getCanvasOperations(engine) : null}
  />
);
