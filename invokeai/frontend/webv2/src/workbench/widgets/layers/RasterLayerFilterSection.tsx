import type { CanvasEngine } from '@workbench/canvas-engine/engine';
import type { CanvasRasterLayerContractV2 } from '@workbench/types';

import { Button } from '@workbench/components/ui';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { runLayerFilterOperation } from './layerPropertiesOperation';

interface RasterLayerFilterSectionProps {
  engine: CanvasEngine | null;
  layer: CanvasRasterLayerContractV2;
  onOperationStarted(): void;
}

export const RasterLayerFilterSection = ({ engine, layer, onOperationStarted }: RasterLayerFilterSectionProps) => {
  const { t } = useTranslation();
  const start = useCallback(
    () => runLayerFilterOperation(() => engine?.startFilterOperation(layer.id), onOperationStarted),
    [engine, layer.id, onOperationStarted]
  );
  return (
    <Button disabled={!engine || layer.isLocked} size="xs" variant="outline" onClick={start}>
      {t('widgets.layers.rasterFilter.title')}
    </Button>
  );
};
