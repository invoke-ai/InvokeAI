import type { CanvasEngine } from '@workbench/canvas-engine/engine';
import type { CanvasRasterLayerContractV2 } from '@workbench/types';

import { Button } from '@workbench/components/ui';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

interface RasterLayerFilterSectionProps {
  engine: CanvasEngine | null;
  layer: CanvasRasterLayerContractV2;
}

export const RasterLayerFilterSection = ({ engine, layer }: RasterLayerFilterSectionProps) => {
  const { t } = useTranslation();
  const start = useCallback(() => engine?.startFilterOperation(layer.id), [engine, layer.id]);
  return (
    <Button disabled={!engine || layer.isLocked} size="xs" variant="outline" onClick={start}>
      {t('widgets.layers.rasterFilter.title')}
    </Button>
  );
};
