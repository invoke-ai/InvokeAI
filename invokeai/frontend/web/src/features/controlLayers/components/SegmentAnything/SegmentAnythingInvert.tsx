import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const SegmentAnythingInvert = memo(
  ({ adapter }: { adapter: CanvasEntityAdapterRasterLayer | CanvasEntityAdapterControlLayer }) => {
    const { t } = useTranslation();
    const invert = useStore(adapter.segmentAnything.$invert);

    const onChange = useCallback(() => {
      adapter.segmentAnything.$invert.set(!adapter.segmentAnything.$invert.get());
    }, [adapter.segmentAnything.$invert]);

    return (
      <FormControl w="min-content">
        <FormLabel m={0}>{t('controlLayers.segment.invertSelection')}</FormLabel>
        <Switch size="sm" isChecked={invert} onChange={onChange} />
      </FormControl>
    );
  }
);

SegmentAnythingInvert.displayName = 'SegmentAnythingInvert';
