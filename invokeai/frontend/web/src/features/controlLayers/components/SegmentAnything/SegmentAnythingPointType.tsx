import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import { zSAMPointLabel } from 'features/controlLayers/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const SegmentAnythingPointType = memo(
  ({ adapter }: { adapter: CanvasEntityAdapterRasterLayer | CanvasEntityAdapterControlLayer }) => {
    const { t } = useTranslation();
    const pointType = useStore(adapter.segmentAnything.$pointType);

    const options = useMemo(() => {
      return [
        { value: 'foreground', label: t('controlLayers.segment.foreground') },
        { value: 'background', label: t('controlLayers.segment.background') },
        { value: 'neutral', label: t('controlLayers.segment.neutral') },
      ];
    }, [t]);

    const value = useMemo(() => options.find((o) => o.value === pointType) ?? null, [options, pointType]);

    const onChange = useCallback<ComboboxOnChange>(
      (v) => {
        if (!v) {
          return;
        }

        adapter.segmentAnything.$pointType.set(zSAMPointLabel.parse(v.value));
      },
      [adapter.segmentAnything.$pointType]
    );

    return (
      <Flex gap={4} w="full">
        <FormControl maxW={64}>
          <FormLabel m={0}>{t('controlLayers.segment.pointType')}</FormLabel>
          <Combobox options={options} value={value} onChange={onChange} isSearchable={false} isClearable={false} />
        </FormControl>
      </Flex>
    );
  }
);

SegmentAnythingPointType.displayName = 'SegmentAnythingPointType';
