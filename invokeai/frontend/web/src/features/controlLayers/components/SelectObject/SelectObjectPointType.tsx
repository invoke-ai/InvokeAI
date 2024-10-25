import { Flex, FormControl, FormLabel, Radio, RadioGroup, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import { SAM_POINT_LABEL_STRING_TO_NUMBER, zSAMPointLabelString } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const SelectObjectPointType = memo(
  ({ adapter }: { adapter: CanvasEntityAdapterRasterLayer | CanvasEntityAdapterControlLayer }) => {
    const { t } = useTranslation();
    const pointType = useStore(adapter.segmentAnything.$pointTypeString);

    const onChange = useCallback(
      (v: string) => {
        const labelAsString = zSAMPointLabelString.parse(v);
        const labelAsNumber = SAM_POINT_LABEL_STRING_TO_NUMBER[labelAsString];
        adapter.segmentAnything.$pointType.set(labelAsNumber);
      },
      [adapter.segmentAnything.$pointType]
    );

    return (
      <FormControl w="min-content">
        <FormLabel m={0}>{t('controlLayers.selectObject.pointType')}</FormLabel>
        <RadioGroup value={pointType} onChange={onChange} w="full" size="md">
          <Flex alignItems="center" w="full" gap={4} fontWeight="semibold" color="base.300">
            <Radio value="foreground">
              <Text>{t('controlLayers.selectObject.include')}</Text>
            </Radio>
            <Radio value="background">
              <Text>{t('controlLayers.selectObject.exclude')}</Text>
            </Radio>
          </Flex>
        </RadioGroup>
      </FormControl>
    );
  }
);

SelectObjectPointType.displayName = 'SelectObject';
