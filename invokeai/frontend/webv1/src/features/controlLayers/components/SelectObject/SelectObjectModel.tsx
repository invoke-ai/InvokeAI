import { Flex, FormControl, FormLabel, Radio, RadioGroup, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import { zSAMModel } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const SelectObjectModel = memo(
  ({ adapter }: { adapter: CanvasEntityAdapterRasterLayer | CanvasEntityAdapterControlLayer }) => {
    const { t } = useTranslation();
    const model = useStore(adapter.segmentAnything.$model);

    const onChange = useCallback(
      (v: string) => {
        const model = zSAMModel.parse(v);
        adapter.segmentAnything.$model.set(model);
      },
      [adapter.segmentAnything.$model]
    );

    return (
      <FormControl w="full">
        <FormLabel m={0}>{t('controlLayers.selectObject.model')}</FormLabel>
        <RadioGroup value={model} onChange={onChange} w="full" size="md">
          <Flex alignItems="center" w="full" gap={4} color="base.300">
            <Radio value="SAM1">
              <Text>{t('controlLayers.selectObject.segmentAnything1')}</Text>
            </Radio>
            <Radio value="SAM2">
              <Text>{t('controlLayers.selectObject.segmentAnything2')}</Text>
            </Radio>
          </Flex>
        </RadioGroup>
      </FormControl>
    );
  }
);

SelectObjectModel.displayName = 'SelectObjectModel';
