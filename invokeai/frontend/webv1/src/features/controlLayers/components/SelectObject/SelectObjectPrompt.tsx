import { FormControl, FormLabel, Input } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const SelectObjectPrompt = memo(
  ({ adapter }: { adapter: CanvasEntityAdapterRasterLayer | CanvasEntityAdapterControlLayer }) => {
    const { t } = useTranslation();
    const inputData = useStore(adapter.segmentAnything.$inputData);

    const onChange = useCallback(
      (e: ChangeEvent<HTMLInputElement>) => {
        adapter.segmentAnything.$inputData.set({ type: 'prompt', prompt: e.target.value });
      },
      [adapter.segmentAnything.$inputData]
    );

    if (inputData.type !== 'prompt') {
      return null;
    }

    return (
      <FormControl w="full">
        <FormLabel m={0}>{t('controlLayers.selectObject.prompt')}</FormLabel>
        <Input value={inputData.prompt} onChange={onChange} />
      </FormControl>
    );
  }
);

SelectObjectPrompt.displayName = 'SelectObjectPrompt';
