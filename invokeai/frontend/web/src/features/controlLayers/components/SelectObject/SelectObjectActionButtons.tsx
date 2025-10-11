import { Button, ButtonGroup, Spacer, Spinner } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import { selectAutoProcess } from 'features/controlLayers/store/canvasSettingsSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

import { SelectObjectSaveAsMenu } from './SelectObjectSaveAsMenu';

interface SelectObjectActionButtonsProps {
  adapter: CanvasEntityAdapterRasterLayer | CanvasEntityAdapterControlLayer;
}

export const SelectObjectActionButtons = memo(({ adapter }: SelectObjectActionButtonsProps) => {
  const { t } = useTranslation();
  const isProcessing = useStore(adapter.segmentAnything.$isProcessing);
  const hasInput = useStore(adapter.segmentAnything.$hasInputData);
  const hasImageState = useStore(adapter.segmentAnything.$hasImageState);
  const autoProcess = useAppSelector((state) => selectAutoProcess(state));

  return (
    <ButtonGroup isAttached={false} size="sm" w="full">
      <Button
        onClick={adapter.segmentAnything.processImmediate}
        loadingText={t('controlLayers.selectObject.process')}
        variant="ghost"
        isDisabled={isProcessing || !hasInput || (autoProcess && hasImageState)}
      >
        {t('controlLayers.selectObject.process')}
        {isProcessing && <Spinner ms={3} boxSize={5} color="base.600" />}
      </Button>
      <Spacer />
      <Button
        onClick={adapter.segmentAnything.reset}
        isDisabled={isProcessing || !hasInput}
        loadingText={t('controlLayers.selectObject.reset')}
        variant="ghost"
      >
        {t('controlLayers.selectObject.reset')}
      </Button>
      <Button
        onClick={adapter.segmentAnything.apply}
        loadingText={t('controlLayers.selectObject.apply')}
        variant="ghost"
        isDisabled={isProcessing || !hasImageState}
      >
        {t('controlLayers.selectObject.apply')}
      </Button>
      <SelectObjectSaveAsMenu adapter={adapter} />
      <Button
        onClick={adapter.segmentAnything.cancel}
        isDisabled={isProcessing}
        loadingText={t('common.cancel')}
        variant="ghost"
      >
        {t('controlLayers.selectObject.cancel')}
      </Button>
    </ButtonGroup>
  );
});

SelectObjectActionButtons.displayName = 'SelectObjectActionButtons';
