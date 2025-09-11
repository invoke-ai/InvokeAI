import { Button, Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';

interface SelectObjectSaveAsMenuProps {
  adapter: CanvasEntityAdapterRasterLayer | CanvasEntityAdapterControlLayer;
}

export const SelectObjectSaveAsMenu = memo(({ adapter }: SelectObjectSaveAsMenuProps) => {
  const { t } = useTranslation();
  const isProcessing = useStore(adapter.segmentAnything.$isProcessing);
  const hasImageState = useStore(adapter.segmentAnything.$hasImageState);

  const saveAsInpaintMask = useCallback(() => {
    adapter.segmentAnything.saveAs('inpaint_mask');
  }, [adapter.segmentAnything]);

  const saveAsRegionalGuidance = useCallback(() => {
    adapter.segmentAnything.saveAs('regional_guidance');
  }, [adapter.segmentAnything]);

  const saveAsRasterLayer = useCallback(() => {
    adapter.segmentAnything.saveAs('raster_layer');
  }, [adapter.segmentAnything]);

  const saveAsControlLayer = useCallback(() => {
    adapter.segmentAnything.saveAs('control_layer');
  }, [adapter.segmentAnything]);

  return (
    <Menu>
      <MenuButton
        as={Button}
        loadingText={t('controlLayers.selectObject.saveAs')}
        variant="ghost"
        isDisabled={isProcessing || !hasImageState}
        rightIcon={<PiCaretDownBold />}
      >
        {t('controlLayers.selectObject.saveAs')}
      </MenuButton>
      <MenuList>
        <MenuItem isDisabled={isProcessing || !hasImageState} onClick={saveAsInpaintMask}>
          {t('controlLayers.newInpaintMask')}
        </MenuItem>
        <MenuItem isDisabled={isProcessing || !hasImageState} onClick={saveAsRegionalGuidance}>
          {t('controlLayers.newRegionalGuidance')}
        </MenuItem>
        <MenuItem isDisabled={isProcessing || !hasImageState} onClick={saveAsControlLayer}>
          {t('controlLayers.newControlLayer')}
        </MenuItem>
        <MenuItem isDisabled={isProcessing || !hasImageState} onClick={saveAsRasterLayer}>
          {t('controlLayers.newRasterLayer')}
        </MenuItem>
      </MenuList>
    </Menu>
  );
});

SelectObjectSaveAsMenu.displayName = 'SelectObjectSaveAsMenu';
