import { Button, Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { guidanceLayerAdded } from 'app/store/middleware/listenerMiddleware/listeners/regionalControlToControlAdapterBridge';
import { useAppDispatch } from 'app/store/storeHooks';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';

export const AddLayerButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const addMaskedGuidanceLayer = useCallback(() => {
    dispatch(guidanceLayerAdded('masked_guidance_layer'));
  }, [dispatch]);
  const addControlNetLayer = useCallback(() => {
    dispatch(guidanceLayerAdded('control_adapter_layer'));
  }, [dispatch]);
  const addIPAdapterLayer = useCallback(() => {
    dispatch(guidanceLayerAdded('ip_adapter_layer'));
  }, [dispatch]);

  return (
    <Menu>
      <MenuButton as={Button} leftIcon={<PiPlusBold />} variant="ghost">
        {t('regionalPrompts.addLayer')}
      </MenuButton>
      <MenuList>
        <MenuItem onClick={addMaskedGuidanceLayer}> {t('regionalPrompts.maskedGuidanceLayer')}</MenuItem>
        <MenuItem onClick={addControlNetLayer}> {t('regionalPrompts.controlNetLayer')}</MenuItem>
        <MenuItem onClick={addIPAdapterLayer}> {t('regionalPrompts.ipAdapterLayer')}</MenuItem>
      </MenuList>
    </Menu>
  );
});

AddLayerButton.displayName = 'AddLayerButton';
