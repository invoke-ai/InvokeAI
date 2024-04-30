import { Button, Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { guidanceLayerAdded } from 'app/store/middleware/listenerMiddleware/listeners/controlLayersToControlAdapterBridge';
import { useAppDispatch } from 'app/store/storeHooks';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';

export const AddLayerButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const addRegionalGuidanceLayer = useCallback(() => {
    dispatch(guidanceLayerAdded('regional_guidance_layer'));
  }, [dispatch]);
  const addControlAdapterLayer = useCallback(() => {
    dispatch(guidanceLayerAdded('control_adapter_layer'));
  }, [dispatch]);
  const addIPAdapterLayer = useCallback(() => {
    dispatch(guidanceLayerAdded('ip_adapter_layer'));
  }, [dispatch]);

  return (
    <Menu>
      <MenuButton as={Button} leftIcon={<PiPlusBold />} variant="ghost">
        {t('controlLayers.addLayer')}
      </MenuButton>
      <MenuList>
        <MenuItem icon={<PiPlusBold />} onClick={addRegionalGuidanceLayer}>
          {t('controlLayers.regionalGuidanceLayer')}
        </MenuItem>
        <MenuItem icon={<PiPlusBold />} onClick={addControlAdapterLayer}>
          {t('controlLayers.globalControlAdapterLayer')}
        </MenuItem>
        <MenuItem icon={<PiPlusBold />} onClick={addIPAdapterLayer}>
          {t('controlLayers.globalIPAdapterLayer')}
        </MenuItem>
      </MenuList>
    </Menu>
  );
});

AddLayerButton.displayName = 'AddLayerButton';
