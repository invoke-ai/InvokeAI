import { Button, Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useAddCALayer, useAddIPALayer } from 'features/controlLayers/hooks/addLayerHooks';
import { rgLayerAdded } from 'features/controlLayers/store/controlLayersSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';

export const AddLayerButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const [addCALayer, isAddCALayerDisabled] = useAddCALayer();
  const [addIPALayer, isAddIPALayerDisabled] = useAddIPALayer();
  const addRGLayer = useCallback(() => {
    dispatch(rgLayerAdded());
  }, [dispatch]);

  return (
    <Menu>
      <MenuButton as={Button} leftIcon={<PiPlusBold />} variant="ghost">
        {t('controlLayers.addLayer')}
      </MenuButton>
      <MenuList>
        <MenuItem icon={<PiPlusBold />} onClick={addRGLayer}>
          {t('controlLayers.regionalGuidanceLayer')}
        </MenuItem>
        <MenuItem icon={<PiPlusBold />} onClick={addCALayer} isDisabled={isAddCALayerDisabled}>
          {t('controlLayers.globalControlAdapterLayer')}
        </MenuItem>
        <MenuItem icon={<PiPlusBold />} onClick={addIPALayer} isDisabled={isAddIPALayerDisabled}>
          {t('controlLayers.globalIPAdapterLayer')}
        </MenuItem>
      </MenuList>
    </Menu>
  );
});

AddLayerButton.displayName = 'AddLayerButton';
