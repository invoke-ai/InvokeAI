import { Button, Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import type { FilterableModelType } from 'features/modelManagerV2/store/modelManagerV2Slice';
import { setFilteredModelType } from 'features/modelManagerV2/store/modelManagerV2Slice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFunnelBold } from 'react-icons/pi';
import { objectKeys } from 'tsafe';

const MODEL_TYPE_LABELS: Record<FilterableModelType, string> = {
  main: 'Main',
  lora: 'LoRA',
  embedding: 'Textual Inversion',
  controlnet: 'ControlNet',
  vae: 'VAE',
  t2i_adapter: 'T2I Adapter',
  ip_adapter: 'IP Adapter',
};

export const ModelTypeFilter = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const filteredModelType = useAppSelector((s) => s.modelmanagerV2.filteredModelType);

  const selectModelType = useCallback(
    (option: FilterableModelType) => {
      dispatch(setFilteredModelType(option));
    },
    [dispatch]
  );

  const clearModelType = useCallback(() => {
    dispatch(setFilteredModelType(null));
  }, [dispatch]);

  return (
    <Menu>
      <MenuButton as={Button} size="sm" leftIcon={<PiFunnelBold />}>
        {filteredModelType ? MODEL_TYPE_LABELS[filteredModelType] : t('modelManager.allModels')}
      </MenuButton>
      <MenuList>
        <MenuItem onClick={clearModelType}>{t('modelManager.allModels')}</MenuItem>
        {objectKeys(MODEL_TYPE_LABELS).map((option) => (
          <MenuItem
            key={option}
            bg={filteredModelType === option ? 'base.700' : 'transparent'}
            onClick={selectModelType.bind(null, option)}
          >
            {MODEL_TYPE_LABELS[option]}
          </MenuItem>
        ))}
      </MenuList>
    </Menu>
  );
};
