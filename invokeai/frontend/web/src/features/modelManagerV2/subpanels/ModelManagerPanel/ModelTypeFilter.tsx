import { Button, Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setFilteredModelType } from 'features/modelManagerV2/store/modelManagerV2Slice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { IoFilter } from 'react-icons/io5';

const MODEL_TYPE_LABELS: { [key: string]: string } = {
  main: 'Main',
  lora: 'LoRA',
  embedding: 'Textual Inversion',
  controlnet: 'ControlNet',
  vae: 'VAE',
  t2i_adapter: 'T2I Adapter',
  ip_adapter: 'IP Adapter',
  clip_vision: 'Clip Vision',
};

export const ModelTypeFilter = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const filteredModelType = useAppSelector((s) => s.modelmanagerV2.filteredModelType);

  const selectModelType = useCallback(
    (option: string) => {
      dispatch(setFilteredModelType(option));
    },
    [dispatch]
  );

  const clearModelType = useCallback(() => {
    dispatch(setFilteredModelType(null));
  }, [dispatch]);

  return (
    <Menu>
      <MenuButton as={Button} leftIcon={<IoFilter />}>
        {filteredModelType ? MODEL_TYPE_LABELS[filteredModelType] : t('modelManager.allModels')}
      </MenuButton>
      <MenuList>
        <MenuItem onClick={clearModelType}>{t('modelManager.allModels')}</MenuItem>
        {Object.keys(MODEL_TYPE_LABELS).map((option) => (
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
