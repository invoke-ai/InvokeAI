import { Button, Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import type { FilterableModelType } from 'features/modelManagerV2/store/modelManagerV2Slice';
import { selectFilteredModelType, setFilteredModelType } from 'features/modelManagerV2/store/modelManagerV2Slice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFunnelBold } from 'react-icons/pi';
import { objectKeys } from 'tsafe';

export const ModelTypeFilter = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const MODEL_TYPE_LABELS: Record<FilterableModelType, string> = useMemo(
    () => ({
      main: t('modelManager.main'),
      refiner: t('sdxl.refiner'),
      lora: 'LoRA',
      embedding: t('modelManager.textualInversions'),
      controlnet: 'ControlNet',
      vae: 'VAE',
      t2i_adapter: t('common.t2iAdapter'),
      t5_encoder: t('modelManager.t5Encoder'),
      clip_embed: t('modelManager.clipEmbed'),
      ip_adapter: t('common.ipAdapter'),
      clip_vision: 'CLIP Vision',
      spandrel_image_to_image: t('modelManager.spandrelImageToImage'),
      control_lora: t('modelManager.controlLora'),
      siglip: t('modelManager.siglip'),
      flux_redux: t('modelManager.fluxRedux'),
    }),
    [t]
  );
  const filteredModelType = useAppSelector(selectFilteredModelType);

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
});

ModelTypeFilter.displayName = 'ModelTypeFilter';
