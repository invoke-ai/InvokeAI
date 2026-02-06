import { Button, Flex, Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import type { ModelCategoryData } from 'features/modelManagerV2/models';
import { MODEL_CATEGORIES, MODEL_CATEGORIES_AS_LIST } from 'features/modelManagerV2/models';
import type { ModelCategoryType } from 'features/modelManagerV2/store/modelManagerV2Slice';
import { selectFilteredModelType, setFilteredModelType } from 'features/modelManagerV2/store/modelManagerV2Slice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFunnelBold, PiWarningBold } from 'react-icons/pi';

const isModelCategoryType = (type: string): type is ModelCategoryType => {
  return type in MODEL_CATEGORIES;
};

export const ModelTypeFilter = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const filteredModelType = useAppSelector(selectFilteredModelType);

  const clearModelType = useCallback(() => {
    dispatch(setFilteredModelType(null));
  }, [dispatch]);

  const setMissingFilter = useCallback(() => {
    dispatch(setFilteredModelType('missing'));
  }, [dispatch]);

  const getButtonLabel = () => {
    if (filteredModelType === 'missing') {
      return t('modelManager.missingFiles');
    }
    if (filteredModelType && isModelCategoryType(filteredModelType)) {
      return t(MODEL_CATEGORIES[filteredModelType].i18nKey);
    }
    return t('modelManager.allModels');
  };

  return (
    <Menu placement="bottom-end">
      <MenuButton as={Button} size="sm" rightIcon={<PiFunnelBold />}>
        {getButtonLabel()}
      </MenuButton>
      <MenuList>
        <MenuItem onClick={clearModelType}>{t('modelManager.allModels')}</MenuItem>
        <MenuItem
          onClick={setMissingFilter}
          bg={filteredModelType === 'missing' ? 'base.700' : 'transparent'}
          color="warning.300"
        >
          <Flex alignItems="center" gap={2}>
            <PiWarningBold />
            {t('modelManager.missingFiles')}
          </Flex>
        </MenuItem>
        {MODEL_CATEGORIES_AS_LIST.map((data) => (
          <ModelMenuItem key={data.category} data={data} />
        ))}
      </MenuList>
    </Menu>
  );
});

ModelTypeFilter.displayName = 'ModelTypeFilter';

const ModelMenuItem = memo(({ data }: { data: ModelCategoryData }) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const filteredModelType = useAppSelector(selectFilteredModelType);
  const onClick = useCallback(() => {
    dispatch(setFilteredModelType(data.category));
  }, [data.category, dispatch]);
  return (
    <MenuItem bg={filteredModelType === data.category ? 'base.700' : 'transparent'} onClick={onClick}>
      {t(data.i18nKey)}
    </MenuItem>
  );
});
ModelMenuItem.displayName = 'ModelMenuItem';
