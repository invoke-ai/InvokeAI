import { Button, Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import type { ModelCategoryData } from 'features/modelManagerV2/models';
import { MODEL_CATEGORIES, MODEL_CATEGORIES_AS_LIST } from 'features/modelManagerV2/models';
import { selectFilteredModelType, setFilteredModelType } from 'features/modelManagerV2/store/modelManagerV2Slice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFunnelBold } from 'react-icons/pi';

export const ModelTypeFilter = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const filteredModelType = useAppSelector(selectFilteredModelType);

  const clearModelType = useCallback(() => {
    dispatch(setFilteredModelType(null));
  }, [dispatch]);

  return (
    <Menu placement="bottom-end">
      <MenuButton as={Button} size="sm" rightIcon={<PiFunnelBold />}>
        {filteredModelType ? t(MODEL_CATEGORIES[filteredModelType].i18nKey) : t('modelManager.allModels')}
      </MenuButton>
      <MenuList>
        <MenuItem onClick={clearModelType}>{t('modelManager.allModels')}</MenuItem>
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
