import { Button, Flex, Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { SubMenuButtonContent, useSubMenu } from 'common/hooks/useSubMenu';
import type { ModelCategoryData } from 'features/modelManagerV2/models';
import { MODEL_CATEGORIES_AS_LIST } from 'features/modelManagerV2/models';
import {
  selectFilteredModelType,
  selectOrderBy,
  selectSortDirection,
  setFilteredModelType,
  setOrderBy,
  setSortDirection,
} from 'features/modelManagerV2/store/modelManagerV2Slice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiCheckBold,
  PiFunnelBold,
  PiListBold,
  PiSortAscendingBold,
  PiSortDescendingBold,
  PiWarningBold,
} from 'react-icons/pi';

type OrderBy = 'default' | 'name' | 'type' | 'base' | 'size' | 'created_at' | 'updated_at' | 'path' | 'format';

const ORDER_BY_OPTIONS: OrderBy[] = [
  'default',
  'name',
  'base',
  'size',
  'created_at',
  'updated_at',
  'path',
  'type',
  'format',
];

const SortByMenuItem = memo(({ option, label }: { option: OrderBy; label: string }) => {
  const dispatch = useAppDispatch();
  const orderBy = useAppSelector(selectOrderBy);
  const onClick = useCallback(() => {
    dispatch(setOrderBy(option));
  }, [dispatch, option]);

  return (
    <MenuItem
      onClick={onClick}
      bg={orderBy === option ? 'base.700' : 'transparent'}
      icon={orderBy === option ? <PiCheckBold /> : <PiCheckBold visibility="hidden" />}
    >
      {label}
    </MenuItem>
  );
});
SortByMenuItem.displayName = 'SortByMenuItem';

const SortBySubMenu = memo(() => {
  const { t } = useTranslation();
  const subMenu = useSubMenu();
  const orderBy = useAppSelector(selectOrderBy);

  const ORDER_BY_LABELS = useMemo(
    () => ({
      default: t('modelManager.sortDefault'),
      name: t('modelManager.sortByName'),
      base: t('modelManager.sortByBase'),
      size: t('modelManager.sortBySize'),
      created_at: t('modelManager.sortByDateAdded'),
      updated_at: t('modelManager.sortByDateModified'),
      path: t('modelManager.sortByPath'),
      type: t('modelManager.sortByType'),
      format: t('modelManager.sortByFormat'),
    }),
    [t]
  );

  return (
    <MenuItem {...subMenu.parentMenuItemProps} icon={<PiListBold />}>
      <Menu {...subMenu.menuProps}>
        <MenuButton {...subMenu.menuButtonProps}>
          <SubMenuButtonContent label={t('modelManager.sortBy', 'Sort By')} value={ORDER_BY_LABELS[orderBy]} />
        </MenuButton>
        <MenuList {...subMenu.menuListProps}>
          {ORDER_BY_OPTIONS.map((option) => (
            <SortByMenuItem key={option} option={option} label={ORDER_BY_LABELS[option]} />
          ))}
        </MenuList>
      </Menu>
    </MenuItem>
  );
});
SortBySubMenu.displayName = 'SortBySubMenu';

const DirectionSubMenu = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const direction = useAppSelector(selectSortDirection);
  const subMenu = useSubMenu();

  const setDirectionAsc = useCallback(() => {
    dispatch(setSortDirection('asc'));
  }, [dispatch]);

  const setDirectionDesc = useCallback(() => {
    dispatch(setSortDirection('desc'));
  }, [dispatch]);

  const currentValue = direction === 'asc' ? t('common.ascending', 'Ascending') : t('common.descending', 'Descending');

  return (
    <MenuItem
      {...subMenu.parentMenuItemProps}
      icon={direction === 'asc' ? <PiSortAscendingBold /> : <PiSortDescendingBold />}
    >
      <Menu {...subMenu.menuProps}>
        <MenuButton {...subMenu.menuButtonProps}>
          <SubMenuButtonContent label={t('common.direction', 'Direction')} value={currentValue} />
        </MenuButton>
        <MenuList {...subMenu.menuListProps}>
          <MenuItem
            onClick={setDirectionAsc}
            bg={direction === 'asc' ? 'base.700' : 'transparent'}
            icon={direction === 'asc' ? <PiCheckBold /> : <PiCheckBold visibility="hidden" />}
          >
            {t('common.ascending', 'Ascending')}
          </MenuItem>
          <MenuItem
            onClick={setDirectionDesc}
            bg={direction === 'desc' ? 'base.700' : 'transparent'}
            icon={direction === 'desc' ? <PiCheckBold /> : <PiCheckBold visibility="hidden" />}
          >
            {t('common.descending', 'Descending')}
          </MenuItem>
        </MenuList>
      </Menu>
    </MenuItem>
  );
});
DirectionSubMenu.displayName = 'DirectionSubMenu';

const ModelTypeSubMenu = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const filteredModelType = useAppSelector(selectFilteredModelType);
  const subMenu = useSubMenu();

  const clearModelType = useCallback(() => {
    dispatch(setFilteredModelType(null));
  }, [dispatch]);

  const setMissingFilter = useCallback(() => {
    dispatch(setFilteredModelType('missing'));
  }, [dispatch]);

  const currentValue = useMemo(() => {
    if (filteredModelType === null) {
      return t('modelManager.allModels');
    }
    if (filteredModelType === 'missing') {
      return t('modelManager.missingFiles');
    }
    const categoryData = MODEL_CATEGORIES_AS_LIST.find((data) => data.category === filteredModelType);
    return categoryData ? t(categoryData.i18nKey) : '';
  }, [filteredModelType, t]);

  return (
    <MenuItem {...subMenu.parentMenuItemProps} icon={<PiFunnelBold />}>
      <Menu {...subMenu.menuProps}>
        <MenuButton {...subMenu.menuButtonProps}>
          <SubMenuButtonContent label={t('modelManager.modelType', 'Model Type')} value={currentValue} />
        </MenuButton>
        <MenuList {...subMenu.menuListProps}>
          <MenuItem
            onClick={clearModelType}
            bg={filteredModelType === null ? 'base.700' : 'transparent'}
            icon={filteredModelType === null ? <PiCheckBold /> : <PiCheckBold visibility="hidden" />}
          >
            {t('modelManager.allModels')}
          </MenuItem>
          <MenuItem
            onClick={setMissingFilter}
            bg={filteredModelType === 'missing' ? 'base.700' : 'transparent'}
            color="warning.300"
            icon={filteredModelType === 'missing' ? <PiCheckBold /> : <PiCheckBold visibility="hidden" />}
          >
            <Flex alignItems="center" gap={2}>
              {filteredModelType !== 'missing' && <PiWarningBold />}
              {t('modelManager.missingFiles')}
            </Flex>
          </MenuItem>
          {MODEL_CATEGORIES_AS_LIST.map((data) => (
            <ModelMenuItem key={data.category} data={data} />
          ))}
        </MenuList>
      </Menu>
    </MenuItem>
  );
});
ModelTypeSubMenu.displayName = 'ModelTypeSubMenu';

const ModelMenuItem = memo(({ data }: { data: ModelCategoryData }) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const filteredModelType = useAppSelector(selectFilteredModelType);
  const onClick = useCallback(() => {
    dispatch(setFilteredModelType(data.category));
  }, [data.category, dispatch]);
  return (
    <MenuItem
      bg={filteredModelType === data.category ? 'base.700' : 'transparent'}
      onClick={onClick}
      icon={filteredModelType === data.category ? <PiCheckBold /> : <PiCheckBold visibility="hidden" />}
    >
      {t(data.i18nKey)}
    </MenuItem>
  );
});
ModelMenuItem.displayName = 'ModelMenuItem';

export const ModelFilterMenu = memo(() => {
  const { t } = useTranslation();

  return (
    <Menu placement="bottom-end">
      <MenuButton as={Button} size="sm" rightIcon={<PiFunnelBold />}>
        {t('common.filtering', 'Filtering')}
      </MenuButton>
      <MenuList>
        <DirectionSubMenu />
        <SortBySubMenu />
        <ModelTypeSubMenu />
      </MenuList>
    </Menu>
  );
});

ModelFilterMenu.displayName = 'ModelFilterMenu';
