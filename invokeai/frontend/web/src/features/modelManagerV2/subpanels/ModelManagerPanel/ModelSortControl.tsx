import { Flex, IconButton, Select } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectOrderBy,
  selectSortDirection,
  setOrderBy,
  setSortDirection,
} from 'features/modelManagerV2/store/modelManagerV2Slice';
import type { ChangeEvent } from 'react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiSortAscendingBold, PiSortDescendingBold } from 'react-icons/pi';
import { z } from 'zod';

const zOrderBy = z.enum(['default', 'name', 'type', 'base', 'size', 'created_at', 'updated_at', 'path', 'format']);
type OrderBy = z.infer<typeof zOrderBy>;
const isOrderBy = (v: unknown): v is OrderBy => zOrderBy.safeParse(v).success;

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

export const ModelSortControl = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const orderBy = useAppSelector(selectOrderBy);
  const direction = useAppSelector(selectSortDirection);

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

  const onChangeOrderBy = useCallback(
    (e: ChangeEvent<HTMLSelectElement>) => {
      if (!isOrderBy(e.target.value)) {
        return;
      }
      dispatch(setOrderBy(e.target.value));
    },
    [dispatch]
  );

  const toggleDirection = useCallback(() => {
    dispatch(setSortDirection(direction === 'asc' ? 'desc' : 'asc'));
  }, [dispatch, direction]);

  return (
    <Flex alignItems="center" gap={2}>
      <Select value={orderBy} onChange={onChangeOrderBy} size="sm">
        {ORDER_BY_OPTIONS.map((option) => (
          <option key={option} value={option}>
            {ORDER_BY_LABELS[option]}
          </option>
        ))}
      </Select>
      <IconButton
        aria-label={t('common.direction')}
        icon={direction === 'asc' ? <PiSortAscendingBold /> : <PiSortDescendingBold />}
        size="sm"
        variant="ghost"
        onClick={toggleDirection}
      />
    </Flex>
  );
};
