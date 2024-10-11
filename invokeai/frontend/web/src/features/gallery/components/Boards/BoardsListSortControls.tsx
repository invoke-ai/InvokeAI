import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectBoardsListOrderBy, selectBoardsListOrderDir } from 'features/gallery/store/gallerySelectors';
import { boardsListOrderByChanged, boardsListOrderDirChanged } from 'features/gallery/store/gallerySlice';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { z } from 'zod';

const zOrderBy = z.enum(['created_at', 'board_name']);
type OrderBy = z.infer<typeof zOrderBy>;
const isOrderBy = (v: unknown): v is OrderBy => zOrderBy.safeParse(v).success;

const zDirection = z.enum(['ASC', 'DESC']);
type Direction = z.infer<typeof zDirection>;
const isDirection = (v: unknown): v is Direction => zDirection.safeParse(v).success;

export const BoardsListSortControls = () => {
  const { t } = useTranslation();

  const orderBy = useAppSelector(selectBoardsListOrderBy);
  const direction = useAppSelector(selectBoardsListOrderDir);

  const ORDER_BY_OPTIONS: ComboboxOption[] = useMemo(
    () => [
      { value: 'created_at', label: t('workflows.created') },
      { value: 'board_name', label: t('workflows.name') },
    ],
    [t]
  );

  const DIRECTION_OPTIONS: ComboboxOption[] = useMemo(
    () => [
      { value: 'ASC', label: t('workflows.ascending') },
      { value: 'DESC', label: t('workflows.descending') },
    ],
    [t]
  );

  const dispatch = useAppDispatch();

  const onChangeOrderBy = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isOrderBy(v?.value) || v.value === orderBy) {
        return;
      }
      dispatch(boardsListOrderByChanged(v.value));
    },
    [orderBy, dispatch]
  );
  const valueOrderBy = useMemo(() => {
    return ORDER_BY_OPTIONS.find((o) => o.value === orderBy) || ORDER_BY_OPTIONS[0];
  }, [orderBy, ORDER_BY_OPTIONS]);

  const onChangeDirection = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isDirection(v?.value) || v.value === direction) {
        return;
      }
      dispatch(boardsListOrderDirChanged(v.value));
    },
    [direction, dispatch]
  );
  const valueDirection = useMemo(
    () => DIRECTION_OPTIONS.find((o) => o.value === direction),
    [direction, DIRECTION_OPTIONS]
  );

  return (
    <Flex flexDir="column" gap={4}>
      <FormControl orientation="horizontal" gap={1}>
        <FormLabel>{t('common.orderBy')}</FormLabel>
        <Combobox isSearchable={false} value={valueOrderBy} options={ORDER_BY_OPTIONS} onChange={onChangeOrderBy} />
      </FormControl>
      <FormControl orientation="horizontal" gap={1}>
        <FormLabel>{t('common.direction')}</FormLabel>
        <Combobox
          isSearchable={false}
          value={valueDirection}
          options={DIRECTION_OPTIONS}
          onChange={onChangeDirection}
        />
      </FormControl>
    </Flex>
  );
};
