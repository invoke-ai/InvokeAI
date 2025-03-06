import { Flex, FormControl, FormLabel, Select } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectWorkflowOrderBy,
  selectWorkflowOrderDirection,
  workflowOrderByChanged,
  workflowOrderDirectionChanged,
} from 'features/nodes/store/workflowSlice';
import type { ChangeEvent } from 'react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { z } from 'zod';

const zOrderBy = z.enum(['opened_at', 'created_at', 'updated_at', 'name']);
type OrderBy = z.infer<typeof zOrderBy>;
const isOrderBy = (v: unknown): v is OrderBy => zOrderBy.safeParse(v).success;

const zDirection = z.enum(['ASC', 'DESC']);
type Direction = z.infer<typeof zDirection>;
const isDirection = (v: unknown): v is Direction => zDirection.safeParse(v).success;

export const WorkflowSortControl = () => {
  const { t } = useTranslation();

  const orderBy = useAppSelector(selectWorkflowOrderBy);
  const direction = useAppSelector(selectWorkflowOrderDirection);

  const ORDER_BY_LABELS = useMemo(
    () => ({
      opened_at: t('workflows.opened'),
      created_at: t('workflows.created'),
      updated_at: t('workflows.updated'),
      name: t('workflows.name'),
    }),
    [t]
  );

  const DIRECTION_LABELS = useMemo(
    () => ({
      ASC: t('workflows.ascending'),
      DESC: t('workflows.descending'),
    }),
    [t]
  );

  const dispatch = useAppDispatch();

  const onChangeOrderBy = useCallback(
    (e: ChangeEvent<HTMLSelectElement>) => {
      if (!isOrderBy(e.target.value)) {
        return;
      }
      dispatch(workflowOrderByChanged(e.target.value));
    },
    [dispatch]
  );

  const onChangeDirection = useCallback(
    (e: ChangeEvent<HTMLSelectElement>) => {
      if (!isDirection(e.target.value)) {
        return;
      }
      dispatch(workflowOrderDirectionChanged(e.target.value));
    },
    [dispatch]
  );

  return (
    <Flex flexDir="row" gap={6}>
      <FormControl orientation="horizontal" gap={0} w="auto">
        <FormLabel>{t('common.orderBy')}</FormLabel>
        <Select value={orderBy ?? 'opened_at'} onChange={onChangeOrderBy} size="sm">
          <option value="opened_at">{ORDER_BY_LABELS['opened_at']}</option>
          <option value="created_at">{ORDER_BY_LABELS['created_at']}</option>
          <option value="updated_at">{ORDER_BY_LABELS['updated_at']}</option>
          <option value="name">{ORDER_BY_LABELS['name']}</option>
        </Select>
      </FormControl>
      <FormControl orientation="horizontal" gap={0} w="auto">
        <FormLabel>{t('common.direction')}</FormLabel>
        <Select value={direction} onChange={onChangeDirection} size="sm">
          <option value="ASC">{DIRECTION_LABELS['ASC']}</option>
          <option value="DESC">{DIRECTION_LABELS['DESC']}</option>
        </Select>
      </FormControl>
    </Flex>
  );
};
