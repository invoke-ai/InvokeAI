import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import {
  Combobox,
  Flex,
  FormControl,
  FormLabel,
  IconButton,
  Popover,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $projectId } from 'app/store/nanostores/projectId';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectWorkflowOrderBy,
  selectWorkflowOrderDirection,
  workflowOrderByChanged,
  workflowOrderDirectionChanged,
} from 'features/nodes/store/workflowSlice';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiSortAscendingBold, PiSortDescendingBold } from 'react-icons/pi';
import { z } from 'zod';

const zOrderBy = z.enum(['opened_at', 'created_at', 'updated_at', 'name']);
type OrderBy = z.infer<typeof zOrderBy>;
const isOrderBy = (v: unknown): v is OrderBy => zOrderBy.safeParse(v).success;

const zDirection = z.enum(['ASC', 'DESC']);
type Direction = z.infer<typeof zDirection>;
const isDirection = (v: unknown): v is Direction => zDirection.safeParse(v).success;

export const WorkflowSortControl = () => {
  const projectId = useStore($projectId);
  const { t } = useTranslation();

  const orderBy = useAppSelector(selectWorkflowOrderBy);
  const direction = useAppSelector(selectWorkflowOrderDirection);

  const ORDER_BY_OPTIONS: ComboboxOption[] = useMemo(
    () => [
      { value: 'opened_at', label: t('workflows.opened') },
      { value: 'created_at', label: t('workflows.created') },
      { value: 'updated_at', label: t('workflows.updated') },
      { value: 'name', label: t('workflows.name') },
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

  const orderByOptions = useMemo(() => {
    return projectId ? ORDER_BY_OPTIONS.filter((option) => option.value !== 'opened_at') : ORDER_BY_OPTIONS;
  }, [projectId, ORDER_BY_OPTIONS]);

  const onChangeOrderBy = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isOrderBy(v?.value) || v.value === orderBy) {
        return;
      }
      dispatch(workflowOrderByChanged(v.value));
    },
    [orderBy, dispatch]
  );
  const valueOrderBy = useMemo(() => {
    return orderByOptions.find((o) => o.value === orderBy) || orderByOptions[0];
  }, [orderBy, orderByOptions]);

  const onChangeDirection = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isDirection(v?.value) || v.value === direction) {
        return;
      }
      dispatch(workflowOrderDirectionChanged(v.value));
    },
    [direction, dispatch]
  );
  const valueDirection = useMemo(
    () => DIRECTION_OPTIONS.find((o) => o.value === direction),
    [direction, DIRECTION_OPTIONS]
  );

  return (
    <Popover placement="bottom">
      <PopoverTrigger>
        <IconButton
          tooltip={`Sorting by ${valueOrderBy?.label} ${valueDirection?.label}`}
          aria-label="Sort Workflow Library"
          icon={direction === 'ASC' ? <PiSortAscendingBold /> : <PiSortDescendingBold />}
          variant="ghost"
        />
      </PopoverTrigger>

      <PopoverContent>
        <PopoverBody>
          <Flex flexDir="column" gap={4}>
            <FormControl orientation="horizontal" gap={1}>
              <FormLabel>{t('common.orderBy')}</FormLabel>
              <Combobox value={valueOrderBy} options={orderByOptions} onChange={onChangeOrderBy} />
            </FormControl>
            <FormControl orientation="horizontal" gap={1}>
              <FormLabel>{t('common.direction')}</FormLabel>
              <Combobox value={valueDirection} options={DIRECTION_OPTIONS} onChange={onChangeDirection} />
            </FormControl>
          </Flex>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
};
