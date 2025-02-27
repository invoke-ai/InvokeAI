import {
  Flex,
  FormControl,
  FormLabel,
  IconButton,
  Popover,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  Select,
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
import type { ChangeEvent } from 'react';
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

  // In OSS, we don't have the concept of "opened_at" for workflows. This is only available in the Enterprise version.
  const defaultOrderBy = projectId !== undefined ? 'opened_at' : 'created_at';

  return (
    <Popover placement="bottom">
      <PopoverTrigger>
        <IconButton
          tooltip={`Sorting by ${ORDER_BY_LABELS[orderBy ?? defaultOrderBy]} ${DIRECTION_LABELS[direction]}`}
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
              <Select value={orderBy ?? defaultOrderBy} onChange={onChangeOrderBy} size="sm">
                {projectId !== undefined && <option value="opened_at">{ORDER_BY_LABELS['opened_at']}</option>}
                <option value="created_at">{ORDER_BY_LABELS['created_at']}</option>
                <option value="updated_at">{ORDER_BY_LABELS['updated_at']}</option>
                <option value="name">{ORDER_BY_LABELS['name']}</option>
              </Select>
            </FormControl>
            <FormControl orientation="horizontal" gap={1}>
              <FormLabel>{t('common.direction')}</FormLabel>
              <Select value={direction} onChange={onChangeDirection} size="sm">
                <option value="ASC">{DIRECTION_LABELS['ASC']}</option>
                <option value="DESC">{DIRECTION_LABELS['DESC']}</option>
              </Select>
            </FormControl>
          </Flex>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
};
