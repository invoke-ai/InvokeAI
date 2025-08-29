import { Flex, IconButton, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import type * as CSS from 'csstype';
import type { SortBy } from 'features/queue/store/queueSlice';
import {
  selectQueueSortBy,
  selectQueueSortOrder,
  sortByChanged,
  sortOrderChanged,
} from 'features/queue/store/queueSlice';
import { memo, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiSortAscendingBold, PiSortDescendingBold } from 'react-icons/pi';
import { useSelector } from 'react-redux';

type QueueListHeaderColumnProps = {
  field?: SortBy;
  displayName: string;
  alignItems?: CSS.Property.AlignItems;
  ps?: CSS.Property.PaddingInlineStart | number;
  w?: CSS.Property.Width | number;
};

const QueueListHeaderColumn = ({ field, displayName, alignItems, ps, w }: QueueListHeaderColumnProps) => {
  const [isMouseHoveringColumn, setIsMouseHoveringColumn] = useState(false);

  const handleMouseEnterColumn = useCallback(() => {
    setIsMouseHoveringColumn(true);
  }, [setIsMouseHoveringColumn]);
  const handleMouseLeaveColumn = useCallback(() => {
    setIsMouseHoveringColumn(false);
  }, [setIsMouseHoveringColumn]);

  return (
    <Flex
      paddingInlineStart={ps}
      width={w}
      alignItems={alignItems}
      onMouseEnter={handleMouseEnterColumn}
      onMouseLeave={handleMouseLeaveColumn}
    >
      <Text variant="subtext">{displayName}</Text>
      {!!field && (
        <ColumnSortIcon field={field} displayName={displayName} isMouseHoveringColumn={isMouseHoveringColumn} />
      )}
    </Flex>
  );
};

export default memo(QueueListHeaderColumn);

type ColumnSortIconProps = {
  field: SortBy;
  displayName: string;
  isMouseHoveringColumn: boolean;
};

const ColumnSortIcon = memo(({ field, displayName, isMouseHoveringColumn }: ColumnSortIconProps) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const sortBy = useSelector(selectQueueSortBy);
  const sortOrder = useSelector(selectQueueSortOrder);
  const isSortByColumn = useMemo(() => sortBy === field, [sortBy, field]);
  const isShown = useMemo(() => isSortByColumn || isMouseHoveringColumn, [isSortByColumn, isMouseHoveringColumn]);
  const tooltip = useMemo(() => {
    if (isSortByColumn) {
      return sortOrder === 'asc' ? t('queue.sortOrderAscending') : t('queue.sortOrderDescending');
    }
    return t('queue.sortBy', { column: displayName });
  }, [isSortByColumn, sortOrder, t, displayName]);
  const icon = useMemo(() => (sortOrder === 'asc' ? <PiSortAscendingBold /> : <PiSortDescendingBold />), [sortOrder]);

  const handleClickSortColumn = useCallback(() => {
    if (isSortByColumn) {
      if (sortOrder === 'asc') {
        dispatch(sortOrderChanged('desc'));
      } else {
        dispatch(sortOrderChanged('asc'));
      }
    } else {
      dispatch(sortByChanged(field));
    }
  }, [isSortByColumn, sortOrder, dispatch, field]);

  return (
    isShown && (
      <IconButton
        size="sm"
        variant="link"
        alignSelf="stretch"
        onClick={handleClickSortColumn}
        tooltip={tooltip}
        aria-label={t('queue.sortColumn')}
        icon={icon}
      />
    )
  );
});
ColumnSortIcon.displayName = 'ColumnSortIcon';
