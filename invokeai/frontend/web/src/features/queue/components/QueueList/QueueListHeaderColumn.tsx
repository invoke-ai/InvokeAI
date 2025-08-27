import { Flex, IconButton, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import type * as CSS from 'csstype';
import type { SortBy } from 'features/queue/store/queueSlice';
import {
  selectQueueSortBy,
  selectQueueSortOrder,
  sortByChanged,
  sortOrderChanged
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
        <SortColumnIcon field={field} displayName={displayName} isMouseHoveringColumn={isMouseHoveringColumn} />
      )}
    </Flex>
  );
};

export default memo(QueueListHeaderColumn);

type SortColumnIconProps = {
  field: SortBy;
  displayName: string;
  isMouseHoveringColumn: boolean;
};

const SortColumnIcon = memo(({ field, displayName, isMouseHoveringColumn }: SortColumnIconProps) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const sortBy = useSelector(selectQueueSortBy);
  const sortOrder = useSelector(selectQueueSortOrder);
  const isSortByColumn = useMemo(() => sortBy === field, [sortBy, field]);
  const isShown = useMemo(() => isSortByColumn || isMouseHoveringColumn, [isSortByColumn, isMouseHoveringColumn]);

  const handleClickSortByColumn = useCallback(() => {
    dispatch(sortByChanged(field));
  }, [dispatch, field]);
  const handleClickSortOrderAscending = useCallback(() => {
    dispatch(sortOrderChanged('asc'));
  }, [dispatch]);
  const handleClickSortOrderDescending = useCallback(() => {
    dispatch(sortOrderChanged('desc'));
  }, [dispatch]);

  return (
    isShown && (
      <>
        {sortOrder === 'asc' && (
          <IconButton
            size="sm"
            variant="link"
            alignSelf="stretch"
            onClick={isSortByColumn ? handleClickSortOrderDescending : handleClickSortByColumn}
            tooltip={isSortByColumn ? t('queue.sortOrderDescending') : t('queue.sortBy', { column: displayName })}
            aria-label={t('queue.sortColumn')}
            icon={<PiSortAscendingBold />}
          />
        )}
        {sortOrder === 'desc' && (
          <IconButton
            size="sm"
            variant="link"
            alignSelf="stretch"
            onClick={isSortByColumn ? handleClickSortOrderAscending : handleClickSortByColumn}
            tooltip={isSortByColumn ? t('queue.sortOrderAscending') : t('queue.sortBy', { column: displayName })}
            aria-label={t('queue.sortColumn')}
            icon={<PiSortDescendingBold />}
          />
        )}
      </>
    )
  );
});
SortColumnIcon.displayName = 'SortColumnIcon';
