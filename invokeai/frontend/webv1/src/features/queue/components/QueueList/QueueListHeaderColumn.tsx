import type { FlexProps } from '@invoke-ai/ui-library';
import { Flex, IconButton, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import type * as CSS from 'csstype';
import { selectQueueSortOrder, sortOrderChanged } from 'features/queue/store/queueSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiSortAscendingBold, PiSortDescendingBold } from 'react-icons/pi';
import { useSelector } from 'react-redux';

interface QueueListHeaderColumnProps extends FlexProps {
  field?: string;
  displayName: string;
  alignItems?: CSS.Property.AlignItems;
  ps?: CSS.Property.PaddingInlineStart | number;
  w?: CSS.Property.Width | number;
}

const QueueListHeaderColumn = ({ field, displayName, alignItems, ps, w, ...props }: QueueListHeaderColumnProps) => {
  return (
    <Flex paddingInlineStart={ps} width={w} alignItems={alignItems} {...props}>
      <Text variant="subtext">{displayName}</Text>
      {!!field && <ColumnSortIcon />}
    </Flex>
  );
};

export default memo(QueueListHeaderColumn);

const ColumnSortIcon = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const sortOrder = useSelector(selectQueueSortOrder);
  const tooltip = useMemo(() => {
    return sortOrder === 'ASC' ? t('queue.sortOrderAscending') : t('queue.sortOrderDescending');
  }, [sortOrder, t]);

  // PiSortDescendingBold is used for ascending because the arrow points up
  const icon = useMemo(() => (sortOrder === 'ASC' ? <PiSortDescendingBold /> : <PiSortAscendingBold />), [sortOrder]);

  const handleClickSortColumn = useCallback(() => {
    if (sortOrder === 'ASC') {
      dispatch(sortOrderChanged('DESC'));
    } else {
      dispatch(sortOrderChanged('ASC'));
    }
  }, [sortOrder, dispatch]);

  return (
    <IconButton
      size="sm"
      variant="link"
      alignSelf="stretch"
      onClick={handleClickSortColumn}
      tooltip={tooltip}
      aria-label={t('queue.sortColumn')}
      icon={icon}
    />
  );
});
ColumnSortIcon.displayName = 'ColumnSortIcon';
