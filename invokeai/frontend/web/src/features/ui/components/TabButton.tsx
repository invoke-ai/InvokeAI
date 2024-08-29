import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { setActiveTab } from 'features/ui/store/uiSlice';
import type { TabName } from 'features/ui/store/uiTypes';
import { forwardRef, memo, type ReactElement, useCallback } from 'react';

export const TabButton = memo(
  forwardRef(({ tab, icon, label }: { tab: TabName; icon: ReactElement; label: string }, ref) => {
    const dispatch = useAppDispatch();
    const activeTabName = useAppSelector(selectActiveTab);
    const onClick = useCallback(() => {
      dispatch(setActiveTab(tab));
    }, [dispatch, tab]);

    return (
      <Tooltip label={label} placement="end">
        <IconButton
          ref={ref}
          p={0}
          onClick={onClick}
          icon={icon}
          size="md"
          fontSize="24px"
          variant="appTab"
          data-selected={activeTabName === tab}
          aria-label={label}
          data-testid={label}
        />
      </Tooltip>
    );
  })
);

TabButton.displayName = 'TabButton';
