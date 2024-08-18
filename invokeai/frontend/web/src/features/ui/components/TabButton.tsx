import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import type { InvokeTabName } from 'features/ui/store/tabMap';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { memo, type ReactElement, useCallback } from 'react';

export const TabButton = memo(({ tab, icon, label }: { tab: InvokeTabName; icon: ReactElement; label: string }) => {
  const dispatch = useAppDispatch();
  const activeTabName = useAppSelector(activeTabNameSelector);
  const onClick = useCallback(() => {
    dispatch(setActiveTab(tab));
  }, [dispatch, tab]);

  return (
    <Tooltip label={label} placement="end">
      <IconButton
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
});

TabButton.displayName = 'TabButton';
