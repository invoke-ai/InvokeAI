import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { IconButton, Tab, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useCallbackOnDragEnter } from 'common/hooks/useCallbackOnDragEnter';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { setActiveTab } from 'features/ui/store/uiSlice';
import type { TabName } from 'features/ui/store/uiTypes';
import type { ReactElement } from 'react';
import { memo, useCallback, useRef } from 'react';

const sx: SystemStyleObject = {
  '&[data-selected=true]': {
    svg: { fill: 'invokeYellow.300' },
  },
};

export const TabButton = memo(({ tab, icon, label }: { tab: TabName; icon: ReactElement; label: string }) => {
  const dispatch = useAppDispatch();
  const ref = useRef<HTMLDivElement>(null);
  const activeTabName = useAppSelector(selectActiveTab);
  const selectTab = useCallback(() => {
    dispatch(setActiveTab(tab));
  }, [dispatch, tab]);
  useCallbackOnDragEnter(selectTab, ref, 300);

  return (
    <Tooltip label={label} placement="end">
      <Tab
        as={IconButton}
        p={0}
        ref={ref}
        onClick={selectTab}
        icon={icon}
        size="md"
        fontSize="24px"
        variant="appTab"
        data-selected={activeTabName === tab}
        aria-label={label}
        data-testid={label}
        sx={sx}
      />
    </Tooltip>
  );
});

TabButton.displayName = 'TabButton';
