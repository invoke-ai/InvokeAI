import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useCallbackOnDragEnter } from 'common/hooks/useCallbackOnDragEnter';
import { selectActiveTab } from 'features/controlLayers/store/selectors';
import type { TabName } from 'features/controlLayers/store/types';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import type { ReactElement } from 'react';
import { memo, useCallback, useRef } from 'react';

const sx: SystemStyleObject = {
  '&[data-selected=true]': {
    svg: { fill: 'invokeYellow.300' },
  },
};

export const TabButton = memo(({ tab, icon, label }: { tab: TabName; icon: ReactElement; label: string }) => {
  const ref = useRef<HTMLDivElement>(null);
  const activeTabName = useAppSelector(selectActiveTab);
  const selectTab = useCallback(() => {
    navigationApi.switchToTab(tab);
  }, [tab]);
  useCallbackOnDragEnter(selectTab, ref, 300);

  return (
    <Tooltip label={label} placement="end">
      <IconButton
        p={0}
        ref={ref}
        onClick={selectTab}
        icon={icon}
        size="md"
        fontSize="24px"
        variant="link"
        data-selected={activeTabName === tab}
        aria-label={label}
        data-testid={label}
        sx={sx}
      />
    </Tooltip>
  );
});

TabButton.displayName = 'TabButton';
