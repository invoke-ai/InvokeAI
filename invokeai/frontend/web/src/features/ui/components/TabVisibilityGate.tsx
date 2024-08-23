import { Box } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import type { TabName } from "features/ui/store/uiTypes";
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

export const TabVisibilityGate = memo(({ tab, children }: PropsWithChildren<{ tab: TabName }>) => {
  const activeTabName = useAppSelector((s) => s.ui.activeTab);

  return (
    <Box
      display={activeTabName === tab ? undefined : 'none'}
      pointerEvents={activeTabName === tab ? undefined : 'none'}
      userSelect={activeTabName === tab ? undefined : 'none'}
      hidden={activeTabName !== tab}
      w="full"
      h="full"
      position="absolute"
      top={0}
      right={0}
      bottom={0}
      left={0}
    >
      {children}
    </Box>
  );
});

TabVisibilityGate.displayName = 'TabVisibilityGate';
