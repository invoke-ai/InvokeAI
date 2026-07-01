import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { chakra, Flex } from '@invoke-ai/ui-library';
import { MiniMap } from '@xyflow/react';
import { useAppSelector } from 'app/store/storeHooks';
import { selectShouldShowMinimapPanel } from 'features/nodes/store/workflowSettingsSlice';
import { memo } from 'react';

const ChakraMiniMap = chakra(MiniMap);

const minimapStyles: SystemStyleObject = {
  m: '0 !important',
  borderRadius: 'base',
  backgroundColor: 'base.500 !important',
  svg: {
    borderRadius: 'inherit',
  },
};

const MinimapPanel = () => {
  const shouldShowMinimapPanel = useAppSelector(selectShouldShowMinimapPanel);

  return (
    <Flex gap={2} position="absolute" bottom={2} insetInlineEnd={2}>
      {shouldShowMinimapPanel && (
        <ChakraMiniMap
          pannable
          zoomable
          nodeBorderRadius={15}
          sx={minimapStyles}
          nodeColor="var(--invoke-colors-base-600)"
          maskColor="var(--invoke-colors-blackAlpha-600)"
        />
      )}
    </Flex>
  );
};

export default memo(MinimapPanel);
