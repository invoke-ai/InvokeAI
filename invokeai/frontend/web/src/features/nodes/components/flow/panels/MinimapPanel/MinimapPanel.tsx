import type { SystemStyleObject } from '@chakra-ui/react';
import { chakra, Flex } from '@chakra-ui/react';
import { useAppSelector } from 'app/store/storeHooks';
import { memo } from 'react';
import { MiniMap } from 'reactflow';

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
  const shouldShowMinimapPanel = useAppSelector(
    (s) => s.nodes.shouldShowMinimapPanel
  );

  return (
    <Flex gap={2} position="absolute" bottom={2} insetInlineEnd={2}>
      {shouldShowMinimapPanel && (
        <ChakraMiniMap
          pannable
          zoomable
          nodeBorderRadius={15}
          sx={minimapStyles}
          nodeColor="var(--invokeai-colors-base-600)"
          maskColor="var(--invokeai-colors-blackAlpha-600)"
        />
      )}
    </Flex>
  );
};

export default memo(MinimapPanel);
