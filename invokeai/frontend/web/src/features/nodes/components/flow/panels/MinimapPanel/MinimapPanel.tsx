import { chakra, Flex } from '@chakra-ui/react';
import type { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { memo } from 'react';
import { MiniMap } from 'reactflow';

const ChakraMiniMap = chakra(MiniMap);

const MinimapPanel = () => {
  const shouldShowMinimapPanel = useAppSelector(
    (state: RootState) => state.nodes.shouldShowMinimapPanel
  );

  return (
    <Flex sx={{ gap: 2, position: 'absolute', bottom: 2, insetInlineEnd: 2 }}>
      {shouldShowMinimapPanel && (
        <ChakraMiniMap
          pannable
          zoomable
          nodeBorderRadius={15}
          sx={{
            m: '0 !important',
            borderRadius: 'base',
            backgroundColor: 'base.500 !important',
            svg: {
              borderRadius: 'inherit',
            },
          }}
          nodeColor="var(--invokeai-colors-blue-600)"
          maskColor="var(--invokeai-colors-blackAlpha-600)"
        />
      )}
    </Flex>
  );
};

export default memo(MinimapPanel);
