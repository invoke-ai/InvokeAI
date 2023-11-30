import { Flex, chakra, useColorModeValue } from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { memo } from 'react';
import { MiniMap } from 'reactflow';

const ChakraMiniMap = chakra(MiniMap);

const MinimapPanel = () => {
  const shouldShowMinimapPanel = useAppSelector(
    (state: RootState) => state.nodes.shouldShowMinimapPanel
  );

  const nodeColor = useColorModeValue(
    'var(--invokeai-colors-accent-300)',
    'var(--invokeai-colors-accent-600)'
  );

  const maskColor = useColorModeValue(
    'var(--invokeai-colors-blackAlpha-300)',
    'var(--invokeai-colors-blackAlpha-600)'
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
            backgroundColor: 'base.200 !important',
            borderRadius: 'base',
            _dark: {
              backgroundColor: 'base.500 !important',
            },
            svg: {
              borderRadius: 'inherit',
            },
          }}
          nodeColor={nodeColor}
          maskColor={maskColor}
        />
      )}
    </Flex>
  );
};

export default memo(MinimapPanel);
