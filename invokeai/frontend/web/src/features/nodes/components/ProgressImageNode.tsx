import { Flex, Image } from '@chakra-ui/react';
import { NodeProps } from 'reactflow';
import { InvocationValue } from '../types/types';

import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { memo } from 'react';
import IAINodeHeader from './IAINode/IAINodeHeader';
import IAINodeResizer from './IAINode/IAINodeResizer';
import NodeWrapper from './NodeWrapper';

const ProgressImageNode = (props: NodeProps<InvocationValue>) => {
  const progressImage = useAppSelector((state) => state.system.progressImage);
  const { selected } = props;

  return (
    <NodeWrapper selected={selected}>
      <IAINodeHeader
        title="Progress Image"
        description="Displays the progress image in the Node Editor"
      />

      <Flex
        sx={{
          flexDirection: 'column',
          borderBottomRadius: 'md',
          p: 2,
          bg: 'base.200',
          _dark: { bg: 'base.800' },
        }}
      >
        {progressImage ? (
          <Image
            src={progressImage.dataURL}
            sx={{
              w: 'full',
              h: 'full',
              objectFit: 'contain',
            }}
          />
        ) : (
          <Flex
            sx={{
              w: 'full',
              h: 'full',
              minW: 32,
              minH: 32,
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <IAINoContentFallback />
          </Flex>
        )}
      </Flex>
      <IAINodeResizer
        maxHeight={progressImage?.height ?? 512}
        maxWidth={progressImage?.width ?? 512}
      />
    </NodeWrapper>
  );
};

export default memo(ProgressImageNode);
