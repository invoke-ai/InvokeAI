import { Flex, Image } from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { memo } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { NodeProps, OnResize } from 'reactflow';
import { setProgressNodeSize } from '../store/nodesSlice';
import IAINodeHeader from './IAINode/IAINodeHeader';
import IAINodeResizer from './IAINode/IAINodeResizer';
import NodeWrapper from './NodeWrapper';

const ProgressImageNode = (props: NodeProps) => {
  const progressImage = useSelector(
    (state: RootState) => state.system.progressImage
  );
  const progressNodeSize = useSelector(
    (state: RootState) => state.nodes.progressNodeSize
  );
  const dispatch = useDispatch();
  const { selected } = props;

  const handleResize: OnResize = (_, newSize) => {
    dispatch(setProgressNodeSize(newSize));
  };

  return (
    <NodeWrapper selected={selected}>
      <IAINodeHeader
        title="Progress Image"
        description="Displays the progress image in the Node Editor"
      />
      <Flex
        sx={{
          flexDirection: 'column',
          flexShrink: 0,
          borderBottomRadius: 'md',
          bg: 'base.200',
          _dark: { bg: 'base.800' },
          width: progressNodeSize.width - 2,
          height: progressNodeSize.height - 2,
          minW: 250,
          minH: 250,
          overflow: 'hidden',
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
              minW: 250,
              minH: 250,
              width: progressNodeSize.width - 2,
              height: progressNodeSize.height - 2,
            }}
          >
            <IAINoContentFallback />
          </Flex>
        )}
      </Flex>
      <IAINodeResizer onResize={handleResize} />
    </NodeWrapper>
  );
};

export default memo(ProgressImageNode);
