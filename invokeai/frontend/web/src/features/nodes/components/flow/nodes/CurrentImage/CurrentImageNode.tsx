import { useState, PropsWithChildren, memo } from 'react';
import { useSelector } from 'react-redux';
import { createSelector } from '@reduxjs/toolkit';
import { Flex, Image, Text } from '@chakra-ui/react';
import { motion } from 'framer-motion';
import { NodeProps } from 'reactflow';
import NodeWrapper from '../common/NodeWrapper';
import NextPrevImageButtons from 'features/gallery/components/NextPrevImageButtons';
import IAIDndImage from 'common/components/IAIDndImage';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { DRAG_HANDLE_CLASSNAME } from 'features/nodes/types/constants';
import { stateSelector } from 'app/store/store';

const selector = createSelector(stateSelector, ({ system, gallery }) => {
  const imageDTO = gallery.selection[gallery.selection.length - 1];

  return {
    imageDTO,
    progressImage: system.progressImage,
  };
});

const CurrentImageNode = (props: NodeProps) => {
  const { progressImage, imageDTO } = useSelector(selector);

  if (progressImage) {
    return (
      <Wrapper nodeProps={props}>
        <Image
          src={progressImage.dataURL}
          sx={{
            w: 'full',
            h: 'full',
            objectFit: 'contain',
            borderRadius: 'base',
          }}
        />
      </Wrapper>
    );
  }

  if (imageDTO) {
    return (
      <Wrapper nodeProps={props}>
        <IAIDndImage imageDTO={imageDTO} isDragDisabled useThumbailFallback />
      </Wrapper>
    );
  }

  return (
    <Wrapper nodeProps={props}>
      <IAINoContentFallback />
    </Wrapper>
  );
};

export default memo(CurrentImageNode);

const Wrapper = (props: PropsWithChildren<{ nodeProps: NodeProps }>) => {
  const [isHovering, setIsHovering] = useState(false);

  const handleMouseEnter = () => {
    setIsHovering(true);
  };

  const handleMouseLeave = () => {
    setIsHovering(false);
  };

  return (
    <NodeWrapper
      nodeId={props.nodeProps.id}
      selected={props.nodeProps.selected}
      width={384}
    >
      <Flex
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        className={DRAG_HANDLE_CLASSNAME}
        sx={{
          position: 'relative',
          flexDirection: 'column',
        }}
      >
        <Flex
          layerStyle="nodeHeader"
          sx={{
            borderTopRadius: 'base',
            alignItems: 'center',
            justifyContent: 'center',
            h: 8,
          }}
        >
          <Text
            sx={{
              fontSize: 'sm',
              fontWeight: 600,
              color: 'base.700',
              _dark: { color: 'base.200' },
            }}
          >
            Current Image
          </Text>
        </Flex>
        <Flex
          layerStyle="nodeBody"
          sx={{
            w: 'full',
            h: 'full',
            borderBottomRadius: 'base',
            p: 2,
          }}
        >
          {props.children}
          {isHovering && (
            <motion.div
              key="nextPrevButtons"
              initial={{
                opacity: 0,
              }}
              animate={{
                opacity: 1,
                transition: { duration: 0.1 },
              }}
              exit={{
                opacity: 0,
                transition: { duration: 0.1 },
              }}
              style={{
                position: 'absolute',
                top: 40,
                left: -2,
                right: -2,
                bottom: 0,
                pointerEvents: 'none',
              }}
            >
              <NextPrevImageButtons />
            </motion.div>
          )}
        </Flex>
      </Flex>
    </NodeWrapper>
  );
};
