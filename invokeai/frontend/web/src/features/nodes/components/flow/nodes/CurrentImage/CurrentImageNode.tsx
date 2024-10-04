import { Flex, Image, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import IAIDndImage from 'common/components/IAIDndImage';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import NextPrevImageButtons from 'features/gallery/components/NextPrevImageButtons';
import { selectLastSelectedImage } from 'features/gallery/store/gallerySelectors';
import NodeWrapper from 'features/nodes/components/flow/nodes/common/NodeWrapper';
import { DRAG_HANDLE_CLASSNAME } from 'features/nodes/types/constants';
import type { AnimationProps } from 'framer-motion';
import { motion } from 'framer-motion';
import type { CSSProperties, PropsWithChildren } from 'react';
import { memo, useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { NodeProps } from 'reactflow';
import { $lastProgressEvent } from 'services/events/stores';

const CurrentImageNode = (props: NodeProps) => {
  const imageDTO = useAppSelector(selectLastSelectedImage);
  const lastProgressEvent = useStore($lastProgressEvent);

  if (lastProgressEvent?.image) {
    return (
      <Wrapper nodeProps={props}>
        <Image src={lastProgressEvent?.image.dataURL} w="full" h="full" objectFit="contain" borderRadius="base" />
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

  const handleMouseEnter = useCallback(() => {
    setIsHovering(true);
  }, []);

  const handleMouseLeave = useCallback(() => {
    setIsHovering(false);
  }, []);
  const { t } = useTranslation();
  return (
    <NodeWrapper nodeId={props.nodeProps.id} selected={props.nodeProps.selected} width={384}>
      <Flex
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        className={DRAG_HANDLE_CLASSNAME}
        position="relative"
        flexDirection="column"
      >
        <Flex layerStyle="nodeHeader" borderTopRadius="base" alignItems="center" justifyContent="center" h={8}>
          <Text fontSize="sm" fontWeight="semibold" color="base.200">
            {t('nodes.currentImage')}
          </Text>
        </Flex>
        <Flex layerStyle="nodeBody" w="full" h="full" borderBottomRadius="base" p={2}>
          {props.children}
          {isHovering && (
            <motion.div key="nextPrevButtons" initial={initial} animate={animate} exit={exit} style={styles}>
              <NextPrevImageButtons inset={2} />
            </motion.div>
          )}
        </Flex>
      </Flex>
    </NodeWrapper>
  );
};

const initial: AnimationProps['initial'] = {
  opacity: 0,
};
const animate: AnimationProps['animate'] = {
  opacity: 1,
  transition: { duration: 0.1 },
};
const exit: AnimationProps['exit'] = {
  opacity: 0,
  transition: { duration: 0.1 },
};
const styles: CSSProperties = {
  position: 'absolute',
  top: 40,
  left: -2,
  right: -2,
  bottom: 0,
  pointerEvents: 'none',
};
