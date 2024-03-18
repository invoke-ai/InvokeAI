import { Box, Flex } from '@invoke-ai/ui-library';
import type { AnimationProps } from 'framer-motion';
import { motion } from 'framer-motion';
import type { ReactNode } from 'react';
import { memo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { v4 as uuidv4 } from 'uuid';
type Props = {
  isOver: boolean;
  label?: ReactNode;
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

const IAIDropOverlay = (props: Props) => {
  const { t } = useTranslation();
  const { isOver, label = t('gallery.drop') } = props;
  const motionId = useRef(uuidv4());
  return (
    <motion.div key={motionId.current} initial={initial} animate={animate} exit={exit}>
      <Flex position="absolute" top={0} insetInlineStart={0} w="full" h="full">
        <Flex
          position="absolute"
          top={0}
          insetInlineStart={0}
          w="full"
          h="full"
          bg="base.900"
          opacity={0.7}
          borderRadius="base"
          alignItems="center"
          justifyContent="center"
          transitionProperty="common"
          transitionDuration="0.1s"
        />

        <Flex
          position="absolute"
          top={0.5}
          insetInlineStart={0.5}
          insetInlineEnd={0.5}
          bottom={0.5}
          opacity={1}
          borderWidth={2}
          borderColor={isOver ? 'base.50' : 'base.300'}
          borderRadius="lg"
          borderStyle="dashed"
          transitionProperty="common"
          transitionDuration="0.1s"
          alignItems="center"
          justifyContent="center"
        >
          <Box
            fontSize="2xl"
            fontWeight="semibold"
            transform={isOver ? 'scale(1.1)' : 'scale(1)'}
            color={isOver ? 'base.50' : 'base.300'}
            transitionProperty="common"
            transitionDuration="0.1s"
          >
            {label}
          </Box>
        </Flex>
      </Flex>
    </motion.div>
  );
};

export default memo(IAIDropOverlay);
