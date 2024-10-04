import { Flex, Text } from '@invoke-ai/ui-library';
import type { AnimationProps } from 'framer-motion';
import { motion } from 'framer-motion';
import { memo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { v4 as uuidv4 } from 'uuid';
type Props = {
  isOver: boolean;
  label?: string;
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
      <Flex position="absolute" top={0} right={0} bottom={0} left={0}>
        <Flex
          position="absolute"
          top={0}
          right={0}
          bottom={0}
          left={0}
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
          right={0.5}
          bottom={0.5}
          left={0.5}
          opacity={1}
          borderWidth={1.5}
          borderColor={isOver ? 'invokeYellow.300' : 'base.500'}
          borderRadius="base"
          borderStyle="dashed"
          transitionProperty="common"
          transitionDuration="0.1s"
          alignItems="center"
          justifyContent="center"
          p={4}
        >
          <Text
            fontSize="lg"
            fontWeight="semibold"
            color={isOver ? 'invokeYellow.300' : 'base.500'}
            transitionProperty="common"
            transitionDuration="0.1s"
            textAlign="center"
          >
            {label}
          </Text>
        </Flex>
      </Flex>
    </motion.div>
  );
};

export default memo(IAIDropOverlay);
