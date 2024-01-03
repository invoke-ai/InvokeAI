/* eslint-disable i18next/no-literal-string */
import { Flex, Image } from '@chakra-ui/react';
import InvokeLogoYellow from 'assets/images/invoke-key-ylw-sm.svg';
import { InvText } from 'common/components/InvText/wrapper';
import type { AnimationProps } from 'framer-motion';
import { AnimatePresence, motion } from 'framer-motion';
import { memo, useRef } from 'react';
import { useHoverDirty } from 'react-use';
import { useGetAppVersionQuery } from 'services/api/endpoints/appInfo';

const InvokeAILogoComponent = () => {
  const { data: appVersion } = useGetAppVersionQuery();
  const ref = useRef(null);
  const isHovered = useHoverDirty(ref);

  return (
    <Flex alignItems="center" gap={5} ps={1} ref={ref}>
      <Image
        src={InvokeLogoYellow}
        alt="invoke-logo"
        w="24px"
        h="24px"
        minW="24px"
        minH="24px"
        userSelect="none"
      />
      <Flex gap={3} alignItems="center">
        <AnimatePresence>
          {isHovered && appVersion && (
            <motion.div
              key="statusText"
              initial={initial}
              animate={animate}
              exit={exit}
            >
              <InvText fontWeight="semibold" marginTop={1} color="base.300">
                v{appVersion.version}
              </InvText>
            </motion.div>
          )}
        </AnimatePresence>
      </Flex>
    </Flex>
  );
};

export default memo(InvokeAILogoComponent);

const initial: AnimationProps['initial'] = {
  opacity: 0,
};
const animate: AnimationProps['animate'] = {
  opacity: 1,
  transition: { duration: 0.1 },
};
const exit: AnimationProps['exit'] = {
  opacity: 0,
  transition: { delay: 0.8 },
};
