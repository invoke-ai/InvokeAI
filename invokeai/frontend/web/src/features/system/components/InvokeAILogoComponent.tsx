import { Flex, Image, Text } from '@chakra-ui/react';
import InvokeAILogoImage from 'assets/images/logo.png';
import { AnimatePresence, motion } from 'framer-motion';
import { memo, useRef } from 'react';
import { useHoverDirty } from 'react-use';
import { useGetAppVersionQuery } from 'services/api/endpoints/appInfo';

interface Props {
  showVersion?: boolean;
}

const InvokeAILogoComponent = ({ showVersion = true }: Props) => {
  const { data: appVersion } = useGetAppVersionQuery();
  const ref = useRef(null);
  const isHovered = useHoverDirty(ref);

  return (
    <Flex alignItems="center" gap={5} ps={1} ref={ref}>
      <Image
        src={InvokeAILogoImage}
        alt="invoke-ai-logo"
        sx={{
          w: '32px',
          h: '32px',
          minW: '32px',
          minH: '32px',
          userSelect: 'none',
        }}
      />
      <Flex sx={{ gap: 3, alignItems: 'center' }}>
        <Text sx={{ fontSize: 'xl', userSelect: 'none' }}>
          invoke <strong>ai</strong>
        </Text>
        <AnimatePresence>
          {showVersion && isHovered && appVersion && (
            <motion.div
              key="statusText"
              initial={{
                opacity: 0,
              }}
              animate={{
                opacity: 1,
                transition: { duration: 0.15 },
              }}
              exit={{
                opacity: 0,
                transition: { delay: 0.8 },
              }}
            >
              <Text
                sx={{
                  fontWeight: 600,
                  marginTop: 1,
                  color: 'base.300',
                  fontSize: 14,
                }}
                variant="subtext"
              >
                {appVersion.version}
              </Text>
            </motion.div>
          )}
        </AnimatePresence>
      </Flex>
    </Flex>
  );
};

export default memo(InvokeAILogoComponent);
