import { ChevronUpIcon } from '@chakra-ui/icons';
import {
  Box,
  Collapse,
  Flex,
  Spacer,
  Text,
  useColorMode,
  useDisclosure,
} from '@chakra-ui/react';
import { AnimatePresence, motion } from 'framer-motion';
import { PropsWithChildren, memo } from 'react';
import { mode } from 'theme/util/mode';

export type IAIToggleCollapseProps = PropsWithChildren & {
  label: string;
  activeLabel?: string;
  defaultIsOpen?: boolean;
};

const IAICollapse = (props: IAIToggleCollapseProps) => {
  const { label, activeLabel, children, defaultIsOpen = false } = props;
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen });
  const { colorMode } = useColorMode();

  return (
    <Box>
      <Flex
        onClick={onToggle}
        sx={{
          alignItems: 'center',
          p: 2,
          px: 4,
          gap: 2,
          borderTopRadius: 'base',
          borderBottomRadius: isOpen ? 0 : 'base',
          bg: mode('base.250', 'base.750')(colorMode),
          color: mode('base.900', 'base.100')(colorMode),
          _hover: {
            bg: mode('base.300', 'base.700')(colorMode),
          },
          fontSize: 'sm',
          fontWeight: 600,
          cursor: 'pointer',
          transitionProperty: 'common',
          transitionDuration: 'normal',
          userSelect: 'none',
        }}
      >
        {label}
        <AnimatePresence>
          {activeLabel && (
            <motion.div
              key="statusText"
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
            >
              <Text
                sx={{ color: 'accent.500', _dark: { color: 'accent.300' } }}
              >
                {activeLabel}
              </Text>
            </motion.div>
          )}
        </AnimatePresence>
        <Spacer />
        <ChevronUpIcon
          sx={{
            w: '1rem',
            h: '1rem',
            transform: isOpen ? 'rotate(0deg)' : 'rotate(180deg)',
            transitionProperty: 'common',
            transitionDuration: 'normal',
          }}
        />
      </Flex>
      <Collapse in={isOpen} animateOpacity style={{ overflow: 'unset' }}>
        <Box
          sx={{
            p: 4,
            pb: 4,
            borderBottomRadius: 'base',
            bg: 'base.150',
            _dark: {
              bg: 'base.800',
            },
          }}
        >
          {children}
        </Box>
      </Collapse>
    </Box>
  );
};

export default memo(IAICollapse);
