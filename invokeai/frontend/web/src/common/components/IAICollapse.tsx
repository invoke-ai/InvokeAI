import { ChevronUpIcon } from '@chakra-ui/icons';
import {
  Box,
  Collapse,
  Flex,
  Spacer,
  Switch,
  useColorMode,
} from '@chakra-ui/react';
import { PropsWithChildren, memo } from 'react';
import { mode } from 'theme/util/mode';

export type IAIToggleCollapseProps = PropsWithChildren & {
  label: string;
  isOpen: boolean;
  onToggle: () => void;
  withSwitch?: boolean;
};

const IAICollapse = (props: IAIToggleCollapseProps) => {
  const { label, isOpen, onToggle, children, withSwitch = false } = props;
  const { colorMode } = useColorMode();
  return (
    <Box>
      <Flex
        onClick={onToggle}
        sx={{
          alignItems: 'center',
          p: 2,
          px: 4,
          borderTopRadius: 'base',
          borderBottomRadius: isOpen ? 0 : 'base',
          bg: isOpen
            ? mode('base.200', 'base.750')(colorMode)
            : mode('base.150', 'base.800')(colorMode),
          color: mode('base.900', 'base.100')(colorMode),
          _hover: {
            bg: isOpen
              ? mode('base.250', 'base.700')(colorMode)
              : mode('base.200', 'base.750')(colorMode),
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
        <Spacer />
        {withSwitch && <Switch isChecked={isOpen} pointerEvents="none" />}
        {!withSwitch && (
          <ChevronUpIcon
            sx={{
              w: '1rem',
              h: '1rem',
              transform: isOpen ? 'rotate(0deg)' : 'rotate(180deg)',
              transitionProperty: 'common',
              transitionDuration: 'normal',
            }}
          />
        )}
      </Flex>
      <Collapse in={isOpen} animateOpacity style={{ overflow: 'unset' }}>
        <Box
          sx={{
            p: 4,
            borderBottomRadius: 'base',
            bg: mode('base.100', 'base.800')(colorMode),
          }}
        >
          {children}
        </Box>
      </Collapse>
    </Box>
  );
};

export default memo(IAICollapse);
