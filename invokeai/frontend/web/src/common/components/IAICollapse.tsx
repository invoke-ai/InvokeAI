import { ChevronUpIcon } from '@chakra-ui/icons';
import { Box, Collapse, Flex, Spacer, Switch } from '@chakra-ui/react';
import { PropsWithChildren, memo } from 'react';

export type IAIToggleCollapseProps = PropsWithChildren & {
  label: string;
  isOpen: boolean;
  onToggle: () => void;
  withSwitch?: boolean;
};

const IAICollapse = (props: IAIToggleCollapseProps) => {
  const { label, isOpen, onToggle, children, withSwitch = false } = props;
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
          bg: isOpen ? 'base.750' : 'base.800',
          color: 'base.100',
          _hover: {
            bg: isOpen ? 'base.700' : 'base.750',
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
      <Collapse in={isOpen} animateOpacity>
        <Box sx={{ p: 4, borderBottomRadius: 'base', bg: 'base.800' }}>
          {children}
        </Box>
      </Collapse>
    </Box>
  );
};

export default memo(IAICollapse);
