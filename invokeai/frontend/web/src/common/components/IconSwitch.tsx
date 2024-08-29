import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Flex, IconButton, Tooltip, useToken } from '@invoke-ai/ui-library';
import type { ReactElement, ReactNode } from 'react';
import { memo, useCallback, useMemo } from 'react';

type IconSwitchProps = {
  isChecked: boolean;
  onChange: (checked: boolean) => void;
  iconChecked: ReactElement;
  tooltipChecked?: ReactNode;
  iconUnchecked: ReactElement;
  tooltipUnchecked?: ReactNode;
  ariaLabel: string;
};

const getSx = (padding: string | number): SystemStyleObject => ({
  transition: 'left 0.1s ease-in-out, transform 0.1s ease-in-out',
  '&[data-checked="true"]': {
    left: `calc(100% - ${padding})`,
    transform: 'translateX(-100%)',
  },
  '&[data-checked="false"]': {
    left: padding,
    transform: 'translateX(0)',
  },
});

export const IconSwitch = memo(
  ({
    isChecked,
    onChange,
    iconChecked,
    tooltipChecked,
    iconUnchecked,
    tooltipUnchecked,
    ariaLabel,
  }: IconSwitchProps) => {
    const onUncheck = useCallback(() => {
      onChange(false);
    }, [onChange]);
    const onCheck = useCallback(() => {
      onChange(true);
    }, [onChange]);

    const gap = useToken('space', 1.5);
    const sx = useMemo(() => getSx(gap), [gap]);

    return (
      <Flex
        position="relative"
        bg="base.800"
        borderRadius="base"
        alignItems="center"
        justifyContent="center"
        h="full"
        p={gap}
        gap={gap}
      >
        <Box
          position="absolute"
          borderRadius="base"
          bg="invokeBlue.400"
          w={12}
          top={gap}
          bottom={gap}
          data-checked={isChecked}
          sx={sx}
        />
        <Tooltip hasArrow label={tooltipUnchecked}>
          <IconButton
            size="sm"
            fontSize={16}
            icon={iconUnchecked}
            onClick={onUncheck}
            variant={!isChecked ? 'solid' : 'ghost'}
            colorScheme={!isChecked ? 'invokeBlue' : 'base'}
            aria-label={ariaLabel}
            data-checked={!isChecked}
            w={12}
            alignSelf="stretch"
            h="auto"
          />
        </Tooltip>
        <Tooltip hasArrow label={tooltipChecked}>
          <IconButton
            size="sm"
            fontSize={16}
            icon={iconChecked}
            onClick={onCheck}
            variant={isChecked ? 'solid' : 'ghost'}
            colorScheme={isChecked ? 'invokeBlue' : 'base'}
            aria-label={ariaLabel}
            data-checked={isChecked}
            w={12}
            alignSelf="stretch"
            h="auto"
          />
        </Tooltip>
      </Flex>
    );
  }
);

IconSwitch.displayName = 'IconSwitch';
