import type { IconButtonProps, SystemStyleObject } from '@invoke-ai/ui-library';
import { IconButton } from '@invoke-ai/ui-library';
import type { MouseEvent } from 'react';
import { memo } from 'react';

export const imageButtonSx: SystemStyleObject = {
  minW: 0,
  svg: {
    transitionProperty: 'common',
    transitionDuration: 'normal',
    fill: 'base.100',
    _hover: { fill: 'base.50' },
    filter: `drop-shadow(0px 0px 0.1rem var(--invoke-colors-base-900))
      drop-shadow(0px 0px 0.3rem var(--invoke-colors-base-900))
      drop-shadow(0px 0px 0.3rem var(--invoke-colors-base-900))`,
  },
};

type Props = Omit<IconButtonProps, 'aria-label' | 'onClick' | 'tooltip'> & {
  onClick: (event: MouseEvent<HTMLButtonElement>) => void;
  tooltip: string;
};

export const DndImageIcon = memo((props: Props) => {
  const { onClick, tooltip, icon, ...rest } = props;

  return (
    <IconButton
      onClick={onClick}
      tooltip={tooltip}
      aria-label={tooltip}
      icon={icon}
      variant="link"
      sx={imageButtonSx}
      data-testid={tooltip}
      {...rest}
    />
  );
});

DndImageIcon.displayName = 'DndImageIcon';
