import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { IconButton } from '@invoke-ai/ui-library';
import type { MouseEvent, ReactElement } from 'react';
import { memo, useMemo } from 'react';

type Props = {
  onClick: (event: MouseEvent<HTMLButtonElement>) => void;
  tooltip: string;
  icon?: ReactElement;
  styleOverrides?: SystemStyleObject;
};

const IAIDndImageIcon = (props: Props) => {
  const { onClick, tooltip, icon, styleOverrides } = props;

  const sx = useMemo(
    () => ({
      position: 'absolute',
      top: 1,
      insetInlineEnd: 1,
      p: 0,
      minW: 0,
      svg: {
        transitionProperty: 'common',
        transitionDuration: 'normal',
        fill: 'base.100',
        _hover: { fill: 'base.50' },
        filter: 'drop-shadow(0px 0px 0.1rem var(--invoke-colors-base-800))',
      },
      ...styleOverrides,
    }),
    [styleOverrides]
  );

  return (
    <IconButton
      onClick={onClick}
      aria-label={tooltip}
      tooltip={tooltip}
      icon={icon}
      size="sm"
      variant="link"
      sx={sx}
      data-testid={tooltip}
    />
  );
};

export default memo(IAIDndImageIcon);
