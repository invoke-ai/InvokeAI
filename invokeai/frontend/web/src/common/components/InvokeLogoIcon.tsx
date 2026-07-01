import type { IconProps } from '@invoke-ai/ui-library';
import { Icon } from '@invoke-ai/ui-library';
import { memo } from 'react';

export const InvokeLogoIcon = memo((props: IconProps) => {
  return (
    <Icon boxSize={8} opacity={1} stroke="base.500" viewBox="0 0 66 66" fill="none" {...props}>
      <path d="M43.9137 16H63.1211V3H3.12109V16H22.3285L43.9137 50H63.1211V63H3.12109V50H22.3285" strokeWidth="5" />
    </Icon>
  );
});

InvokeLogoIcon.displayName = 'InvokeLogoIcon';
