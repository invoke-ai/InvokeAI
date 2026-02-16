import type { TooltipProps } from '@invoke-ai/ui-library';
import { Tooltip } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectSystemShouldEnableInformationalPopovers } from 'features/system/store/systemSlice';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

/**
 * A Tooltip component that respects the user's "Enable Informational Popovers" setting.
 * When the setting is disabled, this component returns its children without a tooltip wrapper.
 */
export const IAITooltip = memo(({ children, ...rest }: PropsWithChildren<TooltipProps>) => {
  const shouldEnableInformationalPopovers = useAppSelector(selectSystemShouldEnableInformationalPopovers);

  if (!shouldEnableInformationalPopovers) {
    return children;
  }

  return <Tooltip {...rest}>{children}</Tooltip>;
});

IAITooltip.displayName = 'IAITooltip';
