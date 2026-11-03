import { useMountEffect } from '@platform/react/useMountEffect';

import { settleStartupLaunch } from './startupLaunch';

/**
 * Rendered after the editor shell, inside the provider that withholds both until hydration: its mount means the editor
 * rendered, so the account's launch mark is stale and the next launch may reopen the work again.
 */
export const StartupLaunchSettled = () => {
  useMountEffect(settleStartupLaunch);
  return null;
};
