import { useMountEffect } from '@platform/react/useMountEffect';

import { settleStartupLaunch } from './startupLaunch';

/** Mounts only once the editor shell rendered, so the account's launch mark is stale. */
export const StartupLaunchSettled = () => {
  useMountEffect(settleStartupLaunch);
  return null;
};
