import { Button } from '@platform/ui/Button';
import { useRouter } from '@tanstack/react-router';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { stayOnLaunchpad, useStartupLaunchPhase } from './startupLaunch';

/** The splash's way out of a launch that is reopening the last work: go to the Launchpad instead. */
export const StayOnLaunchpadButton = () => {
  const { t } = useTranslation();
  const router = useRouter();
  const phase = useStartupLaunchPhase();
  const stay = useCallback(() => {
    stayOnLaunchpad();
    void router.navigate({ replace: true, to: '/' });
  }, [router]);

  if (phase === 'idle') {
    return null;
  }

  return (
    <Button alignSelf="start" size="sm" variant="outline" onClick={stay}>
      {t('splash.stayOnLaunchpad')}
    </Button>
  );
};
