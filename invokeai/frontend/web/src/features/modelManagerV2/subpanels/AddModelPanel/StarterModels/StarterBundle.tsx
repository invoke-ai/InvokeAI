import type { ButtonProps } from '@invoke-ai/ui-library';
import { Button } from '@invoke-ai/ui-library';
import { useStarterBundleInstall } from 'features/modelManagerV2/hooks/useStarterBundleInstall';
import { useStarterBundleInstallStatus } from 'features/modelManagerV2/hooks/useStarterBundleInstallStatus';
import { useCallback } from 'react';
import type { S } from 'services/api/types';

export const StarterBundleButton = ({ bundle, ...rest }: { bundle: S['StarterModelBundle'] } & ButtonProps) => {
  const { installBundle } = useStarterBundleInstall();
  const { install } = useStarterBundleInstallStatus(bundle);

  const handleClickBundle = useCallback(() => {
    installBundle(bundle);
  }, [installBundle, bundle]);

  return (
    <Button onClick={handleClickBundle} isDisabled={install.length === 0} {...rest}>
      {bundle.name}
    </Button>
  );
};
