import type { FeatureHintsAdapter } from '@platform/ui/hints';
import type { ReactNode } from 'react';

import { FeatureHintsProvider } from '@platform/ui/hints';
import { toaster } from '@platform/ui/toaster';
import { patchWorkbenchPreferences, useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

/** App supplies preferences because Platform cannot import Workbench. */
export const FeatureHintsAdapterProvider = ({ children }: { children: ReactNode }) => {
  const { t } = useTranslation();
  const enabled = useWorkbenchPreferenceSelector((preferences) => preferences.enableInformationalPopovers);
  const onDisable = useCallback(() => {
    void patchWorkbenchPreferences({ enableInformationalPopovers: false });
    toaster.create({
      description: t('settings.informationalPopoversDisabledDesc'),
      title: t('settings.informationalPopoversDisabled'),
      type: 'info',
    });
  }, [t]);
  const adapter = useMemo<FeatureHintsAdapter>(() => ({ enabled, onDisable }), [enabled, onDisable]);

  return <FeatureHintsProvider adapter={adapter}>{children}</FeatureHintsProvider>;
};
