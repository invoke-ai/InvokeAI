import type { FeatureHintsAdapter } from '@platform/ui/hints';
import type { ReactNode } from 'react';

import { FeatureHintsProvider } from '@platform/ui/hints';
import { toaster } from '@platform/ui/toaster';
import { patchWorkbenchPreferences, useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * Production binding of the feature-hints port. Platform may not read Workbench
 * preferences, so App supplies both the enabled flag and the write that the
 * card's "don't show me these" action needs.
 */
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
