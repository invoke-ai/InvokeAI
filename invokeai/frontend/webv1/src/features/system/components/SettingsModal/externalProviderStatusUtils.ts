import type { ExternalProviderStatus } from 'services/api/types';

type ExternalProviderStatusBadgeInfo = {
  labelKey: 'settings.externalProviderConfigured' | 'settings.externalProviderNotConfigured';
  tooltipKey: 'settings.externalProviderNotConfiguredHint' | null;
  tooltipMessage: string | null;
  colorScheme: 'green' | 'warning';
};

export const getExternalProviderStatusBadgeInfo = (status: ExternalProviderStatus): ExternalProviderStatusBadgeInfo => {
  if (status.configured) {
    return {
      labelKey: 'settings.externalProviderConfigured',
      tooltipKey: null,
      tooltipMessage: status.message ?? null,
      colorScheme: 'green',
    };
  }

  return {
    labelKey: 'settings.externalProviderNotConfigured',
    tooltipKey: status.message ? null : 'settings.externalProviderNotConfiguredHint',
    tooltipMessage: status.message ?? null,
    colorScheme: 'warning',
  };
};
