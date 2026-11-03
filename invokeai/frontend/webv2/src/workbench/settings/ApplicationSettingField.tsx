import type { SettingFieldProps, SettingLayer } from '@platform/ui/settings/contracts';

import { Text } from '@chakra-ui/react';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { RetryBoundary } from '@platform/ui/RetryBoundary';
import { SettingControl } from '@platform/ui/settings/SettingControl';
import { createDeferredResource } from '@workbench/deferredResource';
import { Suspense, use, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { WorkbenchPreferences } from './contracts';

import { DEFAULT_PREFERENCES, patchWorkbenchPreferences, useWorkbenchPreferenceSelector } from './store';

const customEditorResource = createDeferredResource(() => import('./CustomSettingsEditors'));

export const prepareApplicationSettings = () => customEditorResource.load();

const CustomSettingField = (props: SettingFieldProps) => {
  const { default: Editor } = use(customEditorResource.load());
  return <Editor {...props} />;
};

export const ApplicationSettingField = (props: SettingFieldProps) => {
  const { t } = useTranslation();
  const fallback = useMemo(
    () => (
      <Text role="status" fontSize="md" color="fg.muted">
        {t('common.loading')}
      </Text>
    ),
    [t]
  );
  if (props.field.kind === 'custom') {
    return (
      <RetryBoundary
        retry={customEditorResource.retry}
        message={t('settingsDialog.loadFailed')}
        retryLabel={t('common.retry')}
      >
        <Suspense fallback={fallback}>
          <CustomSettingField {...props} />
        </Suspense>
      </RetryBoundary>
    );
  }
  return <PreferenceSettingField {...props} />;
};

const PreferenceSettingField = ({ field, surface }: SettingFieldProps) => {
  const [accountScope] = useState(captureAccountScope);
  const id = field.id as keyof WorkbenchPreferences;
  const value = useWorkbenchPreferenceSelector((preferences) => preferences[id]);
  const builtIn = DEFAULT_PREFERENCES[id];
  const onChange = useCallback(
    (next: boolean | number | string) => {
      if (isAccountScopeCurrent(accountScope)) {
        void patchWorkbenchPreferences({ [id]: next });
      }
    },
    [accountScope, id]
  );
  const layer = useMemo<SettingLayer | undefined>(
    () =>
      value === builtIn
        ? undefined
        : {
            onResetToBuiltIn: () => {
              if (isAccountScopeCurrent(accountScope)) {
                void patchWorkbenchPreferences({ [id]: builtIn });
              }
            },
            state: 'default',
          },
    [accountScope, builtIn, id, value]
  );
  if (typeof value !== 'boolean' && typeof value !== 'string' && typeof value !== 'number') {
    return null;
  }
  return <SettingControl layer={layer} field={field} surface={surface} value={value} onChange={onChange} />;
};
