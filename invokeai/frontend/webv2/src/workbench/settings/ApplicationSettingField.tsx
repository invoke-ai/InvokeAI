import type { SettingFieldProps } from '@platform/ui/settings/contracts';

import { Text } from '@chakra-ui/react';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { RetryBoundary } from '@platform/ui/RetryBoundary';
import { SettingControl } from '@platform/ui/settings/SettingControl';
import { createDeferredResource } from '@workbench/deferredResource';
import {
  useOptionalWorkbenchCommands,
  useOptionalWorkbenchQueries,
  useOptionalWorkbenchSelector,
} from '@workbench/WorkbenchContext';
import { Suspense, use, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { ProjectSettings, WorkbenchPreferences } from './contracts';

import {
  DEFAULT_PREFERENCES,
  DEFAULT_PROJECT_SETTINGS,
  patchWorkbenchPreferences,
  useWorkbenchPreferenceSelector,
} from './store';

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
      <Text role="status" fontSize="xs" color="fg.muted">
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
  return props.field.scope === 'project' ? <ProjectSettingField {...props} /> : <PreferenceSettingField {...props} />;
};

const PreferenceSettingField = ({ field, surface }: SettingFieldProps) => {
  const [accountScope] = useState(captureAccountScope);
  const value = useWorkbenchPreferenceSelector((preferences) => preferences[field.id as keyof WorkbenchPreferences]);
  const onChange = useCallback(
    (next: boolean | number | string) => {
      if (isAccountScopeCurrent(accountScope)) {
        void patchWorkbenchPreferences({ [field.id]: next });
      }
    },
    [accountScope, field.id]
  );
  if (typeof value !== 'boolean' && typeof value !== 'string' && typeof value !== 'number') {
    return null;
  }
  return (
    <SettingControl
      isModified={value !== DEFAULT_PREFERENCES[field.id as keyof WorkbenchPreferences]}
      field={field}
      surface={surface}
      value={value}
      onChange={onChange}
    />
  );
};

const ProjectSettingField = ({ field, surface, target }: SettingFieldProps) => {
  const [accountScope] = useState(captureAccountScope);
  const commands = useOptionalWorkbenchCommands();
  const queries = useOptionalWorkbenchQueries();
  const projectId = useOptionalWorkbenchSelector((snapshot) => snapshot.activeProject.id, null);
  const value = useOptionalWorkbenchSelector(
    (snapshot) => snapshot.activeProject.settings[field.id as keyof ProjectSettings],
    false
  );
  const onChange = useCallback(
    (next: boolean | number | string) => {
      if (
        commands &&
        projectId &&
        target?.projectId === projectId &&
        queries?.isActiveProject(projectId) &&
        isAccountScopeCurrent(accountScope) &&
        typeof next === 'boolean'
      ) {
        commands.account.updateProjectPreferences({ [field.id]: next });
      }
    },
    [accountScope, commands, field.id, projectId, queries, target?.projectId]
  );
  return (
    <SettingControl
      isModified={value !== DEFAULT_PROJECT_SETTINGS[field.id as keyof ProjectSettings]}
      field={field}
      surface={surface}
      value={value}
      disabled={!commands || target?.projectId !== projectId}
      onChange={onChange}
    />
  );
};
