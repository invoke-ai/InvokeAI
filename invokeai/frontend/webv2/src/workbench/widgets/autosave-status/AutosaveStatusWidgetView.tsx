import type { AutosaveState } from '@workbench/projectContracts';
import type { ProjectSyncAttention } from '@workbench/projects/syncStore';
import type { WidgetViewProps } from '@workbench/widgetContracts';
import type { TFunction } from 'i18next';

import { List, Stack, Text } from '@chakra-ui/react';
import { useProjectSyncAttention } from '@workbench/projects/syncStore';
import { StatusWidgetChip } from '@workbench/widget-frame';
import { useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { CloudAlertIcon, CloudCheckIcon, CloudUploadIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

const getStatusLabel = (status: AutosaveState['status'], t: TFunction): string => {
  switch (status) {
    case 'attention':
      return t('widgets.autosaveStatus.attention');
    case 'error':
      return t('widgets.autosaveStatus.error');
    case 'idle':
      return t('widgets.autosaveStatus.idle');
    case 'pending':
      return t('widgets.autosaveStatus.pending');
    case 'saved':
      return t('common.saved');
    case 'saving':
      return t('widgets.autosaveStatus.saving');
  }
};

const getAttentionReasonLabel = (reason: ProjectSyncAttention, t: TFunction): string => {
  switch (reason) {
    case 'deleted':
      return t('widgets.autosaveStatus.reasonDeleted');
    case 'revision':
      return t('widgets.autosaveStatus.reasonRevision');
    case 'schema':
      return t('widgets.autosaveStatus.reasonSchema');
  }
};

export const AutosaveStatusWidgetView = ({ presentation }: WidgetViewProps) => {
  const { t } = useTranslation();
  const autosave = useWorkbenchSelector((snapshot) => snapshot.autosave);
  const label = getStatusLabel(autosave.status, t);

  if (presentation === 'tooltip') {
    return <AutosaveStatusDetails autosave={autosave} label={label} />;
  }

  const icon =
    autosave.status === 'error' || autosave.status === 'attention'
      ? CloudAlertIcon
      : autosave.status === 'pending' || autosave.status === 'saving'
        ? CloudUploadIcon
        : CloudCheckIcon;
  const tone = autosave.status === 'error' ? 'error' : autosave.status === 'attention' ? 'warning' : undefined;

  return (
    <StatusWidgetChip icon={icon} tone={tone}>
      {t('widgets.autosaveStatus.chipLabel', { status: label })}
    </StatusWidgetChip>
  );
};

const AutosaveStatusDetails = ({ autosave, label }: { autosave: AutosaveState; label: string }) => {
  const { t } = useTranslation();
  const attention = useProjectSyncAttention();
  // Names by id, compared shallowly, so edits that leave every name alone do not rerender the tooltip.
  const projectNames = useWorkbenchSelector((snapshot) =>
    Object.fromEntries(snapshot.projects.map((project) => [project.id, project.name]))
  );
  const attentionProjects = Object.entries(projectNames).flatMap(([projectId, name]) => {
    const reason = attention[projectId];
    return reason ? [{ name, projectId, reason }] : [];
  });

  return (
    <Stack gap="2">
      <Text fontSize="md" fontWeight="700">
        {t('widgets.autosaveStatus.label')}
      </Text>
      <Text color="fg.subtle" fontSize="xs">
        {t('widgets.autosaveStatus.status', { status: label })}
      </Text>
      {autosave.lastSavedAt ? (
        <Text color="fg.subtle" fontSize="xs">
          {t('widgets.autosaveStatus.lastSaved', { time: new Date(autosave.lastSavedAt).toLocaleString() })}
        </Text>
      ) : null}
      {attentionProjects.length > 0 ? (
        <Stack gap="1.5">
          <Text fontSize="xs">{t('widgets.autosaveStatus.attentionDescription')}</Text>
          <List.Root fontSize="xs" gap="0.5" ps="4">
            {attentionProjects.map(({ name, projectId, reason }) => (
              <List.Item key={projectId}>
                {t('widgets.autosaveStatus.attentionItem', { name, reason: getAttentionReasonLabel(reason, t) })}
              </List.Item>
            ))}
          </List.Root>
        </Stack>
      ) : null}
      {autosave.status === 'pending' && autosave.error ? (
        <Text color="fg.subtle" fontSize="xs">
          {t('widgets.autosaveStatus.retrying')}
        </Text>
      ) : null}
      {autosave.status === 'error' && autosave.error ? (
        <Text color="fg.error" fontSize="xs">
          {autosave.error}
        </Text>
      ) : null}
    </Stack>
  );
};
