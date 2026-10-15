import type { WidgetViewProps } from '@workbench/widgetContracts';

import { Stack, Text } from '@chakra-ui/react';
import { StatusWidgetChip } from '@workbench/widget-frame';
import { useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { CloudAlertIcon, CloudCheckIcon, CloudUploadIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

export const AutosaveStatusWidgetView = ({ presentation }: WidgetViewProps) => {
  const { t } = useTranslation();
  const autosave = useWorkbenchSelector((snapshot) => snapshot.autosave);
  const icon =
    autosave.status === 'error'
      ? CloudAlertIcon
      : autosave.status === 'pending' || autosave.status === 'saving'
        ? CloudUploadIcon
        : CloudCheckIcon;
  const label =
    autosave.status === 'saved'
      ? t('common.saved')
      : autosave.status === 'pending'
        ? t('widgets.autosaveStatus.pending')
        : autosave.status;

  if (presentation === 'tooltip') {
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
            {t('widgets.autosaveStatus.lastSaved', { time: autosave.lastSavedAt })}
          </Text>
        ) : null}
        {autosave.error ? (
          <Text color="fg.error" fontSize="xs">
            {autosave.error}
          </Text>
        ) : null}
      </Stack>
    );
  }

  return <StatusWidgetChip icon={icon}>{t('widgets.autosaveStatus.chipLabel', { status: label })}</StatusWidgetChip>;
};
