import { Stack, Text } from '@chakra-ui/react';
import { CloudAlertIcon, CloudCheckIcon } from 'lucide-react';

import { StatusWidgetChip } from '../../components/WidgetFrames';
import type { WidgetViewProps } from '../../types';
import { useWorkbenchSelector } from '../../WorkbenchContext';

export const AutosaveStatusWidgetView = ({ presentation }: WidgetViewProps) => {
  const autosave = useWorkbenchSelector((snapshot) => snapshot.state.autosave);
  const icon = autosave.status === 'error' ? CloudAlertIcon : CloudCheckIcon;
  const label = autosave.status === 'saved' ? 'Saved' : autosave.status;

  if (presentation === 'tooltip') {
    return (
      <Stack gap="2">
        <Text fontSize="xs" fontWeight="700">
          Autosave
        </Text>
        <Text color="fg.subtle" fontSize="2xs">
          Status: {label}
        </Text>
        {autosave.lastSavedAt ? (
          <Text color="fg.subtle" fontSize="2xs">
            Last saved: {autosave.lastSavedAt}
          </Text>
        ) : null}
        {autosave.error ? (
          <Text color="fg.error" fontSize="2xs">
            {autosave.error}
          </Text>
        ) : null}
      </Stack>
    );
  }

  return <StatusWidgetChip icon={icon}>Autosave: {label}</StatusWidgetChip>;
};
