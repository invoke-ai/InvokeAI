import { Stack, Text } from '@chakra-ui/react';
import { CloudAlertIcon, CloudCheckIcon } from 'lucide-react';

import { StatusWidgetChip } from '../../components/WidgetFrames';
import type { WidgetViewProps } from '../../types';
import { useWorkbench } from '../../WorkbenchContext';

export const AutosaveStatusWidgetView = ({ presentation }: WidgetViewProps) => {
  const { state } = useWorkbench();
  const icon = state.autosave.status === 'error' ? CloudAlertIcon : CloudCheckIcon;
  const label = state.autosave.status === 'saved' ? 'Saved' : state.autosave.status;

  if (presentation === 'tooltip') {
    return (
      <Stack gap="2">
        <Text fontSize="xs" fontWeight="700">
          Autosave
        </Text>
        <Text color="fg.subtle" fontSize="2xs">
          Status: {label}
        </Text>
        {state.autosave.lastSavedAt ? (
          <Text color="fg.subtle" fontSize="2xs">
            Last saved: {state.autosave.lastSavedAt}
          </Text>
        ) : null}
        {state.autosave.error ? (
          <Text color="red.300" fontSize="2xs">
            {state.autosave.error}
          </Text>
        ) : null}
      </Stack>
    );
  }

  return <StatusWidgetChip icon={icon}>Autosave: {label}</StatusWidgetChip>;
};
