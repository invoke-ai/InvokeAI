import type { WidgetViewProps } from '@workbench/types';

import { HStack, Stack, Text } from '@chakra-ui/react';
import { Button, Panel } from '@workbench/components/ui';
import { StatusWidgetChip } from '@workbench/widget-frame';
import { useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { BugIcon, ClipboardListIcon } from 'lucide-react';
import { useCallback } from 'react';

export const DiagnosticsWidgetView = ({ presentation, region }: WidgetViewProps) => {
  const errorCount = useWorkbenchSelector((snapshot) => snapshot.state.errorLog.length);
  const label = errorCount === 0 ? 'Clean' : `${errorCount} issue${errorCount === 1 ? '' : 's'}`;

  if (region === 'bottom' && presentation !== 'expanded') {
    return (
      <StatusWidgetChip icon={errorCount > 0 ? BugIcon : ClipboardListIcon}>Diagnostics: {label}</StatusWidgetChip>
    );
  }

  return <DiagnosticsPanel />;
};

const DiagnosticsPanel = () => {
  const errorLog = useWorkbenchSelector((snapshot) => snapshot.state.errorLog);

  return (
    <Stack gap="3" p="2">
      {errorLog.length === 0 ? (
        <Text color="fg.subtle" fontSize="2xs">
          Shell errors and debugging details will appear here without covering the workbench.
        </Text>
      ) : (
        <Stack gap="2">
          {errorLog.map((message, index) => (
            <DiagnosticsErrorRow key={`${message}-${index}`} message={message} />
          ))}
        </Stack>
      )}
    </Stack>
  );
};

const DiagnosticsErrorRow = ({ message }: { message: string }) => {
  const copyMessage = useCallback(() => void navigator.clipboard?.writeText(message), [message]);

  return (
    <Panel gap="2" p="2">
      <Text color="fg.muted" fontFamily="mono" fontSize="2xs" whiteSpace="pre-wrap">
        {message}
      </Text>
      <HStack justify="end">
        <Button size="2xs" variant="outline" onClick={copyMessage}>
          Copy
        </Button>
      </HStack>
    </Panel>
  );
};
