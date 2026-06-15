import type { WidgetViewProps } from '@workbench/types';

import { HStack, Stack, Text } from '@chakra-ui/react';
import { Button } from '@workbench/components/ui/Button';
import { Panel } from '@workbench/components/ui/Panel';
import { StatusWidgetChip } from '@workbench/components/WidgetFrames';
import { useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { BugIcon, ClipboardListIcon } from 'lucide-react';

export const DiagnosticsWidgetView = ({ presentation, region }: WidgetViewProps) => {
  const errorLog = useWorkbenchSelector((snapshot) => snapshot.state.errorLog);
  const errorCount = errorLog.length;
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
    <Stack gap="3">
      {errorLog.length === 0 ? (
        <Text color="fg.subtle" fontSize="2xs">
          Shell errors and debugging details will appear here without covering the workbench.
        </Text>
      ) : (
        <Stack gap="2">
          {errorLog.map((message, index) => (
            <Panel key={`${message}-${index}`} gap="2" p="2">
              <Text color="fg.muted" fontFamily="mono" fontSize="2xs" whiteSpace="pre-wrap">
                {message}
              </Text>
              <HStack justify="end">
                <Button size="2xs" variant="outline" onClick={() => void navigator.clipboard?.writeText(message)}>
                  Copy
                </Button>
              </HStack>
            </Panel>
          ))}
        </Stack>
      )}
    </Stack>
  );
};
