import { HStack, Stack, Text } from '@chakra-ui/react';
import { BugIcon, ClipboardListIcon } from 'lucide-react';

import { StatusWidgetChip } from '../../components/WidgetFrames';
import type { WidgetViewProps } from '../../types';
import { useWorkbench } from '../../WorkbenchContext';
import { Button } from '../../components/ui/Button';

export const DiagnosticsWidgetView = ({ presentation, region }: WidgetViewProps) => {
  const { state } = useWorkbench();
  const errorCount = state.errorLog.length;
  const label = errorCount === 0 ? 'Clean' : `${errorCount} issue${errorCount === 1 ? '' : 's'}`;

  if (region === 'bottom' && presentation !== 'expanded') {
    return (
      <StatusWidgetChip icon={errorCount > 0 ? BugIcon : ClipboardListIcon}>Diagnostics: {label}</StatusWidgetChip>
    );
  }

  return <DiagnosticsPanel />;
};

const DiagnosticsPanel = () => {
  const { state } = useWorkbench();

  return (
    <Stack gap="3">
      {state.errorLog.length === 0 ? (
        <Text color="fg.subtle" fontSize="2xs">
          Shell errors and debugging details will appear here without covering the workbench.
        </Text>
      ) : (
        <Stack gap="2">
          {state.errorLog.map((message, index) => (
            <Stack
              key={`${message}-${index}`}
              bg="bg.surface"
              borderWidth="1px"
              borderColor="border.subtle"
              gap="2"
              p="2"
              rounded="md"
            >
              <Text color="fg.muted" fontFamily="mono" fontSize="2xs" whiteSpace="pre-wrap">
                {message}
              </Text>
              <HStack justify="end">
                <Button size="2xs" variant="outline" onClick={() => void navigator.clipboard?.writeText(message)}>
                  Copy
                </Button>
              </HStack>
            </Stack>
          ))}
        </Stack>
      )}
    </Stack>
  );
};
