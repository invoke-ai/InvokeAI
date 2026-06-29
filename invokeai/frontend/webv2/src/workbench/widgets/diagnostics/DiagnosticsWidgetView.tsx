import type { DiagnosticEntry } from '@workbench/diagnostics/logger';
import type { WidgetViewProps } from '@workbench/types';

import { Badge, Box, HStack, Stack, Text } from '@chakra-ui/react';
import { Button, Panel } from '@workbench/components/ui';
import { clearProjectDiagnostics, useProjectDiagnostics } from '@workbench/diagnostics/logger';
import { StatusWidgetChip } from '@workbench/widget-frame';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { BugIcon, ClipboardListIcon } from 'lucide-react';
import { memo, useCallback, useState } from 'react';

export const DiagnosticsWidgetView = ({ presentation, region }: WidgetViewProps) => {
  const projectId = useActiveProjectSelector((project) => project.id);
  const entries = useProjectDiagnostics(projectId);
  const label = entries.length === 0 ? 'Clean' : `${entries.length} event${entries.length === 1 ? '' : 's'}`;

  if (region === 'bottom' && presentation !== 'expanded') {
    return (
      <StatusWidgetChip icon={entries.length > 0 ? BugIcon : ClipboardListIcon}>Diagnostics: {label}</StatusWidgetChip>
    );
  }

  return <DiagnosticsPanel entries={entries} projectId={projectId} />;
};

const DiagnosticsPanel = ({ entries, projectId }: { entries: DiagnosticEntry[]; projectId: string }) => {
  const copyAll = useCallback(() => void navigator.clipboard?.writeText(JSON.stringify(entries, null, 2)), [entries]);
  const clearEntries = useCallback(() => clearProjectDiagnostics(projectId), [projectId]);

  return (
    <Stack gap="3" p="2">
      <HStack justify="space-between">
        <Text color="fg.subtle" fontSize="2xs">
          Project-scoped diagnostic events from workbench and widget loggers.
        </Text>
        <HStack gap="2">
          <Button disabled={entries.length === 0} size="2xs" variant="outline" onClick={copyAll}>
            Copy JSON
          </Button>
          <Button disabled={entries.length === 0} size="2xs" variant="outline" onClick={clearEntries}>
            Clear
          </Button>
        </HStack>
      </HStack>
      {entries.length === 0 ? (
        <Text color="fg.subtle" fontSize="2xs">
          Logger events and performance timings will appear here without covering the workbench.
        </Text>
      ) : (
        <Stack gap="2">
          {entries.map((entry) => (
            <DiagnosticsEntryRow key={entry.id} entry={entry} />
          ))}
        </Stack>
      )}
    </Stack>
  );
};

const DiagnosticsEntryRow = memo(({ entry }: { entry: DiagnosticEntry }) => {
  const [isRawVisible, setIsRawVisible] = useState(false);
  const copyEntry = useCallback(() => void navigator.clipboard?.writeText(JSON.stringify(entry, null, 2)), [entry]);
  const toggleRaw = useCallback(() => setIsRawVisible((current) => !current), []);

  return (
    <Panel gap="2" p="2">
      <Stack gap="2">
        <HStack align="start" gap="2" justify="space-between">
          <Stack gap="1" minW="0">
            <HStack gap="1.5" wrap="wrap">
              <Badge colorPalette={getLevelColorPalette(entry.level)} size="xs">
                {entry.level}
              </Badge>
              <Badge colorPalette="gray" size="xs">
                {entry.namespace}
              </Badge>
              {entry.durationMs !== undefined ? (
                <Badge colorPalette="purple" size="xs">
                  {entry.durationMs.toFixed(1)}ms
                </Badge>
              ) : null}
              <Text color="fg.muted" fontFamily="mono" fontSize="2xs">
                {formatSource(entry)}
              </Text>
            </HStack>
            <Text color="fg" fontSize="xs" fontWeight="600">
              {entry.message || '(no message)'}
            </Text>
            <Text color="fg.subtle" fontSize="2xs">
              {new Date(entry.createdAt).toLocaleTimeString()}
            </Text>
          </Stack>
          <Button size="2xs" variant="outline" onClick={copyEntry}>
            Copy
          </Button>
        </HStack>
        <Button alignSelf="start" size="2xs" variant="ghost" onClick={toggleRaw}>
          {isRawVisible ? 'Hide raw entry' : 'Raw entry'}
        </Button>
        {isRawVisible ? (
          <Box
            as="pre"
            bg="bg.inset"
            borderColor="border.subtle"
            borderWidth="1px"
            color="fg.muted"
            fontFamily="mono"
            fontSize="2xs"
            maxH="16rem"
            overflow="auto"
            p="2"
            rounded="md"
            whiteSpace="pre-wrap"
          >
            {JSON.stringify(entry, null, 2)}
          </Box>
        ) : null}
      </Stack>
    </Panel>
  );
});

DiagnosticsEntryRow.displayName = 'DiagnosticsEntryRow';

const formatSource = (entry: DiagnosticEntry): string => {
  if (entry.source.kind === 'widget') {
    return `${entry.source.typeId}:${entry.source.instanceId}:${entry.source.region}`;
  }

  return `workbench:${entry.source.area}`;
};

const getLevelColorPalette = (level: DiagnosticEntry['level']): string => {
  switch (level) {
    case 'fatal':
    case 'error':
      return 'red';
    case 'warn':
      return 'orange';
    case 'info':
      return 'blue';
    case 'debug':
      return 'purple';
    case 'trace':
      return 'gray';
  }
};
