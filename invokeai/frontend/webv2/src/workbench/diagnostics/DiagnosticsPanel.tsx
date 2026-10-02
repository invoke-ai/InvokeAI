import type {
  LogEntry,
  LoggingConfig,
  LogLevel,
  LogNamespace,
  LogScope,
  LogSnapshot,
} from '@platform/logging/contracts';

import { Badge, Box, createListCollection, HStack, Icon, Input, InputGroup, Stack, Text } from '@chakra-ui/react';
import { downloadText } from '@platform/browser/downloadBlob';
import { LOG_LEVEL_ORDER, LOG_NAMESPACES, matchesLogScope } from '@platform/logging/contracts';
import { createLogExport } from '@platform/logging/export';
import { clearLogs } from '@platform/logging/logger';
import { useLoggingConfig, useLogSnapshot } from '@platform/logging/react';
import { APP_VERSION } from '@platform/runtime/appMetadata';
import { Button, IconButton, JsonPreview, Panel, Scrollable, Select, Tooltip } from '@platform/ui';
import { toaster } from '@platform/ui/toaster';
import { openWorkbenchSettings } from '@workbench/settings/settingsDialogStore';
import { ChevronLeftIcon, ChevronRightIcon, SearchIcon } from 'lucide-react';
import { memo, useCallback, useMemo, useState, type ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';

export const DIAGNOSTICS_PAGE_SIZE = 100;

type ScopeKind = LogScope['kind'];
type SeverityFilter = 'all' | Exclude<LogLevel, 'trace'>;
type NamespaceFilter = 'all' | LogNamespace | 'performance';

interface DiagnosticsFilters {
  namespace: NamespaceFilter;
  page: number;
  query: string;
  scope: ScopeKind;
  severity: SeverityFilter;
}

const SEVERITY_FILTERS: readonly SeverityFilter[] = ['all', 'debug', 'info', 'warn', 'error'];
const NAMESPACE_FILTERS: readonly NamespaceFilter[] = ['all', ...LOG_NAMESPACES, 'performance'];
const SEARCH_START_ELEMENT = <Icon as={SearchIcon} boxSize="3.5" color="fg.muted" />;
const SELECT_POSITIONING = { sameWidth: false } as const;
const LEVEL_COLOR_PALETTE: Record<LogLevel, string> = {
  debug: 'purple',
  error: 'red',
  fatal: 'red',
  info: 'blue',
  trace: 'gray',
  warn: 'orange',
};

const resolveScope = (kind: ScopeKind, projectId: string | null): LogScope =>
  projectId === null || kind === 'all' ? { kind: 'all' } : { kind, projectId };

const matchesQuery = (entry: LogEntry, query: string): boolean => {
  const haystack = [
    entry.message,
    entry.name,
    entry.namespace,
    entry.source.area,
    entry.source.projectId,
    entry.source.operationId,
    entry.source.widget?.typeId,
    entry.source.widget?.instanceId,
    entry.error?.name,
    entry.error?.message,
  ]
    .filter((part): part is string => typeof part === 'string' && part.length > 0)
    .join('\n')
    .toLowerCase();

  return haystack.includes(query);
};

const filterEntries = (
  entries: readonly LogEntry[],
  scope: LogScope,
  filters: DiagnosticsFilters
): { matched: LogEntry[]; scopedCount: number } => {
  const query = filters.query.trim().toLowerCase();
  const minimum = filters.severity === 'all' ? 0 : LOG_LEVEL_ORDER[filters.severity];
  let scopedCount = 0;
  const matched: LogEntry[] = [];

  for (const entry of entries) {
    if (!matchesLogScope(entry, scope)) {
      continue;
    }

    scopedCount += 1;

    if (LOG_LEVEL_ORDER[entry.level] < minimum) {
      continue;
    }
    if (filters.namespace !== 'all' && entry.namespace !== filters.namespace) {
      continue;
    }
    if (query && !matchesQuery(entry, query)) {
      continue;
    }

    matched.push(entry);
  }

  return { matched, scopedCount };
};

const formatSource = (entry: LogEntry, showProject: boolean): string => {
  const parts = [`${entry.namespace}/${entry.source.area}`];

  if (entry.source.widget) {
    parts.push(`${entry.source.widget.typeId}:${entry.source.widget.instanceId}`);
  }
  if (showProject && entry.source.projectId) {
    parts.push(entry.source.projectId);
  }
  if (entry.source.operationId) {
    parts.push(entry.source.operationId);
  }

  return parts.join(' · ');
};

/**
 * Lists recorded events for one scope with display-only filters, paging and export. Callers supply the project
 * the viewer belongs to; Launchpad-style hosts pass null and see account-wide events only.
 */
export const DiagnosticsPanel = ({ projectId }: { projectId: string | null }) => {
  const { t } = useTranslation();
  const snapshot = useLogSnapshot();
  const config = useLoggingConfig();
  const [filters, setFilters] = useState<DiagnosticsFilters>({
    namespace: 'all',
    page: 1,
    query: '',
    scope: 'project-and-application',
    severity: 'all',
  });
  const scope = useMemo(() => resolveScope(filters.scope, projectId), [filters.scope, projectId]);
  const { matched, scopedCount } = useMemo(
    () => filterEntries(snapshot.entries, scope, filters),
    [filters, scope, snapshot.entries]
  );
  const pageCount = Math.max(1, Math.ceil(matched.length / DIAGNOSTICS_PAGE_SIZE));
  const page = Math.min(filters.page, pageCount);
  const pageEntries = useMemo(
    () => matched.slice((page - 1) * DIAGNOSTICS_PAGE_SIZE, page * DIAGNOSTICS_PAGE_SIZE),
    [matched, page]
  );
  const hasFilters = filters.query.trim() !== '' || filters.severity !== 'all' || filters.namespace !== 'all';

  const levelLabel = useCallback((level: LogLevel) => t(`settings.catalog.options.${level}`), [t]);
  const scopeLabel = useCallback(
    (kind: ScopeKind) =>
      kind === 'all'
        ? t('widgets.diagnostics.scopes.all')
        : kind === 'project'
          ? t('widgets.diagnostics.scopes.project')
          : t('widgets.diagnostics.scopes.projectAndApplication'),
    [t]
  );
  const scopeCollection = useMemo(
    () =>
      createListCollection<{ label: string; value: ScopeKind }>({
        items: (projectId === null ? (['all'] as const) : (['project-and-application', 'project', 'all'] as const)).map(
          (kind) => ({ label: scopeLabel(kind), value: kind })
        ),
      }),
    [projectId, scopeLabel]
  );
  const severityCollection = useMemo(
    () =>
      createListCollection<{ label: string; value: SeverityFilter }>({
        items: SEVERITY_FILTERS.map((severity) => ({
          label:
            severity === 'all'
              ? t('widgets.diagnostics.severityAll')
              : t('widgets.diagnostics.severityAtLeast', { level: levelLabel(severity) }),
          value: severity,
        })),
      }),
    [levelLabel, t]
  );
  const namespaceCollection = useMemo(
    () =>
      createListCollection<{ label: string; value: NamespaceFilter }>({
        items: NAMESPACE_FILTERS.map((namespace) => ({
          label:
            namespace === 'all'
              ? t('widgets.diagnostics.namespaceAll')
              : namespace === 'performance'
                ? t('widgets.diagnostics.performanceNamespace')
                : namespace,
          value: namespace,
        })),
      }),
    [t]
  );
  const scopeValue = useMemo(() => [scope.kind], [scope.kind]);
  const severityValue = useMemo(() => [filters.severity], [filters.severity]);
  const namespaceValue = useMemo(() => [filters.namespace], [filters.namespace]);

  const changeScope = useCallback(({ value }: { value: string[] }) => {
    const next = value[0] as ScopeKind | undefined;

    if (next) {
      setFilters((current) => ({ ...current, page: 1, scope: next }));
    }
  }, []);
  const changeSeverity = useCallback(({ value }: { value: string[] }) => {
    const next = value[0] as SeverityFilter | undefined;

    if (next) {
      setFilters((current) => ({ ...current, page: 1, severity: next }));
    }
  }, []);
  const changeNamespace = useCallback(({ value }: { value: string[] }) => {
    const next = value[0] as NamespaceFilter | undefined;

    if (next) {
      setFilters((current) => ({ ...current, namespace: next, page: 1 }));
    }
  }, []);
  const changeQuery = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const query = event.currentTarget.value;

    setFilters((current) => ({ ...current, page: 1, query }));
  }, []);
  const clearFilters = useCallback(
    () => setFilters((current) => ({ ...current, namespace: 'all', page: 1, query: '', severity: 'all' })),
    []
  );
  const previousPage = useCallback(
    () => setFilters((current) => ({ ...current, page: Math.max(1, page - 1) })),
    [page]
  );
  const nextPage = useCallback(
    () => setFilters((current) => ({ ...current, page: Math.min(pageCount, page + 1) })),
    [page, pageCount]
  );

  const buildExport = useCallback(
    () =>
      JSON.stringify(
        createLogExport(matched, { appVersion: APP_VERSION, capture: config, retention: snapshot.retention, scope }),
        null,
        2
      ),
    [config, matched, scope, snapshot.retention]
  );
  const copyFiltered = useCallback(() => {
    const count = matched.length;
    const clipboard = navigator.clipboard;

    if (!clipboard) {
      toaster.create({ title: t('widgets.diagnostics.copyFailed'), type: 'error' });

      return;
    }

    clipboard.writeText(buildExport()).then(
      () => toaster.create({ duration: 2500, title: t('widgets.diagnostics.copied', { count }), type: 'success' }),
      () => toaster.create({ title: t('widgets.diagnostics.copyFailed'), type: 'error' })
    );
  }, [buildExport, matched.length, t]);
  const downloadFiltered = useCallback(() => {
    try {
      downloadText(
        buildExport(),
        `invoke-logs-${new Date().toISOString().replace(/[:.]/g, '-')}.json`,
        'application/json'
      );
      toaster.create({
        duration: 2500,
        title: t('widgets.diagnostics.downloaded', { count: matched.length }),
        type: 'success',
      });
    } catch {
      toaster.create({ title: t('widgets.diagnostics.downloadFailed'), type: 'error' });
    }
  }, [buildExport, matched.length, t]);
  const clearScope = useCallback(() => clearLogs(scope), [scope]);
  const openDeveloperSettings = useCallback(() => openWorkbenchSettings('developer'), []);

  const showProject = scope.kind === 'all';
  const currentScopeLabel = scopeLabel(scope.kind);

  return (
    <Stack data-diagnostics-panel flex="1" gap="2" minH="0" p="2">
      <HStack flexWrap="wrap" gap="2">
        <Select
          aria-label={t('widgets.diagnostics.scope')}
          collection={scopeCollection}
          flex="1 1 13rem"
          minW="0"
          positioning={SELECT_POSITIONING}
          size="xs"
          value={scopeValue}
          valueText={currentScopeLabel}
          onValueChange={changeScope}
        />
        <Select
          aria-label={t('widgets.diagnostics.severity')}
          collection={severityCollection}
          flex="1 1 9rem"
          minW="0"
          positioning={SELECT_POSITIONING}
          size="xs"
          value={severityValue}
          valueText={severityCollection.items.find((item) => item.value === filters.severity)?.label}
          onValueChange={changeSeverity}
        />
        <Select
          aria-label={t('widgets.diagnostics.namespace')}
          collection={namespaceCollection}
          flex="1 1 9rem"
          minW="0"
          positioning={SELECT_POSITIONING}
          size="xs"
          value={namespaceValue}
          valueText={namespaceCollection.items.find((item) => item.value === filters.namespace)?.label}
          onValueChange={changeNamespace}
        />
        <InputGroup flex="1 1 100%" startElement={SEARCH_START_ELEMENT}>
          <Input
            aria-label={t('widgets.diagnostics.search')}
            placeholder={t('widgets.diagnostics.search')}
            size="xs"
            value={filters.query}
            onChange={changeQuery}
          />
        </InputGroup>
      </HStack>
      <RecordingStatus config={config} snapshot={snapshot} />
      <HStack flexWrap="wrap" gap="2" justify="space-between">
        <HStack flexWrap="wrap" gap="2">
          <Button disabled={matched.length === 0} size="2xs" variant="outline" onClick={copyFiltered}>
            {t('widgets.diagnostics.copyFiltered', { count: matched.length })}
          </Button>
          <Button disabled={matched.length === 0} size="2xs" variant="outline" onClick={downloadFiltered}>
            {t('widgets.diagnostics.downloadFiltered', { count: matched.length })}
          </Button>
          <Button disabled={scopedCount === 0} size="2xs" variant="outline" onClick={clearScope}>
            {t('widgets.diagnostics.clearScope', { scope: currentScopeLabel })}
          </Button>
        </HStack>
        {pageCount > 1 ? (
          <HStack gap="1">
            <Tooltip content={t('common.previousPage')}>
              <IconButton
                aria-label={t('common.previousPage')}
                disabled={page <= 1}
                size="2xs"
                variant="ghost"
                onClick={previousPage}
              >
                <ChevronLeftIcon />
              </IconButton>
            </Tooltip>
            <Text color="fg.muted" fontSize="2xs" role="status">
              {t('widgets.diagnostics.page', { page, total: pageCount })}
            </Text>
            <Tooltip content={t('common.nextPage')}>
              <IconButton
                aria-label={t('common.nextPage')}
                disabled={page >= pageCount}
                size="2xs"
                variant="ghost"
                onClick={nextPage}
              >
                <ChevronRightIcon />
              </IconButton>
            </Tooltip>
          </HStack>
        ) : null}
      </HStack>
      {pageEntries.length === 0 ? (
        <Stack gap="2" py="4">
          {!config.enabled && scopedCount === 0 ? (
            <>
              <Text color="fg" fontSize="xs" fontWeight="600" role="status">
                {t('widgets.diagnostics.recordingOff')}
              </Text>
              <Text color="fg.muted" fontSize="2xs">
                {t('widgets.diagnostics.recordingOffHint')}
              </Text>
              <Button alignSelf="start" size="2xs" variant="outline" onClick={openDeveloperSettings}>
                {t('widgets.diagnostics.openDeveloperSettings')}
              </Button>
            </>
          ) : scopedCount === 0 ? (
            <Text color="fg.muted" fontSize="2xs" role="status">
              {t('widgets.diagnostics.emptyScope')}
            </Text>
          ) : (
            <>
              <Text color="fg.muted" fontSize="2xs" role="status">
                {t('widgets.diagnostics.noMatches')}
              </Text>
              {hasFilters ? (
                <Button alignSelf="start" size="2xs" variant="outline" onClick={clearFilters}>
                  {t('widgets.diagnostics.clearFilters')}
                </Button>
              ) : null}
            </>
          )}
        </Stack>
      ) : (
        <Scrollable flex="1" label={t('widgets.diagnostics.listLabel')} minH="0">
          <Stack gap="2" pe="2">
            {pageEntries.map((entry) => (
              <DiagnosticsEntryRow key={entry.id} entry={entry} showProject={showProject} />
            ))}
          </Stack>
        </Scrollable>
      )}
    </Stack>
  );
};

const RecordingStatus = ({ config, snapshot }: { config: LoggingConfig; snapshot: LogSnapshot }) => {
  const { t } = useTranslation();
  const { retention, truncatedCount } = snapshot;
  const evicted = retention.problems.evicted + retention.verbose.evicted + retention.timings.evicted;

  return (
    <HStack flexWrap="wrap" gap="1.5">
      {config.enabled ? (
        <Badge colorPalette="green" size="xs">
          {t('widgets.diagnostics.recordingStatus', {
            count: config.namespaces.length,
            level: t(`settings.catalog.options.${config.level}`),
            total: LOG_NAMESPACES.length,
          })}
        </Badge>
      ) : (
        <Badge colorPalette="gray" size="xs">
          {t('widgets.diagnostics.recordingOff')}
        </Badge>
      )}
      {config.performanceTimingsEnabled ? (
        <Badge colorPalette="purple" size="xs">
          {t('widgets.diagnostics.timingsOn')}
        </Badge>
      ) : null}
      {config.consoleOutputEnabled ? (
        <Badge colorPalette="blue" size="xs">
          {t('widgets.diagnostics.consoleOn')}
        </Badge>
      ) : null}
      <Text color="fg.muted" fontSize="2xs">
        {t('widgets.diagnostics.retention', {
          problemLimit: retention.problems.limit,
          problems: retention.problems.count,
          timingLimit: retention.timings.limit,
          timings: retention.timings.count,
          verbose: retention.verbose.count,
          verboseLimit: retention.verbose.limit,
        })}
      </Text>
      {evicted > 0 ? (
        <Badge colorPalette="orange" size="xs">
          {t('widgets.diagnostics.evicted', { count: evicted })}
        </Badge>
      ) : null}
      {truncatedCount > 0 ? (
        <Badge colorPalette="orange" size="xs">
          {t('widgets.diagnostics.truncated', { count: truncatedCount })}
        </Badge>
      ) : null}
    </HStack>
  );
};

const DiagnosticsEntryRow = memo(({ entry, showProject }: { entry: LogEntry; showProject: boolean }) => {
  const { t } = useTranslation();
  const [isExpanded, setIsExpanded] = useState(false);
  const toggle = useCallback(() => setIsExpanded((current) => !current), []);
  const isProblem = entry.level === 'warn' || entry.level === 'error' || entry.level === 'fatal';

  return (
    <Panel
      borderColor={isProblem ? (entry.level === 'warn' ? 'fg.warning' : 'border.error') : undefined}
      data-log-entry-id={entry.id}
      data-log-level={entry.level}
      p="2"
    >
      <Stack gap="1.5">
        <HStack align="start" gap="2" justify="space-between">
          <Stack flex="1" gap="1" minW="0">
            <HStack flexWrap="wrap" gap="1.5">
              <Badge colorPalette={LEVEL_COLOR_PALETTE[entry.level]} size="xs">
                {t(`settings.catalog.options.${entry.level}`)}
              </Badge>
              {entry.durationMs !== undefined ? (
                <Badge colorPalette="purple" size="xs">
                  {entry.durationMs.toFixed(1)}ms
                </Badge>
              ) : null}
              {entry.truncated ? (
                <Badge colorPalette="orange" size="xs">
                  {t('widgets.diagnostics.entryTruncated')}
                </Badge>
              ) : null}
              <Text color="fg.muted" fontFamily="mono" fontSize="2xs" minW="0" wordBreak="break-all">
                {formatSource(entry, showProject)}
              </Text>
            </HStack>
            <Text color="fg" fontSize="xs" fontWeight="600" wordBreak="break-word">
              {entry.message || entry.name}
            </Text>
            {entry.error ? (
              <Text color="fg.muted" fontFamily="mono" fontSize="2xs" wordBreak="break-word">
                {entry.error.name}: {entry.error.message}
              </Text>
            ) : null}
            <Text color="fg.muted" fontSize="2xs">
              {new Date(entry.createdAt).toLocaleTimeString()} · {entry.name}
            </Text>
          </Stack>
          <Button aria-expanded={isExpanded} flexShrink="0" size="2xs" variant="ghost" onClick={toggle}>
            {isExpanded ? t('widgets.diagnostics.hideDetails') : t('widgets.diagnostics.details')}
          </Button>
        </HStack>
        {isExpanded ? (
          <Box data-log-entry-details>
            <JsonPreview
              copyFailedLabel={t('widgets.diagnostics.copyFailed')}
              copyLabel={t('widgets.diagnostics.copyEntry')}
              label={t('widgets.diagnostics.detailsLabel')}
              maxH="16rem"
              value={entry}
            />
          </Box>
        ) : null}
      </Stack>
    </Panel>
  );
});

DiagnosticsEntryRow.displayName = 'DiagnosticsEntryRow';
