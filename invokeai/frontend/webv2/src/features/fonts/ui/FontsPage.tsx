/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop */
import type { ListRowProps } from '@platform/ui/list/List';
import type { ChangeEvent } from 'react';

import {
  Badge,
  Box,
  Flex,
  HStack,
  Icon,
  Input,
  InputGroup,
  Menu,
  Portal,
  Spinner,
  Stack,
  Text,
} from '@chakra-ui/react';
import {
  deleteFont,
  fontKeys,
  fontsQueryOptions,
  getFontRuntimeKey,
  rescanFonts,
  uploadFont,
  type FontDownloadReference,
  type FontRecord,
  type FontScope,
  type FontListParams,
} from '@features/fonts';
import { useFontRuntime, useFontRuntimeSnapshot } from '@features/fonts/react';
import { useCapabilities } from '@features/identity';
import { formatBytes } from '@platform/i18n/languages';
import { useMountEffect } from '@platform/react/useMountEffect';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, ConfirmDialog, IconButton, Scrollable, SegmentedControl, Tabs } from '@platform/ui';
import { EmptyState } from '@platform/ui/EmptyState';
import { List } from '@platform/ui/list/List';
import { ListItem } from '@platform/ui/list/ListItem';
import { ListPager } from '@platform/ui/list/ListPager';
import { listRowsFromItems } from '@platform/ui/list/listRows';
import { ManagerColumn, ManagerDetailHeader } from '@platform/ui/ManagerLayout';
import { MenuContent } from '@platform/ui/Menu';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  ArrowRightIcon,
  CircleAlertIcon,
  FileTypeIcon,
  PackageOpenIcon,
  PlusIcon,
  RefreshCwIcon,
  SearchIcon,
  SlidersHorizontalIcon,
  Trash2Icon,
  UploadIcon,
} from 'lucide-react';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

const FONT_ACCEPT = '.ttf,.otf,.woff,.woff2,font/ttf,font/otf,font/woff,font/woff2';
const FONT_PAGE_SIZE = 100;
const NO_FONTS: readonly FontRecord[] = [];
const getFontId = (font: FontRecord): string => font.id;
const FILTER_POSITION = { placement: 'bottom-end' } as const;
const PREVIEW_TEXT_SX = { textWrap: 'pretty' } as const;
const ERROR_ICON = <CircleAlertIcon />;
const EMPTY_ICON = <Icon as={PackageOpenIcon} />;
const EMPTY_SEARCH_ICON = <Icon as={SearchIcon} />;
const SEARCH_ICON = <Icon as={SearchIcon} boxSize="3.5" color="fg.subtle" />;
let nextUploadId = 1;

type FontFilter = 'all' | 'private' | 'shared';
type UploadStatus = 'queued' | 'uploading' | 'created' | 'duplicate' | 'error';

interface UploadItem {
  id: string;
  name: string;
  status: UploadStatus;
  message?: string;
}

const getScopeLabelKey = (scope: FontScope): string =>
  scope === 'private' ? 'fonts.scope.private' : 'fonts.scope.shared';

const FontPreview = ({ font, compact = false }: { font: FontRecord; compact?: boolean }) => {
  const { t } = useTranslation('fonts');
  const runtime = useFontRuntime();
  const snapshot = useFontRuntimeSnapshot();
  const reference = useMemo<FontDownloadReference>(
    () => ({
      contentHash: font.contentHash,
      family: font.family,
      id: font.id,
      style: font.style,
      weight: font.weight,
    }),
    [font.contentHash, font.family, font.id, font.style, font.weight]
  );
  const key = getFontRuntimeKey(reference);
  const loadState = snapshot.states.get(key) ?? 'idle';
  const family = runtime.resolveFamily(reference);

  const load = useCallback(() => {
    void runtime.ensure(reference).catch(() => undefined);
  }, [reference, runtime]);

  useMountEffect(() => {
    const controller = new AbortController();
    // Small list samples may be evicted to make room for the selected face or Canvas.
    const release = compact ? undefined : runtime.retain(reference);
    void runtime.ensure(reference, controller.signal).catch(() => undefined);
    return () => {
      controller.abort();
      release?.();
    };
  });

  if (compact) {
    return (
      <Box
        aria-hidden="true"
        alignItems="center"
        display="flex"
        flexShrink={0}
        h="8"
        justifyContent="center"
        overflow="hidden"
        w="9"
      >
        <Text fontFamily={family} fontSize="xl" fontStyle={font.style} fontWeight={font.weight} lineHeight="1">
          Aa
        </Text>
      </Box>
    );
  }

  return (
    <Box bg="bg.inset" borderColor="border.subtle" borderWidth="1px" minH="16" p="3" rounded="md">
      <Text
        css={PREVIEW_TEXT_SX}
        fontFamily={family}
        fontSize="3xl"
        fontStyle={font.style === 'italic' || font.style === 'oblique' ? font.style : 'normal'}
        fontWeight={font.weight}
        lineHeight="1.25"
        overflowWrap="anywhere"
      >
        {t('fonts.previewText')}
      </Text>
      {loadState === 'loading' ? (
        <HStack color="fg.muted" gap="1.5" mt="2">
          <Spinner size="xs" />
          <Text fontSize="2xs">{t('fonts.loadingPreview')}</Text>
        </HStack>
      ) : loadState === 'error' ? (
        <Button size="2xs" variant="ghost" onClick={load}>
          {t('fonts.retryPreview')}
        </Button>
      ) : null}
    </Box>
  );
};

const FontDetail = ({
  canDelete,
  font,
  onDelete,
}: {
  canDelete: boolean;
  font: FontRecord;
  onDelete: (font: FontRecord) => void;
}) => {
  const { t } = useTranslation('fonts');
  const scopeLabel = t(getScopeLabelKey(font.scope));
  const sourceLabel = font.source === 'directory' ? t('fonts.source.directory') : t('fonts.source.uploaded');
  const variantLabel = `${font.style} · ${font.weight}`;

  return (
    <Stack gap="4" p="3">
      <Flex align="flex-start" gap="2" justify="space-between" minW="0">
        <Stack gap="0.5" minW="0">
          <Text as="h3" fontSize="lg" fontWeight="600" overflowWrap="anywhere">
            {font.label || font.family}
          </Text>
          <Text color="fg.muted" fontSize="2xs" overflowWrap="anywhere">
            {font.filename} · {variantLabel}
          </Text>
        </Stack>
        {canDelete ? (
          <Button
            aria-label={t('fonts.deleteNamed', { name: font.label || font.family })}
            color="fg.muted"
            size="2xs"
            variant="ghost"
            onClick={() => onDelete(font)}
          >
            <Trash2Icon />
          </Button>
        ) : null}
      </Flex>
      <HStack gap="1.5" wrap="wrap">
        <Badge colorPalette={font.scope === 'shared' ? 'purple' : 'gray'} fontSize="2xs" variant="surface">
          {scopeLabel}
        </Badge>
        <Badge fontSize="2xs" variant="surface">
          {sourceLabel}
        </Badge>
        {font.axes.length > 0 ? (
          <Badge fontSize="2xs" variant="surface">
            {t('fonts.axisCount', { count: font.axes.length })}
          </Badge>
        ) : null}
      </HStack>
      <FontPreview key={`${font.id}:${font.contentHash}:${font.family}:${font.style}:${font.weight}`} font={font} />
      <HStack align="center" color="fg.muted" gap="2" minW="0" wrap="wrap">
        <Text fontSize="2xs" overflowWrap="anywhere">
          {font.instances.length > 0
            ? t('fonts.namedInstances', { count: font.instances.length })
            : t('fonts.fileSize', { bytes: formatBytes(font.byteSize) })}
        </Text>
        {font.axes.length > 0 ? (
          <Text fontSize="2xs" overflowWrap="anywhere">
            {font.axes.map((axis) => axis.tag).join(', ')}
          </Text>
        ) : null}
      </HStack>
      {font.axes.length > 0 ? (
        <Stack gap="2">
          <Text fontSize="xs" fontWeight="600">
            {t('fonts.axesTitle')}
          </Text>
          {font.axes.map((axis) => (
            <HStack key={axis.tag} borderBottomWidth="1px" gap="3" justify="space-between" py="2" wrap="wrap">
              <Text fontSize="xs">
                {axis.label}{' '}
                <Text as="span" color="fg.muted">
                  {axis.tag}
                </Text>
              </Text>
              <Text color="fg.muted" fontSize="xs" fontVariantNumeric="tabular-nums">
                {axis.minimum} – {axis.maximum} · {t('fonts.axisDefault', { value: axis.default })}
              </Text>
            </HStack>
          ))}
        </Stack>
      ) : null}
      {font.instances.length > 0 ? (
        <Stack gap="2">
          <Text fontSize="xs" fontWeight="600">
            {t('fonts.instancesTitle')}
          </Text>
          <HStack gap="1.5" wrap="wrap">
            {font.instances.map((instance, index) => (
              <Badge key={index} variant="surface">
                {instance.name}
              </Badge>
            ))}
          </HStack>
        </Stack>
      ) : null}
    </Stack>
  );
};

const UploadStatusList = ({ items }: { items: readonly UploadItem[] }) => {
  const { t } = useTranslation('fonts');

  if (items.length === 0) {
    return null;
  }

  return (
    <Stack aria-live="polite" gap="1.5">
      {items.map((item) => (
        <HStack key={item.id} color={item.status === 'error' ? 'fg.error' : 'fg.muted'} gap="2">
          {item.status === 'uploading' ? (
            <Spinner size="xs" />
          ) : (
            <Icon as={item.status === 'error' ? CircleAlertIcon : FileTypeIcon} />
          )}
          <Text fontSize="2xs" overflowWrap="anywhere">
            {item.name}
          </Text>
          <Text fontSize="2xs">
            {item.status === 'queued' || item.status === 'uploading'
              ? t('fonts.uploading')
              : item.status === 'created'
                ? t('fonts.uploaded')
                : item.status === 'duplicate'
                  ? t('fonts.duplicate')
                  : (item.message ?? t('fonts.uploadFailed'))}
          </Text>
        </HStack>
      ))}
    </Stack>
  );
};

const FontLibrary = () => {
  const { t } = useTranslation('fonts');
  const { canManageSharedFonts } = useCapabilities();
  const queryClient = useQueryClient();
  const [filter, setFilter] = useState<FontFilter>('all');
  const [activeTab, setActiveTab] = useState('details');
  const [selectedFont, setSelectedFont] = useState<FontRecord | null>(null);
  const [search, setSearch] = useState('');
  const [uploadScope, setUploadScope] = useState<FontScope>('private');
  const [uploads, setUploads] = useState<UploadItem[]>([]);
  const [deleteTarget, setDeleteTarget] = useState<FontRecord | null>(null);
  const [isRescanning, setIsRescanning] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const uploadChainRef = useRef<Promise<void>>(Promise.resolve());
  const [offset, setOffset] = useState(0);
  const params = useMemo<FontListParams>(
    () => ({
      limit: FONT_PAGE_SIZE,
      scope: filter === 'all' ? 'all' : filter,
      search,
      ...(offset > 0 ? { offset } : {}),
    }),
    [filter, offset, search]
  );
  const query = useQuery(fontsQueryOptions(params));
  const fonts = query.data?.items ?? NO_FONTS;
  const rows = useMemo(() => listRowsFromItems(fonts, getFontId), [fonts]);
  // A new page, filter or search replaces every row: remount the list so it starts from the top whether or not
  // the query is cached.
  const listKey = `${filter}:${offset}:${search}`;

  const invalidateFonts = useCallback(() => queryClient.invalidateQueries({ queryKey: fontKeys.all }), [queryClient]);
  const handleFiles = useCallback(
    (files: FileList | File[]) => {
      const selected = [...files];
      if (selected.length === 0) {
        return Promise.resolve();
      }
      const scope = canManageSharedFonts ? uploadScope : 'private';
      const owner = captureAccountScope();
      const next = selected.map<UploadItem>((file) => ({
        id: `upload-${nextUploadId++}`,
        name: file.name,
        status: 'queued',
      }));
      setUploads((current) => [...next, ...current].slice(0, 12));

      const runBatch = async (): Promise<void> => {
        for (const [index, file] of selected.entries()) {
          const item = next[index]!;
          try {
            assertAccountScopeCurrent(owner);
            setUploads((current) =>
              current.map((entry) => (entry.id === item.id ? { ...entry, status: 'uploading' } : entry))
            );
            const result = await uploadFont(file, scope, owner.signal);
            assertAccountScopeCurrent(owner);
            setUploads((current) =>
              current.map((item) =>
                item.id === next[index]!.id ? { ...item, status: result.created ? 'created' : 'duplicate' } : item
              )
            );
          } catch (error) {
            const ownerIsCurrent = isAccountScopeCurrent(owner);
            setUploads((current) =>
              current.map((item) =>
                item.id === next[index]!.id
                  ? {
                      ...item,
                      message: ownerIsCurrent
                        ? getApiErrorMessage(error, t('fonts.uploadFailed'))
                        : t('fonts.uploadFailed'),
                      status: 'error',
                    }
                  : item
              )
            );
            if (!ownerIsCurrent) {
              const pendingIds = new Set(next.slice(index + 1).map(({ id }) => id));
              setUploads((current) =>
                current.map((item) =>
                  pendingIds.has(item.id)
                    ? { ...item, message: t('fonts.uploadFailed'), status: 'error' as const }
                    : item
                )
              );
              break;
            }
          }
        }
        if (isAccountScopeCurrent(owner)) {
          await invalidateFonts();
        }
      };

      const queuedBatch = uploadChainRef.current.then(runBatch, runBatch);
      const settledBatch = queuedBatch.catch(() => undefined);
      uploadChainRef.current = settledBatch;
      return settledBatch;
    },
    [canManageSharedFonts, invalidateFonts, t, uploadScope]
  );
  const handleInputChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      if (event.currentTarget.files) {
        void handleFiles(event.currentTarget.files);
      }
      event.currentTarget.value = '';
    },
    [handleFiles]
  );
  const handleDelete = useCallback(async () => {
    if (!deleteTarget) {
      return;
    }
    try {
      await deleteFont(deleteTarget.id);
      setSelectedFont((current) => (current?.id === deleteTarget.id ? null : current));
      setDeleteTarget(null);
      await invalidateFonts();
    } catch (error) {
      setUploads((current) =>
        [
          {
            id: `delete:${deleteTarget.id}`,
            message: getApiErrorMessage(error, t('fonts.deleteFailed')),
            name: deleteTarget.label,
            status: 'error' as const,
          },
          ...current,
        ].slice(0, 12)
      );
    }
  }, [deleteTarget, invalidateFonts, t]);
  const handleRescan = useCallback(async () => {
    setIsRescanning(true);
    try {
      await rescanFonts();
      await queryClient.invalidateQueries({ queryKey: fontKeys.all });
    } catch (error) {
      setUploads((current) =>
        [
          {
            id: `rescan-${Date.now()}`,
            message: getApiErrorMessage(error, t('fonts.rescanFailed')),
            name: t('fonts.rescan'),
            status: 'error' as const,
          },
          ...current,
        ].slice(0, 12)
      );
    } finally {
      setIsRescanning(false);
    }
  }, [queryClient, t]);
  const uploadScopeOptions = useMemo(
    () => [
      { label: t('fonts.scope.private'), value: 'private' },
      ...(canManageSharedFonts ? [{ label: t('fonts.scope.shared'), value: 'shared' }] : []),
    ],
    [canManageSharedFonts, t]
  );
  const filterOptions = useMemo(
    () => [
      { label: t('fonts.filters.all'), value: 'all' },
      { label: t('fonts.filters.my'), value: 'private' },
      { label: t('fonts.filters.shared'), value: 'shared' },
    ],
    [t]
  );
  const handleRetry = useCallback(() => void query.refetch(), [query]);
  const hasPreviousPage = offset > 0;
  const hasNextPage = query.data !== undefined && offset + fonts.length < query.data.total;
  const handlePreviousPage = useCallback(() => {
    setOffset((current) => Math.max(0, current - FONT_PAGE_SIZE));
  }, []);
  const handleNextPage = useCallback(() => {
    if (hasNextPage) {
      setOffset((current) => current + FONT_PAGE_SIZE);
    }
  }, [hasNextPage]);
  const showPagination = offset > 0 || (query.data?.total ?? 0) > FONT_PAGE_SIZE;
  const canDelete = useCallback(
    (font: FontRecord) => font.source === 'uploaded' && (font.scope === 'private' || canManageSharedFonts),
    [canManageSharedFonts]
  );
  const deleteBody = useMemo(
    () => t('fonts.deleteBody', { name: deleteTarget?.label ?? '' }),
    [deleteTarget?.label, t]
  );
  const effectiveUploadScope = canManageSharedFonts ? uploadScope : 'private';

  const activeFont = fonts.find((font) => font.id === selectedFont?.id) ?? selectedFont;

  return (
    <Flex aria-label={t('fonts.title')} role="region" h="full" minH="0" w="full">
      <input ref={inputRef} accept={FONT_ACCEPT} hidden multiple type="file" onChange={handleInputChange} />
      <ManagerColumn
        actions={
          canManageSharedFonts ? (
            <Button
              aria-label={t('fonts.rescan')}
              disabled={isRescanning}
              size="2xs"
              variant="ghost"
              onClick={handleRescan}
            >
              <RefreshCwIcon />
              {t('fonts.rescan')}
            </Button>
          ) : null
        }
        count={query.data?.total ?? '–'}
        title={t('fonts.title')}
      >
        <HStack gap="1.5" p="3">
          <InputGroup startElement={SEARCH_ICON}>
            <Input
              aria-label={t('fonts.searchLabel')}
              placeholder={t('fonts.searchPlaceholder')}
              size="xs"
              value={search}
              onChange={(event) => {
                setSearch(event.currentTarget.value);
                setOffset(0);
              }}
            />
          </InputGroup>
          <Menu.Root closeOnSelect={false} positioning={FILTER_POSITION}>
            <Menu.Trigger asChild>
              <IconButton
                aria-label={t('fonts.filterMenu')}
                color={filter !== 'all' ? 'accent.solid' : 'fg.muted'}
                size="xs"
                variant="outline"
              >
                <Icon as={SlidersHorizontalIcon} boxSize="4" />
              </IconButton>
            </Menu.Trigger>
            <Portal>
              <Menu.Positioner>
                <MenuContent minW="13rem">
                  <Menu.RadioItemGroup
                    value={filter}
                    onValueChange={(event) => {
                      setFilter(event.value as FontFilter);
                      setOffset(0);
                    }}
                  >
                    <Menu.ItemGroupLabel color="fg" fontSize="2xs" textTransform="uppercase">
                      {t('fonts.filterLabel')}
                    </Menu.ItemGroupLabel>
                    {filterOptions.map((option) => (
                      <Menu.RadioItem key={option.value} value={option.value}>
                        <Menu.ItemIndicator />
                        <Menu.ItemText fontSize="xs">{option.label}</Menu.ItemText>
                      </Menu.RadioItem>
                    ))}
                  </Menu.RadioItemGroup>
                </MenuContent>
              </Menu.Positioner>
            </Portal>
          </Menu.Root>
        </HStack>
        <List
          key={listKey}
          activeKey={selectedFont?.id ?? null}
          density="comfortable"
          emptyState={
            <EmptyState
              description={search.trim() ? t('fonts.noSearchMatchesDescription') : t('fonts.emptyDescription')}
              icon={search.trim() ? EMPTY_SEARCH_ICON : EMPTY_ICON}
              title={search.trim() ? t('fonts.noSearchMatches') : t('fonts.emptyTitle')}
            >
              {search.trim() ? (
                <Button
                  size="xs"
                  variant="outline"
                  onClick={() => {
                    setSearch('');
                    setOffset(0);
                  }}
                >
                  {t('common.clearSearch')}
                </Button>
              ) : (
                <Button size="sm" onClick={() => setActiveTab('add')}>
                  {t('fonts.addFonts')}
                  <Icon as={ArrowRightIcon} />
                </Button>
              )}
            </EmptyState>
          }
          errorState={
            <EmptyState
              danger
              description={query.isError ? getApiErrorMessage(query.error, t('fonts.couldNotLoad')) : null}
              icon={ERROR_ICON}
              title={t('fonts.couldNotLoad')}
            >
              <Button size="xs" variant="outline" onClick={handleRetry}>
                {t('common.retry')}
              </Button>
            </EmptyState>
          }
          label={t('fonts.library')}
          renderItem={(font: FontRecord, rowProps: ListRowProps) => (
            <ListItem
              {...rowProps}
              description={font.filename}
              leading={
                <FontPreview compact key={`${font.id}:${font.contentHash}:${font.style}:${font.weight}`} font={font} />
              }
              title={font.label || font.family}
              trailing={
                <Badge fontSize="2xs" variant="surface">
                  {t(getScopeLabelKey(font.scope))}
                </Badge>
              }
              onPress={() => {
                setSelectedFont(font);
                setActiveTab('details');
              }}
            />
          )}
          rows={rows}
          status={query.isPending ? 'loading' : query.isError ? 'error' : 'ready'}
        />
        {showPagination ? (
          <ListPager
            hasNext={hasNextPage}
            hasPrevious={hasPreviousPage}
            isBusy={query.isFetching}
            page={Math.floor(offset / FONT_PAGE_SIZE) + 1}
            onNext={handleNextPage}
            onPrevious={handlePreviousPage}
          />
        ) : null}
      </ManagerColumn>
      <Tabs.Root
        asChild
        lazyMount
        size="sm"
        unmountOnExit
        value={activeTab}
        onValueChange={(event) => setActiveTab(event.value)}
      >
        <Flex direction="column" flex="1" minH="0" minW="0">
          <ManagerDetailHeader>
            <Tabs.List mb="-1px">
              <Tabs.Trigger value="details">
                <Icon as={FileTypeIcon} boxSize="3" />
                <Text maxW="14rem" truncate>
                  {activeFont?.label || t('fonts.details')}
                </Text>
              </Tabs.Trigger>
              <Tabs.Trigger value="add">
                <Icon as={PlusIcon} boxSize="3" />
                {t('fonts.addFonts')}
              </Tabs.Trigger>
            </Tabs.List>
          </ManagerDetailHeader>
          <Box flex="1" minH="0">
            <Tabs.Content h="full" p="0" value="details">
              {activeFont ? (
                <Scrollable h="full" label={t('fonts.details')} minH="0">
                  <FontDetail canDelete={canDelete(activeFont)} font={activeFont} onDelete={setDeleteTarget} />
                </Scrollable>
              ) : (
                <Flex align="center" direction="column" gap="2" h="full" justify="center" p="6">
                  <Icon as={FileTypeIcon} boxSize="8" color="fg.subtle" />
                  <Text color="fg.muted" fontSize="sm" fontWeight="600">
                    {t('fonts.selectFont')}
                  </Text>
                  <Text color="fg.muted" fontSize="xs" maxW="22rem" textAlign="center">
                    {t('fonts.selectFontDescription')}
                  </Text>
                </Flex>
              )}
            </Tabs.Content>
            <Tabs.Content h="full" p="0" value="add">
              <Scrollable h="full" label={t('fonts.addFonts')} minH="0" p="3">
                <Stack align="start" gap="4" maxW="xl">
                  <Stack gap="1">
                    <Text as="h3" fontSize="sm" fontWeight="600">
                      {t('fonts.upload')}
                    </Text>
                    <Text color="fg.muted" fontSize="xs">
                      {t('fonts.description')}
                    </Text>
                    <Text color="fg.muted" fontSize="xs">
                      {t('fonts.uploadDescription')}
                    </Text>
                  </Stack>
                  {canManageSharedFonts ? (
                    <SegmentedControl
                      ariaLabel={t('fonts.uploadScopeLabel')}
                      isFullWidth={false}
                      options={uploadScopeOptions}
                      value={effectiveUploadScope}
                      onChange={(value) => setUploadScope(value as FontScope)}
                    />
                  ) : null}
                  <Button size="xs" onClick={() => inputRef.current?.click()}>
                    <UploadIcon />
                    {effectiveUploadScope === 'shared' ? t('fonts.uploadShared') : t('fonts.upload')}
                  </Button>
                </Stack>
              </Scrollable>
            </Tabs.Content>
          </Box>
          {uploads.length > 0 ? (
            <Box borderTopWidth="1px" maxH="32" overflowY="auto" p="3">
              <UploadStatusList items={uploads} />
            </Box>
          ) : null}
        </Flex>
      </Tabs.Root>
      <ConfirmDialog
        body={deleteBody}
        confirmLabel={t('fonts.deleteConfirm')}
        isOpen={deleteTarget !== null}
        title={t('fonts.deleteTitle')}
        onClose={() => setDeleteTarget(null)}
        onConfirm={handleDelete}
      />
    </Flex>
  );
};

export const FontsPage = () => <FontLibrary />;
