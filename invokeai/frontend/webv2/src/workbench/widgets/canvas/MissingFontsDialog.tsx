import type { CanvasEngine, CanvasFontCapability } from '@workbench/canvas-engine/api';

import { Box, Dialog, Flex, Input, NativeSelect, Portal, Stack, Text } from '@chakra-ui/react';
import { fontKeys, fontsQueryOptions, getFont, uploadFont, type FontRecord } from '@features/fonts';
import { useMountEffect } from '@platform/react/useMountEffect';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { Button, CloseButton } from '@platform/ui';
import { useQueries, useQuery, useQueryClient } from '@tanstack/react-query';
import { registerHotkeyModalLayer } from '@workbench/hotkeys/modalLayer';
import { useCallback, useDeferredValue, useMemo, useRef, useState, type ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';

type FontGroup = ReturnType<CanvasFontCapability['collectReferences']>[number];

export const MissingFontsDialog = ({ engine, groups }: { engine: CanvasEngine; groups: readonly FontGroup[] }) => {
  const { t } = useTranslation('fonts');
  const queries = useQueries({
    queries: groups.map(({ fontRef }) => ({
      queryKey: fontKeys.detail(fontRef.id),
      queryFn: ({ signal }: { signal: AbortSignal }) => getFont(fontRef.id, signal),
      retry: false,
      staleTime: 30_000,
    })),
  });
  const unavailable = useMemo(
    () =>
      groups.filter((group, index) => {
        const query = queries[index];
        return query && !query.isPending && (query.isError || query.data?.contentHash !== group.fontRef.contentHash);
      }),
    [groups, queries]
  );
  const signature = JSON.stringify(unavailable.map(({ fontRef }) => [fontRef.id, fontRef.contentHash]));
  const [dismissed, setDismissed] = useState<string | null>(null);
  const close = useCallback(() => setDismissed(signature), [signature]);
  const reopen = useCallback(() => setDismissed(null), []);
  if (!unavailable.length) {
    return null;
  }
  return (
    <>
      <Box position="absolute" top="2" left="50%" transform="translateX(-50%)" zIndex="2">
        <Button colorPalette="orange" size="xs" onClick={reopen}>
          {t('fonts.missing.warning', { count: unavailable.length })}
        </Button>
      </Box>
      {dismissed !== signature ? <RecoveryDialog groups={unavailable} engine={engine} onClose={close} /> : null}
    </>
  );
};

const RecoveryDialog = ({
  groups,
  engine,
  onClose,
}: {
  groups: FontGroup[];
  engine: CanvasEngine;
  onClose: () => void;
}) => {
  const { t } = useTranslation('fonts');
  const queryClient = useQueryClient();
  useMountEffect(() => registerHotkeyModalLayer('missing-fonts'));
  const onOpenChange = useCallback(
    ({ open }: { open: boolean }) => {
      if (!open) {
        onClose();
      }
    },
    [onClose]
  );
  const retry = useCallback(() => {
    void queryClient.invalidateQueries({ queryKey: fontKeys.all });
  }, [queryClient]);
  return (
    <Dialog.Root open placement="center" size="lg" scrollBehavior="inside" onOpenChange={onOpenChange}>
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content>
            <Dialog.Header>
              <Dialog.Title>{t('fonts.missing.title')}</Dialog.Title>
            </Dialog.Header>
            <Dialog.Body>
              <Stack gap="4">
                <Text color="fg.muted" fontSize="sm">
                  {t('fonts.missing.description')}
                </Text>
                {groups.map((group) => (
                  <RecoveryRow key={`${group.fontRef.id}:${group.fontRef.contentHash}`} group={group} engine={engine} />
                ))}
              </Stack>
            </Dialog.Body>
            <Dialog.Footer>
              <Button color="fg" variant="outline" onClick={retry}>
                {t('common.retry')}
              </Button>
              <Button colorPalette="accent" onClick={onClose}>
                {t('fonts.missing.continue')}
              </Button>
            </Dialog.Footer>
            <Dialog.CloseTrigger asChild>
              <CloseButton aria-label={t('common.close')} />
            </Dialog.CloseTrigger>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};

const RecoveryRow = ({ group, engine }: { group: FontGroup; engine: CanvasEngine }) => {
  const { t } = useTranslation('fonts');
  const queryClient = useQueryClient();
  const [search, setSearch] = useState('');
  const deferredSearch = useDeferredValue(search);
  const [offset, setOffset] = useState(0);
  const catalog = useQuery(fontsQueryOptions({ limit: 50, search: deferredSearch, ...(offset ? { offset } : {}) }));
  const [selected, setSelected] = useState<FontRecord | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const [owner] = useState(captureAccountScope);
  const searchChanged = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    setSearch(event.target.value);
    setOffset(0);
  }, []);
  const previousPage = useCallback(() => setOffset((current) => Math.max(0, current - 50)), []);
  const nextPage = useCallback(() => setOffset((current) => current + 50), []);
  const selectionChanged = useCallback(
    (event: ChangeEvent<HTMLSelectElement>) => {
      setSelected(catalog.data?.items.find((font) => font.id === event.target.value) ?? null);
    },
    [catalog.data]
  );
  const retryCatalog = useCallback(() => void catalog.refetch(), [catalog]);
  const chooseFile = useCallback(() => inputRef.current?.click(), []);
  const uploaded = useCallback(
    async (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      event.target.value = '';
      if (!file || !isAccountScopeCurrent(owner)) {
        return;
      }
      setBusy(true);
      setError(null);
      try {
        const result = await uploadFont(file, 'private', owner.signal);
        if (!isAccountScopeCurrent(owner)) {
          return;
        }
        setSelected(result.font);
        await queryClient.invalidateQueries({ queryKey: fontKeys.all });
      } catch (failure) {
        if (isAccountScopeCurrent(owner)) {
          setError(failure instanceof Error ? failure.message : t('fonts.missing.failed'));
        }
      } finally {
        if (isAccountScopeCurrent(owner)) {
          setBusy(false);
        }
      }
    },
    [owner, queryClient, t]
  );
  const replace = useCallback(() => {
    if (!selected || !isAccountScopeCurrent(owner)) {
      return;
    }
    const result = engine.fonts.replaceAllReferences(group.fontRef, {
      fontRef: { id: selected.id, contentHash: selected.contentHash, family: selected.family, label: selected.label },
      axes: selected.axes,
      style: selected.style === 'italic' || selected.style === 'oblique' ? selected.style : 'normal',
      weight: selected.weight,
    });
    if (result.status !== 'committed' && result.status !== 'unchanged') {
      setError(t('fonts.missing.failed'));
    }
  }, [engine, group.fontRef, owner, selected, t]);
  const choices =
    selected && !catalog.data?.items.some((font) => font.id === selected.id)
      ? [selected, ...(catalog.data?.items ?? [])]
      : (catalog.data?.items ?? []);
  return (
    <Stack borderWidth="1px" borderRadius="md" p="3" gap="2">
      <Text fontWeight="medium">{group.fontRef.label}</Text>
      <Text color="fg.muted" fontSize="xs">
        {t('fonts.missing.layers', { count: group.count })}
      </Text>
      <Input
        aria-label={t('fonts.missing.search')}
        placeholder={t('fonts.missing.search')}
        size="sm"
        value={search}
        onChange={searchChanged}
      />
      <NativeSelect.Root size="sm">
        <NativeSelect.Field
          aria-label={t('fonts.missing.replacement')}
          value={selected?.id ?? ''}
          onChange={selectionChanged}
        >
          <option value="">{t('fonts.missing.select')}</option>
          {choices.map((font) => (
            <option key={font.id} value={font.id}>
              {font.label}
            </option>
          ))}
        </NativeSelect.Field>
        <NativeSelect.Indicator />
      </NativeSelect.Root>
      {offset > 0 || (catalog.data?.total ?? 0) > 50 ? (
        <Flex gap="2" justify="flex-end">
          <Button
            size="xs"
            variant="outline"
            color="fg"
            disabled={offset === 0 || catalog.isFetching}
            onClick={previousPage}
          >
            {t('common.previousPage')}
          </Button>
          <Button
            size="xs"
            variant="outline"
            color="fg"
            disabled={catalog.isFetching || offset + 50 >= (catalog.data?.total ?? 0)}
            onClick={nextPage}
          >
            {t('common.nextPage')}
          </Button>
        </Flex>
      ) : null}
      {catalog.isPending ? <Text fontSize="xs">{t('common.loading')}</Text> : null}
      {catalog.isError ? (
        <Text role="alert" color="fg.error" fontSize="xs">
          {t('fonts.missing.catalogError')}
          <Button size="xs" color="fg" variant="outline" ml="2" onClick={retryCatalog}>
            {t('common.retry')}
          </Button>
        </Text>
      ) : null}
      {selected ? (
        <Text fontSize="xs" color="fg.muted">
          {t('fonts.missing.axesNotice')}
        </Text>
      ) : null}
      {error ? (
        <Text role="alert" color="fg.error" fontSize="xs">
          {error}
        </Text>
      ) : null}
      <Flex gap="2" justify="flex-end">
        <input ref={inputRef} type="file" accept=".ttf,.otf,.woff,.woff2" hidden onChange={uploaded} />
        <Button color="fg" variant="outline" size="sm" loading={busy} onClick={chooseFile}>
          {t('fonts.missing.upload')}
        </Button>
        <Button colorPalette="accent" size="sm" disabled={!selected || busy} onClick={replace}>
          {t('fonts.missing.replaceAll')}
        </Button>
      </Flex>
    </Stack>
  );
};
