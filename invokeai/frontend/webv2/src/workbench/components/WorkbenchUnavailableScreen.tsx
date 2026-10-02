import type { RecoverableProjectDraft, SyncedWorkbenchPersistence } from '@workbench/projects/syncedPersistence';

import { Alert, Box, Button, Flex, Heading, HStack, Spinner, Stack, Text } from '@chakra-ui/react';
import { downloadText } from '@platform/browser/downloadBlob';
import { useMountEffect } from '@platform/react/useMountEffect';
import { QueueRecoveryNotice } from '@workbench/queue-integration/QueueRecoveryNotice';
import { useCallback, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

const getExportName = (draft: RecoverableProjectDraft): string => {
  const safeName = draft.projectId
    .trim()
    .replaceAll(/[^\w.-]+/g, '-')
    .replaceAll(/^-+|-+$/g, '');
  return `${safeName || draft.projectId}.json`;
};

export const WorkbenchUnavailableScreen = ({
  message,
  onRetry,
  persistence,
}: {
  message: string;
  onRetry(): void;
  persistence: SyncedWorkbenchPersistence;
}) => {
  const { t } = useTranslation();
  const { getRecoverableDraftDocument, listRecoverableDrafts } = persistence;
  const [drafts, setDrafts] = useState<RecoverableProjectDraft[] | 'unavailable' | null>(null);
  const [draftError, setDraftError] = useState<string | null>(null);
  const [exportingDraftKey, setExportingDraftKey] = useState<string | null>(null);
  const [nextCursor, setNextCursor] = useState<[string, string] | null>(null);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const loadingMoreRef = useRef(false);
  const listRecoverableDraftsRef = useRef(listRecoverableDrafts);
  const nextCursorRef = useRef<[string, string] | null>(null);

  useMountEffect(() => {
    let active = true;
    void listRecoverableDrafts().then((page) => {
      if (active) {
        if (page.kind === 'unavailable') {
          setDrafts('unavailable');
          return;
        }
        setDrafts(page.items);
        nextCursorRef.current = page.nextCursor;
        setNextCursor(page.nextCursor);
      }
    });
    return () => {
      active = false;
    };
  });

  const exportDraft = useCallback(
    async (draft: RecoverableProjectDraft) => {
      const key = `${draft.projectId}\u0000${draft.editorSessionId}`;
      setDraftError(null);
      setExportingDraftKey(key);
      try {
        const documentJson = await getRecoverableDraftDocument(draft.projectId, draft.editorSessionId);
        if (!documentJson) {
          throw new Error(t('shell.backendUnavailable.draftUnavailable'));
        }
        downloadText(documentJson, getExportName(draft), 'application/json');
      } catch (error) {
        setDraftError(error instanceof Error ? error.message : t('shell.backendUnavailable.exportFailed'));
      } finally {
        setExportingDraftKey(null);
      }
    },
    [getRecoverableDraftDocument, t]
  );
  const loadMore = useCallback(async () => {
    const cursor = nextCursorRef.current;
    if (!cursor || loadingMoreRef.current) {
      return;
    }
    loadingMoreRef.current = true;
    setIsLoadingMore(true);
    try {
      const page = await listRecoverableDraftsRef.current({ after: cursor });
      if (page.kind === 'unavailable') {
        setDrafts('unavailable');
        nextCursorRef.current = null;
        setNextCursor(null);
        return;
      }
      setDrafts((current) => [...(Array.isArray(current) ? current : []), ...page.items]);
      nextCursorRef.current = page.nextCursor;
      setNextCursor(page.nextCursor);
    } finally {
      loadingMoreRef.current = false;
      setIsLoadingMore(false);
    }
  }, []);
  const handleLoadMore = useCallback(() => void loadMore(), [loadMore]);

  return (
    <Flex align="center" bg="bg" color="fg" h="100vh" justify="center" p="6" w="100vw">
      <Box borderColor="border.subtle" borderRadius="xl" borderWidth="1px" maxW="2xl" p="6" shadow="sm" w="full">
        <Stack gap="5">
          <Stack gap="2">
            <Heading size="xl">{t('shell.backendUnavailable.title')}</Heading>
            <Text color="fg.muted">{t('shell.backendUnavailable.description')}</Text>
          </Stack>
          <Alert.Root status="error" variant="surface">
            <Alert.Indicator />
            <Alert.Content>
              <Alert.Title>{t('shell.backendUnavailable.connectionFailed')}</Alert.Title>
              <Alert.Description overflowWrap="anywhere">{message}</Alert.Description>
            </Alert.Content>
          </Alert.Root>
          <Button alignSelf="start" onClick={onRetry}>
            {t('shell.backendUnavailable.retry')}
          </Button>
          <Stack gap="3">
            <Heading size="md">{t('shell.backendUnavailable.draftsTitle')}</Heading>
            {draftError ? (
              <Alert.Root status="error" variant="surface">
                <Alert.Indicator />
                <Alert.Content>
                  <Alert.Description>{draftError}</Alert.Description>
                </Alert.Content>
              </Alert.Root>
            ) : null}
            {drafts === null ? (
              <HStack color="fg.muted">
                <Spinner size="xs" />
                <Text>{t('shell.backendUnavailable.loadingDrafts')}</Text>
              </HStack>
            ) : drafts === 'unavailable' ? (
              <Text color="fg.error">{t('shell.backendUnavailable.draftsUnavailable')}</Text>
            ) : drafts.length === 0 ? (
              <Text color="fg.muted">{t('shell.backendUnavailable.noDrafts')}</Text>
            ) : (
              drafts.map((draft) => (
                <RecoverableDraftRow
                  draft={draft}
                  isExporting={exportingDraftKey === `${draft.projectId}\u0000${draft.editorSessionId}`}
                  key={`${draft.projectId}:${draft.editorSessionId}`}
                  onExport={exportDraft}
                />
              ))
            )}
            {nextCursor ? (
              <Button alignSelf="start" loading={isLoadingMore} onClick={handleLoadMore} size="sm" variant="ghost">
                {t('shell.backendUnavailable.loadMore')}
              </Button>
            ) : null}
          </Stack>
          <QueueRecoveryNotice />
        </Stack>
      </Box>
    </Flex>
  );
};

const RecoverableDraftRow = ({
  draft,
  isExporting,
  onExport,
}: {
  draft: RecoverableProjectDraft;
  isExporting: boolean;
  onExport(draft: RecoverableProjectDraft): Promise<void>;
}) => {
  const { t } = useTranslation();
  const handleExport = useCallback(() => onExport(draft), [draft, onExport]);
  return (
    <HStack borderColor="border.subtle" borderRadius="md" borderWidth="1px" p="3">
      <Stack flex="1" gap="0" minW="0">
        <Text fontWeight="semibold" truncate>
          {draft.projectId}
        </Text>
        <Text color="fg.muted" fontSize="xs">
          {new Date(draft.updatedAt).toLocaleString()}
        </Text>
      </Stack>
      <Button loading={isExporting} onClick={handleExport} size="sm" variant="outline">
        {t('shell.backendUnavailable.exportDraft')}
      </Button>
    </HStack>
  );
};
