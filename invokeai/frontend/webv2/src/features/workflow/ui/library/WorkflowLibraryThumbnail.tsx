import type { GalleryItem } from '@features/gallery';
import type { ProjectGraphState, XYPosition } from '@features/workflow/contracts';
import type { WorkflowLibraryListItem } from '@features/workflow/queries';
import type { WorkflowPreviewGraph } from '@features/workflow/ui/contracts';

import { Box, Flex, HStack, Icon, Image, Spinner, Stack, Text, type SystemStyleObject } from '@chakra-ui/react';
import { GalleryPickerPopover } from '@features/gallery/picker';
import { localizeForLoopValidationReason } from '@features/workflow/core/forLoops';
import {
  deleteLibraryWorkflowThumbnail,
  invalidateWorkflowLibraryCache,
  setLibraryWorkflowThumbnail,
} from '@features/workflow/queries';
import { useInvocationTemplatesSnapshot } from '@features/workflow/react';
import { useWorkflowNotifications } from '@features/workflow/ui/WorkflowUiContext';
import { useMountEffect } from '@platform/react/useMountEffect';
import {
  type AccountScope,
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { IconButton } from '@platform/ui/Button';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { CameraIcon, ImageOffIcon, Trash2Icon, UploadIcon } from 'lucide-react';
import { Suspense, useCallback, useRef, useState, type MouseEvent } from 'react';
import { useTranslation } from 'react-i18next';

import { buildLibraryGraphPreviewSource, DeferredGraphPreviewSnapshot, loadGraphPreview } from './libraryPreviewSource';
import { formatRelativeTime } from './relativeTime';

const THUMBNAIL_ASPECT_RATIO = 3 / 2;
const IMAGE_ONLY = ['image'] as const;
/** Busy controls stay focusable (`aria-disabled`), so the focus the picker returns to survives the action. */
const preventClick = (event: MouseEvent) => event.preventDefault();
const WELL_CSS: SystemStyleObject = {
  aspectRatio: THUMBNAIL_ASPECT_RATIO,
  bg: 'bg.muted',
  overflow: 'hidden',
  position: 'relative',
  rounded: 'md',
  w: 'full',
};

/** Key by workflow id: an action in flight for one workflow must not show as busy on the next selection. */
export const WorkflowLibraryThumbnail = ({
  item,
  workflowDocument,
}: {
  /** The template's parsed workflow once its row has loaded; the snapshot renders its graph. */
  workflowDocument: ProjectGraphState | null;
  item: WorkflowLibraryListItem;
}) => {
  const { t } = useTranslation();
  const notify = useWorkflowNotifications();
  // Keyed by URL: a replaced image arrives under a fresh URL and re-arms the thumbnail.
  const [failedThumbnailUrl, setFailedThumbnailUrl] = useState<string | null>(null);
  // Local rather than `useScopedAction`: the library ships in the editor boot graph, and a second importer of that
  // models-only hook splits it into its own startup chunk.
  const [isBusy, setIsBusy] = useState(false);
  const isBusyRef = useRef(false);
  const pendingSnapshotRef = useRef<{ reject: (error: unknown) => void } | null>(null);
  const isMountedRef = useRef(true);
  useMountEffect(() => {
    isMountedRef.current = true;
    return () => {
      // Leaving the workflow abandons its snapshot; nothing is uploaded and nothing is reported.
      isMountedRef.current = false;
      pendingSnapshotRef.current?.reject(new Error('The snapshot was abandoned.'));
    };
  });
  // One always-mounted live region: a region inserted together with its text is often not announced.
  const [announcement, setAnnouncement] = useState('');
  const chooseButtonRef = useRef<HTMLButtonElement | null>(null);
  const chooseIds = useTooltipTriggerIds();
  const run = useCallback(
    async (
      action: (owner: AccountScope) => Promise<void>,
      successAnnouncement: string,
      onError: (error: unknown) => void
    ) => {
      if (isBusyRef.current) {
        return;
      }

      const owner = captureAccountScope();

      isBusyRef.current = true;
      setIsBusy(true);
      setAnnouncement(t('workflowLibrary.thumbnailUpdating'));

      try {
        await action(owner);
        setAnnouncement(successAnnouncement);
      } catch (error) {
        if (isAccountScopeCurrent(owner)) {
          setAnnouncement('');
          onError(error);
        }
      } finally {
        isBusyRef.current = false;
        if (isAccountScopeCurrent(owner)) {
          setIsBusy(false);
        }
      }
    },
    [t]
  );
  // Bundled defaults are not the account's to change; the same rule gates rename and delete.
  const isEditable = item.category === 'user';
  const workflowId = item.workflow_id;

  const handleThumbnailError = useCallback(() => setFailedThumbnailUrl(item.thumbnail_url ?? null), [item]);

  const upload = useCallback(
    (getImage: (signal: AbortSignal) => Promise<Blob>) =>
      run(
        async (owner) => {
          const image = await getImage(owner.signal);

          assertAccountScopeCurrent(owner);
          await setLibraryWorkflowThumbnail(workflowId, image, owner.signal);
          assertAccountScopeCurrent(owner);
          invalidateWorkflowLibraryCache(workflowId);
        },
        t('workflowLibrary.thumbnailUpdated'),
        (error) => {
          if (isMountedRef.current) {
            notify.error(
              t('workflowLibrary.thumbnailUpdateFailed'),
              getApiErrorMessage(error, t('common.unknownError'))
            );
          }
        }
      ),
    [notify, run, t, workflowId]
  );

  const templatesSnapshot = useInvocationTemplatesSnapshot();
  const canSnapshot = workflowDocument !== null && templatesSnapshot.status === 'loaded';
  // Settled by the off-screen preview once it has captured the graph.
  const [snapshotRequest, setSnapshotRequest] = useState<{
    graph: WorkflowPreviewGraph;
    positionHints?: Record<string, XYPosition>;
    reject: (error: unknown) => void;
    resolve: (image: Blob) => void;
  } | null>(null);
  const takeSnapshot = useCallback(() => {
    if (!workflowDocument || templatesSnapshot.status !== 'loaded') {
      return;
    }

    void upload(async () => {
      const source = buildLibraryGraphPreviewSource(workflowDocument, templatesSnapshot.templates);

      if (!source.graph) {
        const [reason] = source.invalidReasons;
        throw new Error(reason ? localizeForLoopValidationReason(reason, t) : t('common.unknownError'));
      }

      const graph = source.graph;
      await loadGraphPreview();

      if (!isMountedRef.current) {
        throw new Error('The snapshot was abandoned.');
      }

      return new Promise<Blob>((resolve, reject) => {
        const settle = () => {
          pendingSnapshotRef.current = null;
          setSnapshotRequest(null);
        };

        pendingSnapshotRef.current = { reject };
        setSnapshotRequest({
          graph,
          positionHints: source.positionHints,
          reject: (error) => {
            settle();
            reject(error);
          },
          resolve: (image) => {
            settle();
            resolve(image);
          },
        });
      });
    });
  }, [t, templatesSnapshot, upload, workflowDocument]);

  // The server keeps a 256px copy, so the gallery's own thumbnail carries everything it will store.
  const handlePick = useCallback(
    (picked: GalleryItem) =>
      void upload(async (signal) => {
        const response = await fetch(picked.thumbnailUrl, { signal });

        if (!response.ok) {
          throw new Error(`${response.status} ${response.statusText}`);
        }

        return response.blob();
      }),
    [upload]
  );

  const handleRemove = useCallback(
    () =>
      void run(
        async (owner) => {
          await deleteLibraryWorkflowThumbnail(workflowId, owner.signal);
          assertAccountScopeCurrent(owner);
          // Remove unmounts once the refreshed record arrives; hand focus on before it takes focus down with it.
          if (document.activeElement?.closest('[data-thumbnail-remove]')) {
            chooseButtonRef.current?.focus();
          }
          invalidateWorkflowLibraryCache(workflowId);
        },
        t('workflowLibrary.thumbnailRemoved'),
        (error) =>
          notify.error(t('workflowLibrary.thumbnailRemoveFailed'), getApiErrorMessage(error, t('common.unknownError')))
      ),
    [notify, run, t, workflowId]
  );

  const showThumbnail = Boolean(item.thumbnail_url) && item.thumbnail_url !== failedThumbnailUrl;
  const lastRun = item.last_run_at ? formatRelativeTime(item.last_run_at, new Date()) : '';
  // A user template's thumbnail is always one someone set; only bundled defaults ship a sample output.
  const caption = lastRun
    ? t('workflowLibrary.lastRun', { when: lastRun })
    : showThumbnail && item.category === 'default'
      ? t('workflowLibrary.sampleOutput')
      : null;

  const wellContent = (
    <>
      {showThumbnail ? (
        <Image
          alt=""
          h="full"
          objectFit="cover"
          src={item.thumbnail_url ?? undefined}
          w="full"
          onError={handleThumbnailError}
        />
      ) : (
        <Flex align="center" direction="column" gap="1" h="full" justify="center" w="full">
          <Icon aria-hidden as={ImageOffIcon} boxSize="5" color="fg.subtle" opacity={0.6} />
          <Text color="fg.subtle" fontSize="xs">
            {t('workflowLibrary.notRunYet')}
          </Text>
        </Flex>
      )}
      {isBusy ? (
        <Flex align="center" bg="bg.muted/90" inset="0" justify="center" position="absolute">
          <Spinner size="lg" />
        </Flex>
      ) : null}
    </>
  );

  return (
    <Stack gap="1" minW="0">
      <Box aria-busy={isBusy || undefined} css={WELL_CSS}>
        {wellContent}
        {/* Bundled defaults keep their shipped image; the account's own templates choose theirs from the gallery. */}
        {isEditable ? (
          <HStack
            aria-label={t('workflowLibrary.thumbnail')}
            gap="1"
            position="absolute"
            right="1.5"
            role="group"
            top="1.5"
          >
            <GalleryPickerPopover
              accept={IMAGE_ONLY}
              ids={chooseIds}
              label={t('workflowLibrary.thumbnailChoose')}
              onPick={handlePick}
            >
              {/* Inside a dialog the tooltip must sit inside the popover trigger (see ProjectWorkflowsView). */}
              <Tooltip content={t('workflowLibrary.thumbnailChoose')} ids={chooseIds}>
                <IconButton
                  ref={chooseButtonRef}
                  aria-disabled={isBusy || undefined}
                  aria-label={t('workflowLibrary.thumbnailChoose')}
                  variant="outline"
                  onClickCapture={isBusy ? preventClick : undefined}
                >
                  <UploadIcon />
                </IconButton>
              </Tooltip>
            </GalleryPickerPopover>
            <Tooltip content={t('workflowLibrary.thumbnailSnapshot')}>
              <IconButton
                aria-disabled={isBusy || !canSnapshot || undefined}
                aria-label={t('workflowLibrary.thumbnailSnapshot')}
                variant="outline"
                onClick={isBusy || !canSnapshot ? undefined : takeSnapshot}
              >
                <CameraIcon />
              </IconButton>
            </Tooltip>
            {item.thumbnail_url ? (
              <Tooltip content={t('workflowLibrary.thumbnailRemove')}>
                <IconButton
                  aria-disabled={isBusy || undefined}
                  aria-label={t('workflowLibrary.thumbnailRemove')}
                  data-thumbnail-remove
                  variant="outline"
                  onClick={isBusy ? undefined : handleRemove}
                >
                  <Trash2Icon />
                </IconButton>
              </Tooltip>
            ) : null}
          </HStack>
        ) : null}
      </Box>
      {snapshotRequest ? (
        <Suspense fallback={null}>
          <DeferredGraphPreviewSnapshot
            graph={snapshotRequest.graph}
            positionHints={snapshotRequest.positionHints}
            onCapture={snapshotRequest.resolve}
            onError={snapshotRequest.reject}
          />
        </Suspense>
      ) : null}
      <Text role="status" srOnly>
        {announcement}
      </Text>
      {caption ? (
        <Text color="fg.subtle" fontSize="xs">
          {caption}
        </Text>
      ) : null}
    </Stack>
  );
};
