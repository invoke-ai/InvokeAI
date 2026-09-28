import type { GalleryItem } from '@features/gallery';
import type { WorkflowLibraryListItem } from '@features/workflow/queries';

import { Box, Flex, HStack, Icon, Image, Spinner, Stack, Text } from '@chakra-ui/react';
import { GalleryPickerPopover, getGalleryUploadAccept } from '@features/gallery/picker';
import {
  deleteLibraryWorkflowThumbnail,
  invalidateWorkflowLibraryCache,
  setLibraryWorkflowThumbnail,
} from '@features/workflow/queries';
import { useWorkflowNotifications } from '@features/workflow/ui/WorkflowUiContext';
import {
  type AccountScope,
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, IconButton, Tooltip } from '@platform/ui';
import { ChevronDownIcon, ImageOffIcon, ImagePlusIcon, Trash2Icon, UploadIcon } from 'lucide-react';
import { useCallback, useRef, useState, type ChangeEvent, type MouseEvent } from 'react';
import { useTranslation } from 'react-i18next';

import { formatRelativeTime } from './relativeTime';

const THUMBNAIL_ASPECT_RATIO = 3 / 2;
const IMAGE_ONLY = ['image'] as const;
const UPLOAD_ACCEPT = getGalleryUploadAccept(IMAGE_ONLY);
/** Busy controls stay focusable (`aria-disabled`), so focus the picker or file dialog returns to survives the action. */
const preventClick = (event: MouseEvent) => event.preventDefault();

/** Key by workflow id: an action in flight for one workflow must not show as busy on the next selection. */
export const WorkflowLibraryThumbnail = ({ item }: { item: WorkflowLibraryListItem }) => {
  const { t } = useTranslation();
  const notify = useWorkflowNotifications();
  const inputRef = useRef<HTMLInputElement | null>(null);
  // Keyed by URL: a replaced image arrives under a fresh URL and re-arms the thumbnail.
  const [failedThumbnailUrl, setFailedThumbnailUrl] = useState<string | null>(null);
  // Local rather than `useScopedAction`: the library ships in the editor boot graph, and a second importer of that
  // models-only hook splits it into its own startup chunk.
  const [isBusy, setIsBusy] = useState(false);
  const isBusyRef = useRef(false);
  // One always-mounted live region: a region inserted together with its text is often not announced.
  const [announcement, setAnnouncement] = useState('');
  const uploadButtonRef = useRef<HTMLButtonElement | null>(null);
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
        (error) =>
          notify.error(t('workflowLibrary.thumbnailUpdateFailed'), getApiErrorMessage(error, t('common.unknownError')))
      ),
    [notify, run, t, workflowId]
  );

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

  const openFilePicker = useCallback(() => inputRef.current?.click(), []);
  const handleFileChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.currentTarget.files?.[0];

      event.currentTarget.value = '';

      if (!file) {
        return;
      }

      // `accept` is advisory; the server takes only parts typed as images.
      if (!file.type.startsWith('image/')) {
        notify.error(t('workflowLibrary.thumbnailUpdateFailed'), t('workflowLibrary.thumbnailNotAnImage'));
        return;
      }

      void upload(() => Promise.resolve(file));
    },
    [notify, t, upload]
  );

  const handleRemove = useCallback(
    () =>
      void run(
        async (owner) => {
          await deleteLibraryWorkflowThumbnail(workflowId, owner.signal);
          assertAccountScopeCurrent(owner);
          // Remove unmounts once the refreshed record arrives; hand focus on before it takes focus down with it.
          if (document.activeElement?.closest('[data-thumbnail-remove]')) {
            uploadButtonRef.current?.focus();
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

  return (
    <Stack gap="1" minW="0">
      <Box
        aria-busy={isBusy || undefined}
        aspectRatio={THUMBNAIL_ASPECT_RATIO}
        bg="bg.muted"
        overflow="hidden"
        position="relative"
        rounded="md"
        w="full"
      >
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
            <Text color="fg.subtle" fontSize="2xs">
              {t('workflowLibrary.notRunYet')}
            </Text>
          </Flex>
        )}
        {isBusy ? (
          <Flex align="center" bg="bg.muted/90" inset="0" justify="center" position="absolute">
            <Spinner size="sm" />
          </Flex>
        ) : null}
      </Box>
      <Text role="status" srOnly>
        {announcement}
      </Text>
      {caption ? (
        <Text color="fg.subtle" fontSize="2xs">
          {caption}
        </Text>
      ) : null}
      {isEditable ? (
        <HStack aria-label={t('workflowLibrary.thumbnail')} flexWrap="wrap" gap="0.5" role="group">
          <GalleryPickerPopover
            accept={IMAGE_ONLY}
            label={t('workflowLibrary.thumbnailChooseTitle')}
            onPick={handlePick}
          >
            <Button
              aria-disabled={isBusy || undefined}
              size="xs"
              variant="ghost"
              onClickCapture={isBusy ? preventClick : undefined}
            >
              <Icon as={ImagePlusIcon} boxSize="3.5" />
              {t('workflowLibrary.thumbnailFromGallery')}
              <Icon as={ChevronDownIcon} boxSize="3" color="fg.subtle" />
            </Button>
          </GalleryPickerPopover>
          <Button
            ref={uploadButtonRef}
            aria-disabled={isBusy || undefined}
            size="xs"
            variant="ghost"
            onClick={isBusy ? undefined : openFilePicker}
          >
            <Icon as={UploadIcon} boxSize="3" />
            {t('workflowLibrary.thumbnailUpload')}
          </Button>
          {item.thumbnail_url ? (
            <Tooltip content={t('workflowLibrary.thumbnailRemove')}>
              <IconButton
                aria-disabled={isBusy || undefined}
                aria-label={t('workflowLibrary.thumbnailRemove')}
                data-thumbnail-remove
                ms="auto"
                size="xs"
                variant="ghost"
                onClick={isBusy ? undefined : handleRemove}
              >
                <Trash2Icon />
              </IconButton>
            </Tooltip>
          ) : null}
          <input ref={inputRef} accept={UPLOAD_ACCEPT} hidden type="file" onChange={handleFileChange} />
        </HStack>
      ) : null}
    </Stack>
  );
};
