import { HStack, Icon, Stack, Text } from '@chakra-ui/react';
import {
  deleteManagedFile,
  MANAGED_FILE_ACCEPT,
  MANAGED_FILE_ACCEPTED_EXTENSIONS,
  managedFileQueryKey,
  managedFileQueryOptions,
  uploadManagedFile,
} from '@features/workflow/data/managedFiles';
import { useWorkflowUi } from '@features/workflow/ui/WorkflowUiContext';
import { getFieldRecordId } from '@features/workflow/utility';
import { formatBytes } from '@platform/i18n/languages';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button } from '@platform/ui/Button';
import { DropZone } from '@platform/ui/DropZone';
import { toaster } from '@platform/ui/toaster';
import { Tooltip } from '@platform/ui/Tooltip';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { FileTextIcon, RotateCcwIcon, UploadIcon, XIcon } from 'lucide-react';
import { useCallback, useRef, useState, type ChangeEvent, type DragEvent } from 'react';
import { useTranslation } from 'react-i18next';

import type { WorkflowFieldInputProps } from './WorkflowFieldInput';

const HIDDEN_FILE_INPUT_STYLE = { display: 'none' } as const;

const isAcceptedManagedFile = (file: File): boolean => {
  const name = file.name.toLowerCase();

  return MANAGED_FILE_ACCEPTED_EXTENSIONS.some((extension) => name.endsWith(extension));
};

/** A managed file reference: upload picks or drops one, while Clear removes only the field reference. */
export const FileFieldInput = ({
  id,
  invalid,
  nodeId,
  onChange,
  template,
  value,
  workflowId,
}: WorkflowFieldInputProps) => {
  const { t } = useTranslation();
  const { project } = useWorkflowUi();
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const uploadInFlightRef = useRef(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileId = getFieldRecordId(value, 'file_id');
  const { data: file, isError, isFetching, refetch } = useQuery(managedFileQueryOptions(fileId ?? ''));
  const metadataUnavailable = fileId !== null && isError;

  const onPickClick = useCallback(() => fileInputRef.current?.click(), []);
  const onClearClick = useCallback(() => onChange(undefined), [onChange]);
  const onRetryClick = useCallback(() => void refetch(), [refetch]);

  const startUpload = useCallback(
    async (picked: File) => {
      if (uploadInFlightRef.current) {
        return;
      }

      if (!isAcceptedManagedFile(picked)) {
        toaster.create({
          description: t('nodes.fileUploadRejectedDescription'),
          title: t('nodes.fileUploadRejected'),
          type: 'error',
        });

        return;
      }

      const owner = captureAccountScope();
      const documentId = workflowId ?? project.getSnapshot().projectGraph.id;

      uploadInFlightRef.current = true;
      setIsUploading(true);

      try {
        const uploaded = await uploadManagedFile(picked, owner.signal);
        // The document that owned this widget may have closed or switched while the upload ran; a late success must
        // not write into another document. The file was never adopted, so it is safe to remove as an orphan.
        const snapshot = project.getSnapshot();
        const document =
          workflowId !== undefined
            ? snapshot.workflows.find((entry) => entry.document.id === workflowId)?.document
            : snapshot.projectGraph;
        const nodeExists =
          nodeId === undefined || document?.nodes.some((candidate) => candidate.id === nodeId) === true;
        const isCurrent =
          isAccountScopeCurrent(owner) &&
          !owner.signal.aborted &&
          document !== undefined &&
          document.id === documentId &&
          nodeExists;

        if (!isCurrent) {
          void deleteManagedFile(uploaded.id).catch(() => undefined);

          return;
        }

        queryClient.setQueryData(managedFileQueryKey(uploaded.id), uploaded);
        onChange({ file_id: uploaded.id });
      } catch (error) {
        if (!isAccountScopeCurrent(owner) || owner.signal.aborted) {
          return;
        }

        toaster.create({
          description: getApiErrorMessage(error, t('nodes.fileUploadFailed')),
          title: t('nodes.fileUploadFailed'),
          type: 'error',
        });
      } finally {
        uploadInFlightRef.current = false;
        setIsUploading(false);
      }
    },
    [nodeId, onChange, project, queryClient, t, workflowId]
  );

  const onFileChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const picked = event.currentTarget.files?.[0];

      // Reset so picking the same file again re-fires the change event.
      event.currentTarget.value = '';

      if (picked) {
        void startUpload(picked);
      }
    },
    [startUpload]
  );

  const onDragEnter = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      if (isUploading) {
        return;
      }

      event.preventDefault();
      setIsDragOver(true);
    },
    [isUploading]
  );

  const onDragOver = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      if (isUploading) {
        return;
      }

      event.preventDefault();

      if (event.dataTransfer) {
        event.dataTransfer.dropEffect = 'copy';
      }
    },
    [isUploading]
  );

  const onDragLeave = useCallback((event: DragEvent<HTMLDivElement>) => {
    const nextTarget = event.relatedTarget;

    if (!(nextTarget instanceof Node) || !event.currentTarget.contains(nextTarget)) {
      setIsDragOver(false);
    }
  }, []);

  const onDrop = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      setIsDragOver(false);

      if (isUploading) {
        return;
      }

      const dropped = event.dataTransfer?.files?.[0];

      if (dropped) {
        void startUpload(dropped);
      }
    },
    [isUploading, startUpload]
  );

  return (
    <Stack aria-invalid={invalid || undefined} gap="1" w="full">
      <DropZone
        aria-label={template.title}
        className="nodrag"
        isDisabled={isUploading}
        isInvalid={invalid || metadataUnavailable}
        isOver={isDragOver}
        minH="14"
        px="2"
        py="2"
        role="group"
        w="full"
        onDragEnter={onDragEnter}
        onDragLeave={onDragLeave}
        onDragOver={onDragOver}
        onDrop={onDrop}
      >
        {metadataUnavailable ? (
          <Text color="fg.error" fontSize="xs" textAlign="center">
            {t('nodes.fileUnavailable')}
          </Text>
        ) : file ? (
          <HStack align="center" gap="2" minW="0">
            <Icon as={FileTextIcon} boxSize="5" color="fg.muted" flexShrink="0" />
            <Stack flex="1" gap="0" minW="0">
              <Tooltip content={file.name}>
                <Text fontSize="sm" fontWeight="semibold" truncate>
                  {file.name}
                </Text>
              </Tooltip>
              <Text color="fg.subtle" fontSize="xs">
                {formatBytes(file.sizeBytes)}
              </Text>
            </Stack>
          </HStack>
        ) : fileId ? (
          <Text color="fg.subtle" fontSize="xs">
            {t('common.loading')}
          </Text>
        ) : (
          <Stack align="center" gap="1">
            <Text color="fg" fontSize="sm" fontWeight="600">
              {t('nodes.fileUpload')}
            </Text>
            <Text color="fg.subtle" fontSize="xs" textAlign="center">
              {t('nodes.fileUploadDropHint')}
            </Text>
          </Stack>
        )}
      </DropZone>
      <HStack gap="1.5" w="full">
        <Button id={id} className="nodrag" disabled={isUploading} size="sm" variant="outline" onClick={onPickClick}>
          <Icon as={UploadIcon} boxSize="3" />
          {isUploading ? t('common.loading') : t('nodes.fileUpload')}
        </Button>
        {metadataUnavailable ? (
          <Button className="nodrag" disabled={isFetching} size="sm" variant="ghost" onClick={onRetryClick}>
            <Icon as={RotateCcwIcon} boxSize="3" />
            {t('common.retry')}
          </Button>
        ) : null}
        {fileId !== null ? (
          <Button className="nodrag" disabled={isUploading} size="sm" variant="ghost" onClick={onClearClick}>
            <Icon as={XIcon} boxSize="3" />
            {t('common.clear')}
          </Button>
        ) : null}
      </HStack>
      <input
        ref={fileInputRef}
        accept={MANAGED_FILE_ACCEPT}
        aria-label={t('nodes.fileUpload')}
        disabled={isUploading}
        style={HIDDEN_FILE_INPUT_STYLE}
        type="file"
        onChange={onFileChange}
      />
    </Stack>
  );
};
