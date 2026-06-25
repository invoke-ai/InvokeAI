import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Button, Flex, Icon, IconButton, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch } from 'app/store/storeHooks';
import { fieldFileValueChanged } from 'features/nodes/store/nodesSlice';
import { NO_DRAG_CLASS } from 'features/nodes/types/constants';
import type { FileFieldInputInstance, FileFieldInputTemplate } from 'features/nodes/types/field';
import { toast } from 'features/toast/toast';
import { filesize } from 'filesize';
import { memo, type MouseEvent, useCallback, useEffect } from 'react';
import type { Accept, FileRejection } from 'react-dropzone';
import { useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import { PiFileTextBold, PiUploadBold, PiXBold } from 'react-icons/pi';
import { useGetFileDTOQuery, useUploadFileMutation } from 'services/api/endpoints/files';
import type { FileDTO } from 'services/api/types';
import { $isConnected } from 'services/events/stores';

import type { FieldComponentProps } from './types';

const addUpperCaseReducer = (acc: string[], ext: string) => {
  acc.push(ext);
  acc.push(ext.toUpperCase());
  return acc;
};

const textFileExtensions = ['.csv', '.json', '.md', '.markdown', '.txt', '.yaml', '.yml'].reduce(
  addUpperCaseReducer,
  [] as string[]
);

const fileUploadAccept: Accept = {
  'application/pdf': ['.pdf'].reduce(addUpperCaseReducer, [] as string[]),
  'application/json': ['.json'].reduce(addUpperCaseReducer, [] as string[]),
  'application/yaml': ['.yaml', '.yml'].reduce(addUpperCaseReducer, [] as string[]),
  'application/x-yaml': ['.yaml', '.yml'].reduce(addUpperCaseReducer, [] as string[]),
  'text/csv': ['.csv'].reduce(addUpperCaseReducer, [] as string[]),
  'text/markdown': ['.md', '.markdown'].reduce(addUpperCaseReducer, [] as string[]),
  'text/plain': textFileExtensions,
  'text/x-markdown': ['.md', '.markdown'].reduce(addUpperCaseReducer, [] as string[]),
  'text/x-yaml': ['.yaml', '.yml'].reduce(addUpperCaseReducer, [] as string[]),
  'text/yaml': ['.yaml', '.yml'].reduce(addUpperCaseReducer, [] as string[]),
};

const sx = {
  '&[data-error=true]': {
    borderColor: 'error.500',
    borderStyle: 'solid',
  },
  '&[data-active=true]': {
    borderColor: 'invokeBlue.500',
    borderStyle: 'solid',
  },
} satisfies SystemStyleObject;

const FileFieldInputComponent = (props: FieldComponentProps<FileFieldInputInstance, FileFieldInputTemplate>) => {
  const { nodeId, field, fieldTemplate } = props;
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isConnected = useStore($isConnected);
  const [uploadFile, uploadRequest] = useUploadFileMutation();
  const { currentData: fileDTO, isError } = useGetFileDTOQuery(field.value?.file_id ?? skipToken);

  const setValue = useCallback(
    (value: FileDTO | undefined) => {
      dispatch(
        fieldFileValueChanged({
          nodeId,
          fieldName: field.name,
          value: value ? { file_id: value.file_id } : undefined,
        })
      );
    },
    [dispatch, field.name, nodeId]
  );

  const handleReset = useCallback(() => {
    setValue(undefined);
  }, [setValue]);

  const handleResetClick = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      event.stopPropagation();
      handleReset();
    },
    [handleReset]
  );

  useEffect(() => {
    if (isConnected && isError) {
      handleReset();
    }
  }, [handleReset, isConnected, isError]);

  const onDropAccepted = useCallback(
    async (files: File[]) => {
      const file = files[0];
      if (!file) {
        return;
      }
      try {
        const uploadedFileDTO = await uploadFile({ file }).unwrap();
        setValue(uploadedFileDTO);
      } catch {
        toast({
          id: 'FILE_UPLOAD_FAILED',
          title: t('toast.fileUploadFailed'),
          status: 'error',
        });
      }
    },
    [setValue, t, uploadFile]
  );

  const onDropRejected = useCallback(
    (fileRejections: FileRejection[]) => {
      if (fileRejections.length === 0) {
        return;
      }
      toast({
        id: 'FILE_UPLOAD_REJECTED',
        title: t('toast.uploadFailed'),
        description: t('toast.fileUploadFailedInvalidUploadDesc'),
        status: 'error',
      });
    },
    [t]
  );

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    accept: fileUploadAccept,
    multiple: false,
    noClick: true,
    noKeyboard: true,
    onDropAccepted,
    onDropRejected,
  });

  const handleOpenClick = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      event.stopPropagation();
      open();
    },
    [open]
  );

  return (
    <Flex
      {...getRootProps()}
      className={NO_DRAG_CLASS}
      w="full"
      minH={14}
      alignItems="center"
      borderRadius="base"
      borderWidth={1}
      borderStyle="dashed"
      borderColor="base.600"
      bg="base.850"
      px={2}
      py={1}
      gap={2}
      sx={sx}
      data-active={isDragActive}
      data-error={fieldTemplate.required && !field.value}
    >
      <input {...getInputProps()} />
      {!fileDTO && (
        <Button
          aria-label={t('accessibility.uploadFile')}
          variant="ghost"
          size="sm"
          leftIcon={<PiUploadBold />}
          isLoading={uploadRequest.isLoading}
          onClick={handleOpenClick}
          w="full"
          minW={0}
        >
          {t('gallery.dropOrUpload')}
        </Button>
      )}
      {fileDTO && (
        <>
          <Icon as={PiFileTextBold} boxSize={5} color="base.300" flexShrink={0} />
          <Flex direction="column" minW={0} flex={1} gap={0.5}>
            <Text fontSize="sm" fontWeight="semibold" noOfLines={1} title={fileDTO.file_name}>
              {fileDTO.file_name}
            </Text>
            <Text fontSize="xs" color="base.400" noOfLines={1}>
              {filesize(fileDTO.size_bytes)}
            </Text>
          </Flex>
          <IconButton
            aria-label={t('common.reset')}
            icon={<PiXBold />}
            variant="ghost"
            size="sm"
            onClick={handleResetClick}
            flexShrink={0}
          />
        </>
      )}
    </Flex>
  );
};

export default memo(FileFieldInputComponent);
