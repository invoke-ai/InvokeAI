import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { dropTargetForExternal, monitorForExternal } from '@atlaskit/pragmatic-drag-and-drop/external/adapter';
import { containsFiles, getFiles } from '@atlaskit/pragmatic-drag-and-drop/external/file';
import { preventUnhandled } from '@atlaskit/pragmatic-drag-and-drop/prevent-unhandled';
import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Flex, Heading } from '@invoke-ai/ui-library';
import { logger } from 'app/logging/logger';
import { getStore } from 'app/store/nanostores/store';
import { useAppSelector } from 'app/store/storeHooks';
import { DndDropOverlay } from 'features/dnd/DndDropOverlay';
import type { DndTargetState } from 'features/dnd/types';
import { selectAutoAddBoardId } from 'features/gallery/store/gallerySelectors';
import { toast } from 'features/toast/toast';
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { uploadImages } from 'services/api/endpoints/images';
import { useBoardName } from 'services/api/hooks/useBoardName';
import type { UploadImageArg } from 'services/api/types';
import { z } from 'zod';

const ACCEPTED_IMAGE_TYPES = ['image/png', 'image/jpg', 'image/jpeg'];
const ACCEPTED_FILE_EXTENSIONS = ['.png', '.jpg', '.jpeg'];
const log = logger('paste');

// const MAX_IMAGE_SIZE = 4; //In MegaBytes
// const sizeInMB = (sizeInBytes: number, decimalsNum = 2) => {
//   const result = sizeInBytes / (1024 * 1024);
//   return +result.toFixed(decimalsNum);
// };

const zUploadFile = z
  .custom<File>()
  // .refine(
  //   (file) => {
  //     return sizeInMB(file.size) <= MAX_IMAGE_SIZE;
  //   },
  //   () => ({ message: `The maximum image size is ${MAX_IMAGE_SIZE}MB` })
  // )
  .refine(
    (file) => {
      return ACCEPTED_IMAGE_TYPES.includes(file.type);
    },
    (file) => ({ message: `File type ${file.type} is not supported` })
  )
  .refine(
    (file) => {
      return ACCEPTED_FILE_EXTENSIONS.some((ext) => file.name.endsWith(ext));
    },
    (file) => ({ message: `File extension .${file.name.split('.').at(-1)} is not supported` })
  );

const getFilesSchema = (max?: number) => {
  if (max === undefined) {
    return z.array(zUploadFile);
  }
  return z.array(zUploadFile).max(max);
};

const sx = {
  position: 'absolute',
  top: 2,
  right: 2,
  bottom: 2,
  left: 2,
  '&[data-dnd-state="idle"]': {
    pointerEvents: 'none',
  },
} satisfies SystemStyleObject;

const maxImageUploadCount = undefined;

export const FullscreenDropzone = memo(() => {
  const { t } = useTranslation();
  const ref = useRef<HTMLDivElement>(null);

  const [dndState, setDndState] = useState<DndTargetState>('idle');

  const validateAndUploadFiles = useCallback(
    (files: File[]) => {
      const { getState } = getStore();
      const uploadFilesSchema = getFilesSchema(maxImageUploadCount);
      const parseResult = uploadFilesSchema.safeParse(files);

      if (!parseResult.success) {
        const description =
          maxImageUploadCount === undefined
            ? t('toast.uploadFailedInvalidUploadDesc')
            : t('toast.uploadFailedInvalidUploadDesc_withCount', { count: maxImageUploadCount });

        toast({
          id: 'UPLOAD_FAILED',
          title: t('toast.uploadFailed'),
          description,
          status: 'error',
        });
        return;
      }
      const autoAddBoardId = selectAutoAddBoardId(getState());

      const uploadArgs: UploadImageArg[] = files.map((file, i) => ({
        file,
        image_category: 'user',
        is_intermediate: false,
        board_id: autoAddBoardId === 'none' ? undefined : autoAddBoardId,
        isFirstUploadOfBatch: i === 0,
      }));

      uploadImages(uploadArgs);
    },
    [maxImageUploadCount, t]
  );

  useEffect(() => {
    const element = ref.current;
    if (!element) {
      return;
    }

    return combine(
      dropTargetForExternal({
        element,
        canDrop: containsFiles,
        onDrop: ({ source }) => {
          const files = getFiles({ source });
          validateAndUploadFiles(files);
        },
        onDragEnter: () => {
          setDndState('over');
        },
        onDragLeave: () => {
          setDndState('idle');
        },
      }),
      monitorForExternal({
        canMonitor: containsFiles,
        onDragStart: () => {
          setDndState('potential');
          preventUnhandled.start();
        },
        onDrop: () => {
          setDndState('idle');
          preventUnhandled.stop();
        },
      })
    );
  }, [validateAndUploadFiles]);

  useEffect(() => {
    log.info('use effect');
    const controller = new AbortController();

    window.addEventListener(
      'paste',
      (e) => {
        log.info('event listener');
        log.info(JSON.stringify(e.clipboardData));
        if (!e.clipboardData?.files) {
          log.info('no files');
          return;
        }
        const files = Array.from(e.clipboardData.files);
        const { getState } = getStore();
        const uploadFilesSchema = getFilesSchema(undefined);
        const parseResult = uploadFilesSchema.safeParse(files);

        if (!parseResult.success) {
          // const description =
          //   maxImageUploadCount === undefined
          //     ? t('toast.uploadFailedInvalidUploadDesc')
          //     : t('toast.uploadFailedInvalidUploadDesc_withCount', { count: maxImageUploadCount });

          // toast({
          //   id: 'UPLOAD_FAILED',
          //   title: t('toast.uploadFailed'),
          //   description,
          //   status: 'error',
          // });
          log.info("couldn't parse");
          return;
        }
        const autoAddBoardId = selectAutoAddBoardId(getState());

        const uploadArgs: UploadImageArg[] = files.map((file, i) => ({
          file,
          image_category: 'user',
          is_intermediate: false,
          board_id: autoAddBoardId === 'none' ? undefined : autoAddBoardId,
          isFirstUploadOfBatch: i === 0,
        }));

        uploadImages(uploadArgs);
        // validateAndUploadFiles(files);
      },
      { signal: controller.signal }
    );

    return () => {
      log.info('aborted');
      controller.abort();
    };
  }, []);

  return (
    <Box ref={ref} data-dnd-state={dndState} sx={sx}>
      <DndDropOverlay dndState={dndState} label={<DropLabel />} />
    </Box>
  );
});

FullscreenDropzone.displayName = 'FullscreenDropzone';

const DropLabel = memo(() => {
  const { t } = useTranslation();
  const boardId = useAppSelector(selectAutoAddBoardId);
  const boardName = useBoardName(boardId);

  return (
    <Flex flexDir="column" gap={4} color="base.100" alignItems="center">
      <Heading size="lg">{t('gallery.dropToUpload')}</Heading>
      <Heading size="md">{t('toast.imagesWillBeAddedTo', { boardName })}</Heading>
    </Flex>
  );
});

DropLabel.displayName = 'DropLabel';
