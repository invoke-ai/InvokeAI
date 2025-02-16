import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { dropTargetForExternal, monitorForExternal } from '@atlaskit/pragmatic-drag-and-drop/external/adapter';
import { containsFiles, getFiles } from '@atlaskit/pragmatic-drag-and-drop/external/file';
import { preventUnhandled } from '@atlaskit/pragmatic-drag-and-drop/prevent-unhandled';
import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Flex, Heading } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { getStore } from 'app/store/nanostores/store';
import { useAppSelector } from 'app/store/storeHooks';
import { setFileToPaste } from 'features/controlLayers/components/CanvasPasteModal';
import { DndDropOverlay } from 'features/dnd/DndDropOverlay';
import type { DndTargetState } from 'features/dnd/types';
import { $imageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { selectAutoAddBoardId } from 'features/gallery/store/gallerySelectors';
import { selectMaxImageUploadCount } from 'features/system/store/configSlice';
import { toast } from 'features/toast/toast';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { uploadImages } from 'services/api/endpoints/images';
import { useBoardName } from 'services/api/hooks/useBoardName';
import type { UploadImageArg } from 'services/api/types';
import { z } from 'zod';

const ACCEPTED_IMAGE_TYPES = ['image/png', 'image/jpg', 'image/jpeg'];
const ACCEPTED_FILE_EXTENSIONS = ['.png', '.jpg', '.jpeg'];

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

export const FullscreenDropzone = memo(() => {
  const { t } = useTranslation();
  const ref = useRef<HTMLDivElement>(null);
  const maxImageUploadCount = useAppSelector(selectMaxImageUploadCount);
  const [dndState, setDndState] = useState<DndTargetState>('idle');
  const activeTab = useAppSelector(selectActiveTab);
  const isImageViewerOpen = useStore($imageViewer);

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

      // While on the canvas tab and when pasting a single image, canvas may want to create a new layer. Let it handle
      // the paste event.
      const [firstImageFile] = files;
      if (!isImageViewerOpen && activeTab === 'canvas' && files.length === 1 && firstImageFile) {
        setFileToPaste(firstImageFile);
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
    [activeTab, isImageViewerOpen, maxImageUploadCount, t]
  );

  const onPaste = useCallback(
    (e: ClipboardEvent) => {
      if (!e.clipboardData?.files) {
        return;
      }
      const files = Array.from(e.clipboardData.files);
      validateAndUploadFiles(files);
    },
    [validateAndUploadFiles]
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
    window.addEventListener('paste', onPaste);

    return () => {
      window.removeEventListener('paste', onPaste);
    };
  }, [onPaste]);

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
