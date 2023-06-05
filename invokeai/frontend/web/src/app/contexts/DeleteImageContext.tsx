import { useDisclosure } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { requestedImageDeletion } from 'features/gallery/store/actions';
import { systemSelector } from 'features/system/store/systemSelectors';
import { PropsWithChildren, createContext, useCallback, useState } from 'react';
import { ImageDTO } from 'services/api';

import { useImageUsage } from 'common/hooks/useImageUsage';
import { resetCanvas } from 'features/canvas/store/canvasSlice';
import { controlNetReset } from 'features/controlNet/store/controlNetSlice';
import { clearInitialImage } from 'features/parameters/store/generationSlice';
import { nodeEditorReset } from 'features/nodes/store/nodesSlice';

type DeleteImageContextValue = {
  /**
   * Whether the delete image dialog is open.
   */
  isOpen: boolean;
  /**
   * Closes the delete image dialog.
   */
  onClose: () => void;
  /**
   * Opens the delete image dialog and handles all deletion-related checks.
   */
  onDelete: (image?: ImageDTO) => void;
  /**
   * The image pending deletion
   */
  image?: ImageDTO;
  /**
   * Immediately deletes an image.
   *
   * You probably don't want to use this - use `onDelete` instead.
   */
  onImmediatelyDelete: () => void;
};

export const DeleteImageContext = createContext<DeleteImageContextValue>({
  isOpen: false,
  onClose: () => undefined,
  onImmediatelyDelete: () => undefined,
  onDelete: () => undefined,
});

const selector = createSelector(
  [systemSelector],
  (system) => {
    const { isProcessing, isConnected, shouldConfirmOnDelete } = system;

    return {
      canDeleteImage: isConnected && !isProcessing,
      shouldConfirmOnDelete,
    };
  },
  defaultSelectorOptions
);

type Props = PropsWithChildren;

export const DeleteImageContextProvider = (props: Props) => {
  const { canDeleteImage, shouldConfirmOnDelete } = useAppSelector(selector);
  const [imageToDelete, setImageToDelete] = useState<ImageDTO>();
  const dispatch = useAppDispatch();
  const { isOpen, onOpen, onClose } = useDisclosure();
  const imageUsage = useImageUsage(imageToDelete?.image_name);

  const handleActualDeletion = useCallback(
    (image: ImageDTO) => {
      dispatch(requestedImageDeletion(image));

      if (imageUsage.isCanvasImage) {
        dispatch(resetCanvas());
      }

      if (imageUsage.isControlNetImage) {
        dispatch(controlNetReset());
      }

      if (imageUsage.isInitialImage) {
        dispatch(clearInitialImage());
      }

      if (imageUsage.isControlNetImage) {
        dispatch(nodeEditorReset());
      }
    },
    [
      dispatch,
      imageUsage.isCanvasImage,
      imageUsage.isControlNetImage,
      imageUsage.isInitialImage,
    ]
  );

  const closeAndClearImageToDelete = useCallback(() => {
    setImageToDelete(undefined);
    onClose();
  }, [onClose]);

  const onImmediatelyDelete = useCallback(() => {
    if (canDeleteImage && imageToDelete) {
      handleActualDeletion(imageToDelete);
    }
    closeAndClearImageToDelete();
  }, [
    canDeleteImage,
    imageToDelete,
    closeAndClearImageToDelete,
    handleActualDeletion,
  ]);

  const handleGatedDeletion = useCallback(
    (image: ImageDTO) => {
      if (shouldConfirmOnDelete || imageUsage) {
        onOpen();
      } else {
        handleActualDeletion(image);
      }
    },
    [shouldConfirmOnDelete, imageUsage, onOpen, handleActualDeletion]
  );

  const onDelete = useCallback(
    (image?: ImageDTO) => {
      if (!image) {
        return;
      }
      setImageToDelete(image);
      handleGatedDeletion(image);
    },
    [handleGatedDeletion]
  );

  return (
    <DeleteImageContext.Provider
      value={{
        isOpen,
        image: imageToDelete,
        onClose: closeAndClearImageToDelete,
        onDelete,
        onImmediatelyDelete,
      }}
    >
      {props.children}
    </DeleteImageContext.Provider>
  );
};
