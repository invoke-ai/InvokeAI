import { useDisclosure } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { requestedImageDeletion } from 'features/gallery/store/actions';
import { systemSelector } from 'features/system/store/systemSelectors';
import {
  PropsWithChildren,
  createContext,
  useCallback,
  useEffect,
  useState,
} from 'react';
import { ImageDTO } from 'services/api/types';
import { RootState } from 'app/store/store';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import { controlNetSelector } from 'features/controlNet/store/controlNetSlice';
import { nodesSelecter } from 'features/nodes/store/nodesSlice';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { some } from 'lodash-es';

export type ImageUsage = {
  isInitialImage: boolean;
  isCanvasImage: boolean;
  isNodesImage: boolean;
  isControlNetImage: boolean;
};

export const selectImageUsage = createSelector(
  [
    generationSelector,
    canvasSelector,
    nodesSelecter,
    controlNetSelector,
    (state: RootState, image_name?: string) => image_name,
  ],
  (generation, canvas, nodes, controlNet, image_name) => {
    const isInitialImage = generation.initialImage?.imageName === image_name;

    const isCanvasImage = canvas.layerState.objects.some(
      (obj) => obj.kind === 'image' && obj.imageName === image_name
    );

    const isNodesImage = nodes.nodes.some((node) => {
      return some(
        node.data.inputs,
        (input) => input.type === 'image' && input.value === image_name
      );
    });

    const isControlNetImage = some(
      controlNet.controlNets,
      (c) =>
        c.controlImage === image_name || c.processedControlImage === image_name
    );

    const imageUsage: ImageUsage = {
      isInitialImage,
      isCanvasImage,
      isNodesImage,
      isControlNetImage,
    };

    return imageUsage;
  },
  defaultSelectorOptions
);

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
   * The features in which this image is used
   */
  imageUsage?: ImageUsage;
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

  // Check where the image to be deleted is used (eg init image, controlnet, etc.)
  const imageUsage = useAppSelector((state) =>
    selectImageUsage(state, imageToDelete?.image_name)
  );

  // Clean up after deleting or dismissing the modal
  const closeAndClearImageToDelete = useCallback(() => {
    setImageToDelete(undefined);
    onClose();
  }, [onClose]);

  // Dispatch the actual deletion action, to be handled by listener middleware
  const handleActualDeletion = useCallback(
    (image: ImageDTO) => {
      dispatch(requestedImageDeletion({ image, imageUsage }));
      closeAndClearImageToDelete();
    },
    [closeAndClearImageToDelete, dispatch, imageUsage]
  );

  // This is intended to be called by the delete button in the dialog
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
      if (shouldConfirmOnDelete || some(imageUsage)) {
        // If we should confirm on delete, or if the image is in use, open the dialog
        onOpen();
      } else {
        handleActualDeletion(image);
      }
    },
    [imageUsage, shouldConfirmOnDelete, onOpen, handleActualDeletion]
  );

  // Consumers of the context call this to delete an image
  const onDelete = useCallback((image?: ImageDTO) => {
    if (!image) {
      return;
    }
    // Set the image to delete, then let the effect call the actual deletion
    setImageToDelete(image);
  }, []);

  useEffect(() => {
    // We need to use an effect here to trigger the image usage selector, else we get a stale value
    if (imageToDelete) {
      handleGatedDeletion(imageToDelete);
    }
  }, [handleGatedDeletion, imageToDelete]);

  return (
    <DeleteImageContext.Provider
      value={{
        isOpen,
        image: imageToDelete,
        onClose: closeAndClearImageToDelete,
        onDelete,
        onImmediatelyDelete,
        imageUsage,
      }}
    >
      {props.children}
    </DeleteImageContext.Provider>
  );
};
