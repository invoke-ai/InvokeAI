import { useDisclosure } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { requestedImageDeletion } from 'features/gallery/store/actions';
import { systemSelector } from 'features/system/store/systemSelectors';
import { PropsWithChildren, createContext, useCallback, useState } from 'react';
import { ImageDTO } from 'services/api';

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
   * Immediately deletes an image.
   *
   * You probably don't want to use this - use `onDelete` instead.
   */
  onImmediatelyDelete: () => void;
  /**
   * Opens the delete image dialog and handles all deletion-related checks.
   */
  onDelete: (image?: ImageDTO) => void;
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
      isProcessing,
      isConnected,
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

  const closeAndClearImageToDelete = useCallback(() => {
    setImageToDelete(undefined);
    onClose();
  }, [onClose]);

  const onImmediatelyDelete = useCallback(() => {
    if (canDeleteImage && imageToDelete) {
      dispatch(requestedImageDeletion(imageToDelete));
    }
    closeAndClearImageToDelete();
  }, [canDeleteImage, imageToDelete, closeAndClearImageToDelete, dispatch]);

  const handleDelete = useCallback(
    (image: ImageDTO) => {
      if (shouldConfirmOnDelete) {
        onOpen();
      } else {
        dispatch(requestedImageDeletion(image));
      }
    },
    [shouldConfirmOnDelete, onOpen, dispatch]
  );

  const onDelete = useCallback(
    (image?: ImageDTO) => {
      if (!image) {
        return;
      }
      setImageToDelete(image);
      handleDelete(image);
    },
    [handleDelete]
  );

  return (
    <DeleteImageContext.Provider
      value={{
        isOpen,
        onClose: closeAndClearImageToDelete,
        onDelete,
        onImmediatelyDelete,
      }}
    >
      {props.children}
    </DeleteImageContext.Provider>
  );
};
