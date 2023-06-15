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
import { ImageDTO } from 'services/api';
import { RootState } from 'app/store/store';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import { controlNetSelector } from 'features/controlNet/store/controlNetSlice';
import { nodesSelecter } from 'features/nodes/store/nodesSlice';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { some } from 'lodash-es';
import { imageAddedToBoard } from '../../services/thunks/board';

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
    const isInitialImage = generation.initialImage?.image_name === image_name;

    const isCanvasImage = canvas.layerState.objects.some(
      (obj) => obj.kind === 'image' && obj.image.image_name === image_name
    );

    const isNodesImage = nodes.nodes.some((node) => {
      return some(
        node.data.inputs,
        (input) =>
          input.type === 'image' && input.value?.image_name === image_name
      );
    });

    const isControlNetImage = some(
      controlNet.controlNets,
      (c) =>
        c.controlImage?.image_name === image_name ||
        c.processedControlImage?.image_name === image_name
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

type AddImageToBoardContextValue = {
  /**
   * Whether the move image dialog is open.
   */
  isOpen: boolean;
  /**
   * Closes the move image dialog.
   */
  onClose: () => void;
  /**
   * The image pending movement
   */
  image?: ImageDTO;
  onClickAddToBoard: (image: ImageDTO) => void;
  handleAddToBoard: (boardId: string) => void;
};

export const AddImageToBoardContext =
  createContext<AddImageToBoardContextValue>({
    isOpen: false,
    onClose: () => undefined,
    onClickAddToBoard: () => undefined,
    handleAddToBoard: () => undefined,
  });

type Props = PropsWithChildren;

export const AddImageToBoardContextProvider = (props: Props) => {
  const [imageToMove, setImageToMove] = useState<ImageDTO>();
  const dispatch = useAppDispatch();
  const { isOpen, onOpen, onClose } = useDisclosure();

  // Clean up after deleting or dismissing the modal
  const closeAndClearImageToDelete = useCallback(() => {
    setImageToMove(undefined);
    onClose();
  }, [onClose]);

  const onClickAddToBoard = useCallback(
    (image?: ImageDTO) => {
      if (!image) {
        return;
      }
      setImageToMove(image);
      onOpen();
    },
    [setImageToMove, onOpen]
  );

  const handleAddToBoard = useCallback(
    (boardId: string) => {
      if (imageToMove) {
        dispatch(
          imageAddedToBoard({
            requestBody: {
              board_id: boardId,
              image_name: imageToMove.image_name,
            },
          })
        );
        closeAndClearImageToDelete();
      }
    },
    [closeAndClearImageToDelete, dispatch, imageToMove]
  );

  return (
    <AddImageToBoardContext.Provider
      value={{
        isOpen,
        image: imageToMove,
        onClose: closeAndClearImageToDelete,
        onClickAddToBoard,
        handleAddToBoard,
      }}
    >
      {props.children}
    </AddImageToBoardContext.Provider>
  );
};
