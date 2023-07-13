import { useDisclosure } from '@chakra-ui/react';
import { PropsWithChildren, createContext, useCallback, useState } from 'react';
import { BoardDTO } from 'services/api/types';
import { useDeleteBoardMutation } from '../../services/api/endpoints/boards';
import { defaultSelectorOptions } from '../store/util/defaultMemoizeOptions';
import { createSelector } from '@reduxjs/toolkit';
import { some } from 'lodash-es';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import { controlNetSelector } from 'features/controlNet/store/controlNetSlice';
import { selectImagesById } from 'features/gallery/store/gallerySlice';
import { nodesSelector } from 'features/nodes/store/nodesSlice';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { RootState } from '../store/store';
import { useAppDispatch, useAppSelector } from '../store/storeHooks';
import { ImageUsage } from './DeleteImageContext';
import { requestedBoardImagesDeletion } from 'features/gallery/store/actions';

export const selectBoardImagesUsage = createSelector(
  [
    (state: RootState) => state,
    generationSelector,
    canvasSelector,
    nodesSelector,
    controlNetSelector,
    (state: RootState, board_id?: string) => board_id,
  ],
  (state, generation, canvas, nodes, controlNet, board_id) => {
    const initialImage = generation.initialImage
      ? selectImagesById(state, generation.initialImage.imageName)
      : undefined;
    const isInitialImage = initialImage?.board_id === board_id;

    const isCanvasImage = canvas.layerState.objects.some((obj) => {
      if (obj.kind === 'image') {
        const image = selectImagesById(state, obj.imageName);
        return image?.board_id === board_id;
      }
      return false;
    });

    const isNodesImage = nodes.nodes.some((node) => {
      return some(node.data.inputs, (input) => {
        if (input.type === 'image' && input.value) {
          const image = selectImagesById(state, input.value.image_name);
          return image?.board_id === board_id;
        }
        return false;
      });
    });

    const isControlNetImage = some(controlNet.controlNets, (c) => {
      const controlImage = c.controlImage
        ? selectImagesById(state, c.controlImage)
        : undefined;
      const processedControlImage = c.processedControlImage
        ? selectImagesById(state, c.processedControlImage)
        : undefined;
      return (
        controlImage?.board_id === board_id ||
        processedControlImage?.board_id === board_id
      );
    });

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

type DeleteBoardImagesContextValue = {
  /**
   * Whether the move image dialog is open.
   */
  isOpen: boolean;
  /**
   * Closes the move image dialog.
   */
  onClose: () => void;
  imagesUsage?: ImageUsage;
  board?: BoardDTO;
  onClickDeleteBoardImages: (board: BoardDTO) => void;
  handleDeleteBoardImages: (boardId: string) => void;
  handleDeleteBoardOnly: (boardId: string) => void;
};

export const DeleteBoardImagesContext =
  createContext<DeleteBoardImagesContextValue>({
    isOpen: false,
    onClose: () => undefined,
    onClickDeleteBoardImages: () => undefined,
    handleDeleteBoardImages: () => undefined,
    handleDeleteBoardOnly: () => undefined,
  });

type Props = PropsWithChildren;

export const DeleteBoardImagesContextProvider = (props: Props) => {
  const [boardToDelete, setBoardToDelete] = useState<BoardDTO>();
  const { isOpen, onOpen, onClose } = useDisclosure();
  const dispatch = useAppDispatch();

  // Check where the board images to be deleted are used (eg init image, controlnet, etc.)
  const imagesUsage = useAppSelector((state) =>
    selectBoardImagesUsage(state, boardToDelete?.board_id)
  );

  const [deleteBoard] = useDeleteBoardMutation();

  // Clean up after deleting or dismissing the modal
  const closeAndClearBoardToDelete = useCallback(() => {
    setBoardToDelete(undefined);
    onClose();
  }, [onClose]);

  const onClickDeleteBoardImages = useCallback(
    (board?: BoardDTO) => {
      console.log({ board });
      if (!board) {
        return;
      }
      setBoardToDelete(board);
      onOpen();
    },
    [setBoardToDelete, onOpen]
  );

  const handleDeleteBoardImages = useCallback(
    (boardId: string) => {
      if (boardToDelete) {
        dispatch(
          requestedBoardImagesDeletion({ board: boardToDelete, imagesUsage })
        );
        closeAndClearBoardToDelete();
      }
    },
    [dispatch, closeAndClearBoardToDelete, boardToDelete, imagesUsage]
  );

  const handleDeleteBoardOnly = useCallback(
    (boardId: string) => {
      if (boardToDelete) {
        deleteBoard(boardId);
        closeAndClearBoardToDelete();
      }
    },
    [deleteBoard, closeAndClearBoardToDelete, boardToDelete]
  );

  return (
    <DeleteBoardImagesContext.Provider
      value={{
        isOpen,
        board: boardToDelete,
        onClose: closeAndClearBoardToDelete,
        onClickDeleteBoardImages,
        handleDeleteBoardImages,
        handleDeleteBoardOnly,
        imagesUsage,
      }}
    >
      {props.children}
    </DeleteBoardImagesContext.Provider>
  );
};
