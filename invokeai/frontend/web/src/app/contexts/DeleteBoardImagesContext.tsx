import { useDisclosure } from '@chakra-ui/react';
import { PropsWithChildren, createContext, useCallback, useState } from 'react';
import { BoardDTO } from 'services/api/types';
import { useDeleteBoardMutation } from '../../services/api/endpoints/boards';

export type ImageUsage = {
  isInitialImage: boolean;
  isCanvasImage: boolean;
  isNodesImage: boolean;
  isControlNetImage: boolean;
};

type DeleteBoardImagesContextValue = {
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

  const [deleteBoardAndImages] = useDeleteBoardAndImagesMutation();
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
        deleteBoardAndImages(boardId);
        closeAndClearBoardToDelete();
      }
    },
    [deleteBoardAndImages, closeAndClearBoardToDelete, boardToDelete]
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
      }}
    >
      {props.children}
    </DeleteBoardImagesContext.Provider>
  );
};
