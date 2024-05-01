import type { TypesafeActive, TypesafeDroppableData } from 'features/dnd/types';

export const isValidDrop = (overData: TypesafeDroppableData | undefined, active: TypesafeActive | null) => {
  if (!overData || !active?.data.current) {
    return false;
  }

  const { actionType } = overData;
  const { payloadType } = active.data.current;

  if (overData.id === active.data.current.id) {
    return false;
  }

  switch (actionType) {
    case 'SET_CURRENT_IMAGE':
      return payloadType === 'IMAGE_DTO';
    case 'SET_INITIAL_IMAGE':
      return payloadType === 'IMAGE_DTO';
    case 'SET_CONTROL_ADAPTER_IMAGE':
      return payloadType === 'IMAGE_DTO';
    case 'SET_CA_LAYER_IMAGE':
      return payloadType === 'IMAGE_DTO';
    case 'SET_IPA_LAYER_IMAGE':
      return payloadType === 'IMAGE_DTO';
    case 'SET_RG_LAYER_IP_ADAPTER_IMAGE':
      return payloadType === 'IMAGE_DTO';
    case 'SET_CANVAS_INITIAL_IMAGE':
      return payloadType === 'IMAGE_DTO';
    case 'SET_NODES_IMAGE':
      return payloadType === 'IMAGE_DTO';
    case 'ADD_TO_BOARD': {
      // If the board is the same, don't allow the drop

      // Check the payload types
      const isPayloadValid = ['IMAGE_DTO', 'GALLERY_SELECTION'].includes(payloadType);
      if (!isPayloadValid) {
        return false;
      }

      // Check if the image's board is the board we are dragging onto
      if (payloadType === 'IMAGE_DTO') {
        const { imageDTO } = active.data.current.payload;
        const currentBoard = imageDTO.board_id ?? 'none';
        const destinationBoard = overData.context.boardId;

        return currentBoard !== destinationBoard;
      }

      if (payloadType === 'GALLERY_SELECTION') {
        // Assume all images are on the same board - this is true for the moment
        const currentBoard = active.data.current.payload.boardId;
        const destinationBoard = overData.context.boardId;
        return currentBoard !== destinationBoard;
      }

      return false;
    }
    case 'REMOVE_FROM_BOARD': {
      // If the board is the same, don't allow the drop

      // Check the payload types
      const isPayloadValid = ['IMAGE_DTO', 'GALLERY_SELECTION'].includes(payloadType);
      if (!isPayloadValid) {
        return false;
      }

      // Check if the image's board is the board we are dragging onto
      if (payloadType === 'IMAGE_DTO') {
        const { imageDTO } = active.data.current.payload;
        const currentBoard = imageDTO.board_id ?? 'none';

        return currentBoard !== 'none';
      }

      if (payloadType === 'GALLERY_SELECTION') {
        const currentBoard = active.data.current.payload.boardId;
        return currentBoard !== 'none';
      }

      return false;
    }
    default:
      return false;
  }
};
