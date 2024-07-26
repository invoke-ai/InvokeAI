import type { TypesafeDraggableData, TypesafeDroppableData } from 'features/dnd/types';

export const isValidDrop = (overData?: TypesafeDroppableData | null, activeData?: TypesafeDraggableData | null) => {
  if (!overData || !activeData) {
    return false;
  }

  const { actionType } = overData;
  const { payloadType } = activeData;

  if (overData.id === activeData.id) {
    return false;
  }

  switch (actionType) {
    case 'SET_CURRENT_IMAGE':
      return payloadType === 'IMAGE_DTO';
    case 'SET_CA_IMAGE':
      return payloadType === 'IMAGE_DTO';
    case 'SET_IPA_IMAGE':
      return payloadType === 'IMAGE_DTO';
    case 'SET_RG_IP_ADAPTER_IMAGE':
      return payloadType === 'IMAGE_DTO';
    case 'ADD_LAYER_IMAGE':
      return payloadType === 'IMAGE_DTO';
    case 'SET_UPSCALE_INITIAL_IMAGE':
      return payloadType === 'IMAGE_DTO';
    case 'SET_NODES_IMAGE':
      return payloadType === 'IMAGE_DTO';
    case 'SELECT_FOR_COMPARE':
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
        const { imageDTO } = activeData.payload;
        const currentBoard = imageDTO.board_id ?? 'none';
        const destinationBoard = overData.context.boardId;

        return currentBoard !== destinationBoard;
      }

      if (payloadType === 'GALLERY_SELECTION') {
        // Assume all images are on the same board - this is true for the moment
        const currentBoard = activeData.payload.boardId;
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
        const { imageDTO } = activeData.payload;
        const currentBoard = imageDTO.board_id ?? 'none';

        return currentBoard !== 'none';
      }

      if (payloadType === 'GALLERY_SELECTION') {
        const currentBoard = activeData.payload.boardId;
        return currentBoard !== 'none';
      }

      return false;
    }
    default:
      return false;
  }
};
