import { TypesafeActive, TypesafeDroppableData } from '../types';

export const isValidDrop = (
  overData: TypesafeDroppableData | undefined,
  active: TypesafeActive | null
) => {
  if (!overData || !active?.data.current) {
    return false;
  }

  const { actionType } = overData;
  const { payloadType } = active.data.current;

  if (overData.id === active.data.current.id) {
    return false;
  }

  switch (actionType) {
    case 'ADD_FIELD_TO_LINEAR':
      return payloadType === 'NODE_FIELD';
    case 'SET_CURRENT_IMAGE':
      return payloadType === 'IMAGE_DTO';
    case 'SET_INITIAL_IMAGE':
      return payloadType === 'IMAGE_DTO';
    case 'SET_CONTROLNET_IMAGE':
      return payloadType === 'IMAGE_DTO';
    case 'SET_CANVAS_INITIAL_IMAGE':
      return payloadType === 'IMAGE_DTO';
    case 'SET_NODES_IMAGE':
      return payloadType === 'IMAGE_DTO';
    case 'SET_MULTI_NODES_IMAGE':
      return payloadType === 'IMAGE_DTO' || 'IMAGE_DTOS';
    case 'ADD_TO_BATCH':
      return payloadType === 'IMAGE_DTO' || 'IMAGE_DTOS';
    case 'ADD_TO_BOARD': {
      // If the board is the same, don't allow the drop

      // Check the payload types
      const isPayloadValid = payloadType === 'IMAGE_DTO' || 'IMAGE_DTOS';
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

      if (payloadType === 'IMAGE_DTOS') {
        // TODO (multi-select)
        return true;
      }

      return false;
    }
    case 'REMOVE_FROM_BOARD': {
      // If the board is the same, don't allow the drop

      // Check the payload types
      const isPayloadValid = payloadType === 'IMAGE_DTO' || 'IMAGE_DTOS';
      if (!isPayloadValid) {
        return false;
      }

      // Check if the image's board is the board we are dragging onto
      if (payloadType === 'IMAGE_DTO') {
        const { imageDTO } = active.data.current.payload;
        const currentBoard = imageDTO.board_id;

        return currentBoard !== 'none';
      }

      if (payloadType === 'IMAGE_DTOS') {
        // TODO (multi-select)
        return true;
      }

      return false;
    }
    default:
      return false;
  }
};
