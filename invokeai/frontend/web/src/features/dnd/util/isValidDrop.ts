import type { TypesafeActive, TypesafeDroppableData } from 'features/dnd/types';

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
    case 'SET_CONTROL_ADAPTER_IMAGE':
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
        // Assume all images are on the same board - this is true for the moment
        const { imageDTOs } = active.data.current.payload;
        const currentBoard = imageDTOs[0]?.board_id ?? 'none';
        const destinationBoard = overData.context.boardId;

        return currentBoard !== destinationBoard;
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
        const currentBoard = imageDTO.board_id ?? 'none';

        return currentBoard !== 'none';
      }

      if (payloadType === 'IMAGE_DTOS') {
        // Assume all images are on the same board - this is true for the moment
        const { imageDTOs } = active.data.current.payload;
        const currentBoard = imageDTOs[0]?.board_id ?? 'none';

        return currentBoard !== 'none';
      }

      return false;
    }
    default:
      return false;
  }
};
