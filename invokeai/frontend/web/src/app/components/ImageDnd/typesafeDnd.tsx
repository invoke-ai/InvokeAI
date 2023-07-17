// type-safe dnd from https://github.com/clauderic/dnd-kit/issues/935
import {
  Active,
  Collision,
  DndContextProps,
  DndContext as OriginalDndContext,
  Over,
  Translate,
  UseDraggableArguments,
  UseDroppableArguments,
  useDraggable as useOriginalDraggable,
  useDroppable as useOriginalDroppable,
} from '@dnd-kit/core';
import { ImageDTO } from 'services/api/types';

type BaseDropData = {
  id: string;
};

export type CurrentImageDropData = BaseDropData & {
  actionType: 'SET_CURRENT_IMAGE';
};

export type InitialImageDropData = BaseDropData & {
  actionType: 'SET_INITIAL_IMAGE';
};

export type ControlNetDropData = BaseDropData & {
  actionType: 'SET_CONTROLNET_IMAGE';
  context: {
    controlNetId: string;
  };
};

export type CanvasInitialImageDropData = BaseDropData & {
  actionType: 'SET_CANVAS_INITIAL_IMAGE';
};

export type NodesImageDropData = BaseDropData & {
  actionType: 'SET_NODES_IMAGE';
  context: {
    nodeId: string;
    fieldName: string;
  };
};

export type NodesMultiImageDropData = BaseDropData & {
  actionType: 'SET_MULTI_NODES_IMAGE';
  context: { nodeId: string; fieldName: string };
};

export type AddToBatchDropData = BaseDropData & {
  actionType: 'ADD_TO_BATCH';
};

export type MoveBoardDropData = BaseDropData & {
  actionType: 'MOVE_BOARD';
  context: { boardId: string | null };
};

export type TypesafeDroppableData =
  | CurrentImageDropData
  | InitialImageDropData
  | ControlNetDropData
  | CanvasInitialImageDropData
  | NodesImageDropData
  | AddToBatchDropData
  | NodesMultiImageDropData
  | MoveBoardDropData;

type BaseDragData = {
  id: string;
};

export type ImageDraggableData = BaseDragData & {
  payloadType: 'IMAGE_DTO';
  payload: { imageDTO: ImageDTO };
};

export type ImageNamesDraggableData = BaseDragData & {
  payloadType: 'IMAGE_NAMES';
  payload: { image_names: string[] };
};

export type TypesafeDraggableData =
  | ImageDraggableData
  | ImageNamesDraggableData;

interface UseDroppableTypesafeArguments
  extends Omit<UseDroppableArguments, 'data'> {
  data?: TypesafeDroppableData;
}

type UseDroppableTypesafeReturnValue = Omit<
  ReturnType<typeof useOriginalDroppable>,
  'active' | 'over'
> & {
  active: TypesafeActive | null;
  over: TypesafeOver | null;
};

export function useDroppable(props: UseDroppableTypesafeArguments) {
  return useOriginalDroppable(props) as UseDroppableTypesafeReturnValue;
}

interface UseDraggableTypesafeArguments
  extends Omit<UseDraggableArguments, 'data'> {
  data?: TypesafeDraggableData;
}

type UseDraggableTypesafeReturnValue = Omit<
  ReturnType<typeof useOriginalDraggable>,
  'active' | 'over'
> & {
  active: TypesafeActive | null;
  over: TypesafeOver | null;
};

export function useDraggable(props: UseDraggableTypesafeArguments) {
  return useOriginalDraggable(props) as UseDraggableTypesafeReturnValue;
}

interface TypesafeActive extends Omit<Active, 'data'> {
  data: React.MutableRefObject<TypesafeDraggableData | undefined>;
}

interface TypesafeOver extends Omit<Over, 'data'> {
  data: React.MutableRefObject<TypesafeDroppableData | undefined>;
}

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
      return payloadType === 'IMAGE_DTO' || 'IMAGE_NAMES';
    case 'ADD_TO_BATCH':
      return payloadType === 'IMAGE_DTO' || 'IMAGE_NAMES';
    case 'MOVE_BOARD':
      return payloadType === 'IMAGE_DTO' || 'IMAGE_NAMES';
    default:
      return false;
  }
};

interface DragEvent {
  activatorEvent: Event;
  active: TypesafeActive;
  collisions: Collision[] | null;
  delta: Translate;
  over: TypesafeOver | null;
}

export interface DragStartEvent extends Pick<DragEvent, 'active'> {}
export interface DragMoveEvent extends DragEvent {}
export interface DragOverEvent extends DragMoveEvent {}
export interface DragEndEvent extends DragEvent {}
export interface DragCancelEvent extends DragEndEvent {}

export interface DndContextTypesafeProps
  extends Omit<
    DndContextProps,
    'onDragStart' | 'onDragMove' | 'onDragOver' | 'onDragEnd' | 'onDragCancel'
  > {
  onDragStart?(event: DragStartEvent): void;
  onDragMove?(event: DragMoveEvent): void;
  onDragOver?(event: DragOverEvent): void;
  onDragEnd?(event: DragEndEvent): void;
  onDragCancel?(event: DragCancelEvent): void;
}
export function DndContext(props: DndContextTypesafeProps) {
  return <OriginalDndContext {...props} />;
}
