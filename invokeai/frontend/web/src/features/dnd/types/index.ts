// type-safe dnd from https://github.com/clauderic/dnd-kit/issues/935
import {
  Active,
  Collision,
  DndContextProps,
  Over,
  Translate,
  UseDraggableArguments,
  UseDroppableArguments,
  useDraggable as useOriginalDraggable,
  useDroppable as useOriginalDroppable,
} from '@dnd-kit/core';
import {
  InputFieldTemplate,
  InputFieldValue,
} from 'features/nodes/types/types';
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

export type AddToBoardDropData = BaseDropData & {
  actionType: 'ADD_TO_BOARD';
  context: { boardId: string };
};

export type RemoveFromBoardDropData = BaseDropData & {
  actionType: 'REMOVE_FROM_BOARD';
};

export type AddFieldToLinearViewDropData = BaseDropData & {
  actionType: 'ADD_FIELD_TO_LINEAR';
};

export type TypesafeDroppableData =
  | CurrentImageDropData
  | InitialImageDropData
  | ControlNetDropData
  | CanvasInitialImageDropData
  | NodesImageDropData
  | AddToBatchDropData
  | NodesMultiImageDropData
  | AddToBoardDropData
  | RemoveFromBoardDropData
  | AddFieldToLinearViewDropData;

type BaseDragData = {
  id: string;
};

export type NodeFieldDraggableData = BaseDragData & {
  payloadType: 'NODE_FIELD';
  payload: {
    nodeId: string;
    field: InputFieldValue;
    fieldTemplate: InputFieldTemplate;
  };
};

export type ImageDraggableData = BaseDragData & {
  payloadType: 'IMAGE_DTO';
  payload: { imageDTO: ImageDTO };
};

export type ImageDTOsDraggableData = BaseDragData & {
  payloadType: 'IMAGE_DTOS';
  payload: { imageDTOs: ImageDTO[] };
};

export type TypesafeDraggableData =
  | NodeFieldDraggableData
  | ImageDraggableData
  | ImageDTOsDraggableData;

export interface UseDroppableTypesafeArguments
  extends Omit<UseDroppableArguments, 'data'> {
  data?: TypesafeDroppableData;
}

export type UseDroppableTypesafeReturnValue = Omit<
  ReturnType<typeof useOriginalDroppable>,
  'active' | 'over'
> & {
  active: TypesafeActive | null;
  over: TypesafeOver | null;
};

export interface UseDraggableTypesafeArguments
  extends Omit<UseDraggableArguments, 'data'> {
  data?: TypesafeDraggableData;
}

export type UseDraggableTypesafeReturnValue = Omit<
  ReturnType<typeof useOriginalDraggable>,
  'active' | 'over'
> & {
  active: TypesafeActive | null;
  over: TypesafeOver | null;
};

export interface TypesafeActive extends Omit<Active, 'data'> {
  data: React.MutableRefObject<TypesafeDraggableData | undefined>;
}

export interface TypesafeOver extends Omit<Over, 'data'> {
  data: React.MutableRefObject<TypesafeDroppableData | undefined>;
}

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
