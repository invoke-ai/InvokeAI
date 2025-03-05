import type { AnyNode } from 'features/nodes/types/invocation';

/**
 * How long to wait before showing a tooltip when hovering a field handle.
 */
export const HANDLE_TOOLTIP_OPEN_DELAY = 500;

/**
 * The width of a node in the UI in pixels.
 */
export const NODE_WIDTH = 320;

/**
 * This class name is special - reactflow uses it to identify the drag handle of a node,
 * applying the appropriate listeners to it.
 */
export const DRAG_HANDLE_CLASSNAME = 'node-drag-handle';

/**
 * reactflow-specifc properties shared between all node types.
 */
export const SHARED_NODE_PROPERTIES: Partial<AnyNode> = {
  dragHandle: `.${DRAG_HANDLE_CLASSNAME}`,
};

export const NO_FIT_ON_DOUBLE_CLICK_CLASS = 'no-fit-on-double-click';
// These are the reactflow classes that are used to disable dragging, panning, and zooming.
export const NO_DRAG_CLASS = 'nodrag';
export const NO_WHEEL_CLASS = 'nowheel';
export const NO_PAN_CLASS = 'nopan';

/**
 * Colors for each field type - applies to their handles and edges.
 */
export const FIELD_COLORS: { [key: string]: string } = {
  BoardField: 'purple.500',
  BooleanField: 'green.500',
  CLIPField: 'green.500',
  ColorField: 'pink.300',
  ConditioningField: 'cyan.500',
  FluxConditioningField: 'cyan.500',
  ControlField: 'teal.500',
  ControlNetModelField: 'teal.500',
  EnumField: 'blue.500',
  FloatField: 'orange.500',
  ImageField: 'purple.500',
  ImageBatchField: 'purple.500',
  IntegerField: 'red.500',
  IPAdapterField: 'teal.500',
  IPAdapterModelField: 'teal.500',
  LatentsField: 'pink.500',
  LoRAModelField: 'teal.500',
  MainModelField: 'teal.500',
  FluxMainModelField: 'teal.500',
  SD3MainModelField: 'teal.500',
  SDXLMainModelField: 'teal.500',
  SDXLRefinerModelField: 'teal.500',
  SpandrelImageToImageModelField: 'teal.500',
  StringField: 'yellow.500',
  T2IAdapterField: 'teal.500',
  T2IAdapterModelField: 'teal.500',
  UNetField: 'red.500',
  T5EncoderField: 'green.500',
  TransformerField: 'red.500',
  VAEField: 'blue.500',
  VAEModelField: 'teal.500',
};
