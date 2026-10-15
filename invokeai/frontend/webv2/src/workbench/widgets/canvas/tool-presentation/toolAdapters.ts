import type { ToolId } from '@workbench/canvas-engine/api';

import { filterOperationForm } from '@workbench/widgets/canvas/tool-options/FilterOptions';
import { bboxForm, moveForm, transformForm } from '@workbench/widgets/canvas/tool-options/geometryForm';
import { gradientForm } from '@workbench/widgets/canvas/tool-options/GradientOptions';
import { brushForm, eraserForm } from '@workbench/widgets/canvas/tool-options/paintForm';
import { selectObjectOperationForm } from '@workbench/widgets/canvas/tool-options/SamOptions';
import { lassoForm, marqueeForm } from '@workbench/widgets/canvas/tool-options/selectionForm';
import { shapeForm } from '@workbench/widgets/canvas/tool-options/ShapeOptions';
import { textForm } from '@workbench/widgets/canvas/tool-options/textForm';

import type { CanvasOperationKind, OperationPropertyForm, ToolPropertyForm } from './toolFormContracts';

export const TOOL_PRESENTATION_ADAPTERS: Readonly<Record<ToolId, ToolPropertyForm>> = {
  bbox: bboxForm,
  brush: brushForm,
  colorPicker: { groups: [], id: 'colorPicker' },
  eraser: eraserForm,
  gradient: gradientForm,
  lasso: lassoForm,
  marquee: marqueeForm,
  move: moveForm,
  sam: { groups: [], id: 'sam' },
  shape: shapeForm,
  text: textForm,
  transform: transformForm,
  view: { groups: [], id: 'view' },
};

export const OPERATION_PRESENTATION_ADAPTERS: Readonly<Record<CanvasOperationKind, OperationPropertyForm>> = {
  filter: filterOperationForm,
  'select-object': selectObjectOperationForm,
};
