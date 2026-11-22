import type { ToolId } from '@workbench/canvas-engine/api';

import { Skeleton } from '@chakra-ui/react';
import { filterOperationForm } from '@workbench/widgets/canvas/tool-options/FilterOptions';
import { bboxForm, moveForm, transformForm } from '@workbench/widgets/canvas/tool-options/geometryForm';
import { gradientForm } from '@workbench/widgets/canvas/tool-options/GradientOptions';
import { brushForm, eraserForm } from '@workbench/widgets/canvas/tool-options/paintForm';
import { selectObjectOperationForm } from '@workbench/widgets/canvas/tool-options/SamOptions';
import { lassoForm, marqueeForm } from '@workbench/widgets/canvas/tool-options/selectionForm';
import { shapeForm } from '@workbench/widgets/canvas/tool-options/ShapeOptions';
import { textForm } from '@workbench/widgets/canvas/tool-options/textForm';
import { createElement, lazy, Suspense } from 'react';

import type { CanvasOperationKind, OperationPropertyForm, ToolFormProps, ToolPropertyForm } from './toolFormContracts';

const SplineSettings = lazy(() =>
  import('@workbench/widgets/canvas/tool-options/SplineOptions').then((module) => ({ default: module.SplineSettings }))
);
const splineLoadingRows = createElement(Skeleton, { height: '20' });
const SplineBody = (props: ToolFormProps) =>
  createElement(Suspense, { fallback: splineLoadingRows }, createElement(SplineSettings, props));
const splineForm: ToolPropertyForm = {
  id: 'spline',
  groups: [{ body: SplineBody, id: 'spline-points', labelKey: 'widgets.layers.vector.pointsTitle' }],
};

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
  spline: splineForm,
  text: textForm,
  transform: transformForm,
  view: { groups: [], id: 'view' },
};

export const OPERATION_PRESENTATION_ADAPTERS: Readonly<Record<CanvasOperationKind, OperationPropertyForm>> = {
  filter: filterOperationForm,
  'select-object': selectObjectOperationForm,
};
