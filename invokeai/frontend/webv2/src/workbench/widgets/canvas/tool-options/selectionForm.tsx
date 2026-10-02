import type { SelectionOp } from '@workbench/canvas-engine/api';
import type {
  ToolFormProps,
  ToolPropertyForm,
  ToolPropertyGroup,
} from '@workbench/widgets/canvas/tool-presentation/toolFormContracts';

import { Text } from '@chakra-ui/react';
import {
  useCanvasActiveTool,
  useCanvasHasSelection,
  useLassoOptions,
  useMarqueeOptions,
} from '@workbench/widgets/canvas/engineStoreHooks';
import {
  PropertyControlRow,
  PropertySegmentedRow,
} from '@workbench/widgets/canvas/tool-presentation/PropertyPrimitives';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { SelectionActions, SelectionOpModeButtons } from './SelectionOptionsRow';

/** Share selection rows across lasso/marquee to preserve DOM identity while active-tool stores supply values. */
const SelectionModeSettings = ({ engine, isSurfaceInteractionLocked }: ToolFormProps) => {
  const { t } = useTranslation();
  const activeTool = useCanvasActiveTool(engine);
  const isMarquee = activeTool === 'marquee';
  const lasso = useLassoOptions(engine);
  const marquee = useMarqueeOptions(engine);
  const hasSelection = useCanvasHasSelection(engine);
  void isSurfaceInteractionLocked;

  const shapeOptions = useMemo(
    () =>
      isMarquee
        ? [
            { label: t('widgets.canvas.toolOptions.shapeRect'), value: 'rect' },
            { label: t('widgets.canvas.toolOptions.shapeEllipse'), value: 'ellipse' },
          ]
        : [
            { label: t('widgets.canvas.toolOptions.lassoFreehand'), value: 'freehand' },
            { label: t('widgets.canvas.toolOptions.lassoPolygon'), value: 'polygon' },
          ],
    [isMarquee, t]
  );
  const shapeValue = isMarquee ? marquee.kind : lasso.shape;
  const onShapeChange = useCallback(
    (next: string) => {
      if (isMarquee) {
        engine.interaction.set('marqueeOptions', { ...marquee, kind: next as typeof marquee.kind });
      } else {
        engine.interaction.set('lassoOptions', { ...lasso, shape: next as typeof lasso.shape });
      }
    },
    [engine, isMarquee, lasso, marquee]
  );
  const mode = isMarquee ? marquee.mode : lasso.mode;
  const onModeChange = useCallback(
    (next: SelectionOp) => {
      if (isMarquee) {
        engine.interaction.set('marqueeOptions', { ...marquee, mode: next });
      } else {
        engine.interaction.set('lassoOptions', { ...lasso, mode: next });
      }
    },
    [engine, isMarquee, lasso, marquee]
  );
  const hintKey = isMarquee
    ? 'widgets.canvas.toolOptions.marqueeHint'
    : lasso.shape === 'polygon'
      ? 'widgets.canvas.toolOptions.lassoPolygonHint'
      : 'widgets.canvas.toolOptions.lassoHint';

  return (
    <>
      <PropertySegmentedRow
        label={t('widgets.properties.rows.shape')}
        options={shapeOptions}
        value={shapeValue}
        onValueChange={onShapeChange}
      />
      <PropertyControlRow label={t('widgets.properties.rows.mode')}>
        <SelectionOpModeButtons mode={mode} onModeChange={onModeChange} />
      </PropertyControlRow>
      {hasSelection ? null : (
        <Text color="fg.muted" fontSize="2xs">
          {t(hintKey)}
        </Text>
      )}
    </>
  );
};

/** Shared literals: the same objects in both forms, so the rows' DOM survives the switch. */
const MODE_GROUP: ToolPropertyGroup = {
  body: SelectionModeSettings,
  id: 'selection-mode',
  labelKey: 'widgets.properties.groups.selection',
};
const ACTIONS_GROUP: ToolPropertyGroup = {
  body: SelectionActions,
  id: 'selection-actions',
  labelKey: 'widgets.canvas.toolOptions.selectionActions',
};

export const lassoForm: ToolPropertyForm = {
  groups: [MODE_GROUP, ACTIONS_GROUP],
  id: 'lasso',
};

export const marqueeForm: ToolPropertyForm = {
  groups: [MODE_GROUP, ACTIONS_GROUP],
  id: 'marquee',
};
