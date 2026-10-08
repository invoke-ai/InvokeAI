/**
 * Canvas mode's render size: generate at the frame's size, the model's optimal size, or a custom size, then resize
 * the result back to the frame. The choice persists in canvas values for invocation compilation.
 */

import type { NumberInput as ChakraNumberInput, SelectValueChangeDetails } from '@chakra-ui/react';
import type { CanvasScaleMethod, CanvasScalingSettings, PidMode } from '@features/generation/contracts';
import type { CanvasRenderSizeProps } from '@features/generation/react';
import type { Project } from '@workbench/projectContracts';

import { Box, chakra, createListCollection, Grid, HStack, NumberInput, Text } from '@chakra-ui/react';
import { resolveCanvasProcessingSize } from '@features/generation/canvasProcessingSize';
import { getArchitectureCapabilitiesSnapshot, subscribeArchitectureCapabilities } from '@features/generation/runtime';
import { clampDimension, getGenerationDimensions } from '@features/generation/settings';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { FieldLabel, Select, Tooltip } from '@platform/ui';
import {
  CANVAS_SCALE_METHODS,
  CANVAS_SCALING_KEYS,
  readCanvasScaling,
} from '@workbench/widgets/canvas/invoke/canvasScaling';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { type MouseEvent, useCallback, useId, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const SELECT_POSITIONING = { placement: 'bottom-start', sameWidth: true } as const;
const SIDE_INPUT_CSS = { flex: '1', minW: '0' } as const;
const VALUE_TEXT_PROPS = { truncate: true } as const;

const selectScaling = (project: Project): CanvasScalingSettings =>
  readCanvasScaling(getProjectWidgetValues(project, 'canvas'));

/** Other canvas values (denoising, compositing, colours) change often and must not re-render the Size section. */
const scalingEqual = (a: CanvasScalingSettings, b: CanvasScalingSettings): boolean =>
  a.method === b.method && a.width === b.width && a.height === b.height;

const readSize = (value: unknown): number | null =>
  typeof value === 'number' && Number.isFinite(value) ? value : null;

/**
 * Use model base/type/variant and PiD mode matching compileCanvasGraph so displayed size follows identical
 * grid/area policy.
 */
const selectProcessingContext = (project: Project) => {
  const generate = getProjectWidgetValues(project, 'generate') as {
    height?: unknown;
    model?: { base?: string; type?: string; variant?: unknown } | null;
    pidMode?: PidMode;
    width?: unknown;
  };
  const model = generate.model;
  return {
    bbox: project.canvas.document.bbox,
    committedFrame: { height: readSize(generate.height), width: readSize(generate.width) },
    model:
      model && typeof model.base === 'string'
        ? {
            base: model.base,
            type: model.type ?? 'main',
            variant: typeof model.variant === 'string' ? model.variant : null,
          }
        : null,
    pidMode: generate.pidMode ?? 'off',
  };
};

type ProcessingContext = ReturnType<typeof selectProcessingContext>;

const processingContextEqual = (a: ProcessingContext, b: ProcessingContext): boolean =>
  a.bbox.width === b.bbox.width &&
  a.bbox.height === b.bbox.height &&
  a.committedFrame.width === b.committedFrame.width &&
  a.committedFrame.height === b.committedFrame.height &&
  a.model?.base === b.model?.base &&
  a.model?.type === b.model?.type &&
  a.model?.variant === b.model?.variant &&
  a.pidMode === b.pidMode;

export const GenerateCanvasRenderSize = ({ children, frame }: CanvasRenderSizeProps) => {
  const { t } = useTranslation();
  const { widgets } = useWorkbenchCommands();
  const scaling = useActiveProjectSelector(selectScaling, scalingEqual);
  const context = useActiveProjectSelector(selectProcessingContext, processingContextEqual);
  // The committed frame resolves from the exact bbox, as the compiler does; a resize the Size section has not
  // committed yet resolves from the size it shows, so the preview follows a drag.
  const isFrameCommitted =
    (context.committedFrame.width === null || context.committedFrame.width === frame.width) &&
    (context.committedFrame.height === null || context.committedFrame.height === frame.height);
  const sourceWidth = isFrameCommitted ? context.bbox.width : frame.width;
  const sourceHeight = isFrameCommitted ? context.bbox.height : frame.height;
  // Read policy inside the capability selector so arrival replaces fallback grid/area despite compiler
  // memoization.
  const { dimensions, processingSize } = useExternalStoreSelector(
    subscribeArchitectureCapabilities,
    getArchitectureCapabilitiesSnapshot,
    useCallback(() => {
      const model = context.model as Parameters<typeof resolveCanvasProcessingSize>[0] | null;
      return {
        dimensions: getGenerationDimensions(model ?? undefined, context.pidMode),
        processingSize: model
          ? resolveCanvasProcessingSize(model, context.pidMode, { height: sourceHeight, width: sourceWidth }, scaling)
          : null,
      };
    }, [context, scaling, sourceHeight, sourceWidth])
  );

  const methodLabel = useCallback(
    (method: CanvasScaleMethod) => t(`widgets.generate.renderSize.methods.${method}`),
    [t]
  );
  const patch = useCallback((partial: Record<string, unknown>) => widgets.patchValues('canvas', partial), [widgets]);

  const methodCollection = useMemo(
    () =>
      createListCollection({
        items: CANVAS_SCALE_METHODS.map((method) => ({ label: methodLabel(method), value: method })),
      }),
    [methodLabel]
  );
  const methodValue = useMemo(() => [scaling.method], [scaling.method]);
  const handleMethodChange = useCallback(
    ({ value }: SelectValueChangeDetails) => {
      const method = value[0] as CanvasScaleMethod | undefined;
      if (!method) {
        return;
      }
      // Custom starts from the size the frame renders at today, so both
      // sides are pinned and a later frame resize cannot move one of them.
      const seed =
        method === 'manual' && processingSize
          ? {
              [CANVAS_SCALING_KEYS.height]: scaling.height ?? processingSize.height,
              [CANVAS_SCALING_KEYS.width]: scaling.width ?? processingSize.width,
            }
          : {};
      patch({ ...seed, [CANVAS_SCALING_KEYS.method]: method });
    },
    [patch, processingSize, scaling.height, scaling.width]
  );
  const snap = useCallback((value: number) => clampDimension(value, dimensions.grid), [dimensions.grid]);
  // Typing needs the raw digits to land; the size snaps to the grid on commit
  // (Enter / blur), and the compiler clamps whatever is persisted meanwhile.
  const sideHandlers = useCallback(
    (key: string) => ({
      onValueChange: ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
        if (Number.isFinite(valueAsNumber)) {
          patch({ [key]: valueAsNumber });
        }
      },
      onValueCommit: ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
        if (Number.isFinite(valueAsNumber)) {
          patch({ [key]: snap(valueAsNumber) });
        }
      },
    }),
    [patch, snap]
  );
  const widthHandlers = useMemo(() => sideHandlers(CANVAS_SCALING_KEYS.width), [sideHandlers]);
  const heightHandlers = useMemo(() => sideHandlers(CANVAS_SCALING_KEYS.height), [sideHandlers]);

  const sideInput = (label: string, value: number, handlers: ReturnType<typeof sideHandlers>) => (
    <NumberInput.Root
      css={SIDE_INPUT_CSS}
      max={dimensions.max}
      min={dimensions.min}
      step={dimensions.grid}
      value={String(value)}
      {...handlers}
    >
      <NumberInput.Control />
      <NumberInput.Input aria-label={label} />
    </NumberInput.Root>
  );

  const label = t('widgets.generate.renderSize.label');
  // The tooltip's trigger is the wrapper, so the select points at an always-present copy of the tip itself.
  const descriptionId = useId();
  const triggerId = useId();
  const selectIds = useMemo(() => ({ trigger: triggerId }), [triggerId]);
  const triggerProps = useMemo(() => ({ 'aria-describedby': descriptionId }), [descriptionId]);
  const description = t('widgets.generate.renderSize.description');
  // Like a native select's label, a click focuses the select rather than activating (opening) it.
  const focusSelect = useCallback(
    (event: MouseEvent<HTMLLabelElement>) => {
      event.preventDefault();
      document.getElementById(triggerId)?.focus();
    },
    [triggerId]
  );
  // Custom's sides sit under the select they belong to, so the grid keeps the label in its own column.
  const controls = (
    <Grid alignItems="center" columnGap="2" rowGap="1.5" templateColumns="fit-content(40%) minmax(0, 1fr)">
      {/* The select carries the same name for assistive technology; this copy is what sighted people read and click. */}
      <chakra.label aria-hidden cursor="default" htmlFor={triggerId} onClick={focusSelect}>
        <FieldLabel>{label}</FieldLabel>
      </chakra.label>
      {/* The legacy name for this setting lives in the tooltip, so people who knew it can still find it. */}
      <Tooltip content={description} placement="top">
        <Box minW="0">
          <Text id={descriptionId} srOnly>
            {description}
          </Text>
          <Select
            aria-label={label}
            collection={methodCollection}
            ids={selectIds}
            positioning={SELECT_POSITIONING}
            triggerProps={triggerProps}
            value={methodValue}
            valueText={methodLabel(scaling.method)}
            valueTextProps={VALUE_TEXT_PROPS}
            onValueChange={handleMethodChange}
          />
        </Box>
      </Tooltip>
      {scaling.method === 'manual' ? (
        <HStack gap="1" gridColumn="2" minW="0">
          {sideInput(t('widgets.generate.renderSize.width'), scaling.width ?? context.bbox.width, widthHandlers)}
          <Text aria-hidden color="fg.muted" flexShrink="0" fontSize="xs">
            ×
          </Text>
          {sideInput(t('widgets.generate.renderSize.height'), scaling.height ?? context.bbox.height, heightHandlers)}
        </HStack>
      ) : null}
    </Grid>
  );

  return children({ controls, size: processingSize });
};
