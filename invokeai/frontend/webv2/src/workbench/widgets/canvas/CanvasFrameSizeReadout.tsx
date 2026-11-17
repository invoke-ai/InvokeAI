/**
 * A glanceable readout of the generation frame's size and, when it differs, the size the frame renders at. It sits in
 * the canvas's top-right corner, under the center chrome, follows a Frame tool drag before it commits, and never takes
 * the pointer. The render size comes from the policy the canvas graph compiler applies, so the two cannot disagree.
 */

import type { CanvasScalingSettings, PidMode } from '@features/generation/contracts';
import type { Project } from '@workbench/projectContracts';

import { Box } from '@chakra-ui/react';
import { resolveCanvasProcessingSize } from '@features/generation/canvasProcessingSize';
import { getArchitectureCapabilitiesSnapshot, subscribeArchitectureCapabilities } from '@features/generation/runtime';
import { isGenerateModelConfig } from '@features/generation/settings';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { roundBbox, type CanvasCoreStoreCapability } from '@workbench/canvas-engine/api';
import { readCanvasScaling } from '@workbench/widgets/canvas/invoke/canvasScaling';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { useCanvasBboxPreview } from './engineStoreHooks';

interface Size {
  height: number;
  width: number;
}

const sizeEqual = (a: Size | null, b: Size | null): boolean =>
  a === b || (!!a && !!b && a.width === b.width && a.height === b.height);

const selectCommittedFrame = (project: Project): Size => {
  const { height, width } = project.canvas.document.bbox;
  return { height, width };
};

/** The inputs `compileCanvasGraph` resolves its processing size from, read from the stored settings as it reads them. */
const selectRenderInputs = (project: Project) => {
  const generate = getProjectWidgetValues(project, 'generate') as { model?: unknown; pidMode?: PidMode };
  return {
    model: isGenerateModelConfig(generate.model) ? generate.model : null,
    pidMode: generate.pidMode ?? 'off',
    scaling: readCanvasScaling(getProjectWidgetValues(project, 'canvas')),
  };
};

type RenderInputs = ReturnType<typeof selectRenderInputs>;

const scalingEqual = (a: CanvasScalingSettings, b: CanvasScalingSettings): boolean =>
  a.method === b.method && a.width === b.width && a.height === b.height;

const renderInputsEqual = (a: RenderInputs, b: RenderInputs): boolean =>
  a.model === b.model && a.pidMode === b.pidMode && scalingEqual(a.scaling, b.scaling);

// Below the center view's header row, as the tool strip sits.
const READOUT_TOP = 'var(--wb-center-chrome-inset, var(--chakra-spacing-2))';

/**
 * The layer spans the canvas so the readout can query its size: on a canvas too narrow it would crowd the top-center
 * chips (the control layer view switch), and on one too short, the tool strip's row, so it steps aside. It stacks under
 * the drop zones and the tool strip (z-index 2).
 */
const LAYER_CSS = { containerType: 'size' } as const;
const READOUT_CSS = {
  '@container (width < 34rem)': { display: 'none' },
  '@container (height < 12rem)': { display: 'none' },
} as const;
// A definition list's own margins would offset the grid.
const LIST_CSS = { '& dd': { margin: 0 } } as const;

export const CanvasFrameSizeReadout = ({ engine }: { engine: CanvasCoreStoreCapability }) => {
  const { t } = useTranslation();
  const preview = useCanvasBboxPreview(engine);
  const committed = useActiveProjectSelector(selectCommittedFrame, sizeEqual);
  const inputs = useActiveProjectSelector(selectRenderInputs, renderInputsEqual);
  // A drag shows the frame it would commit, which rounds to whole pixels.
  const frame = preview ? roundBbox(preview) : committed;
  const frameWidth = frame.width;
  const frameHeight = frame.height;
  // Read policy inside the capability selector so the model's grid replaces the fallback grid when it arrives.
  const render = useExternalStoreSelector(
    subscribeArchitectureCapabilities,
    getArchitectureCapabilitiesSnapshot,
    useCallback(
      () =>
        inputs.model
          ? resolveCanvasProcessingSize(
              inputs.model,
              inputs.pidMode,
              { height: frameHeight, width: frameWidth },
              inputs.scaling
            )
          : null,
      [frameHeight, frameWidth, inputs]
    ),
    sizeEqual
  );
  const isResized = !!render && (render.width !== frameWidth || render.height !== frameHeight);

  return (
    <Box css={LAYER_CSS} inset="0" pointerEvents="none" position="absolute" zIndex="1">
      <Box
        aria-label={t('widgets.canvas.frameReadout.label')}
        bg="bg.subtle/90"
        borderColor="border.subtle"
        borderWidth="1px"
        css={READOUT_CSS}
        position="absolute"
        right="2"
        role="group"
        rounded="md"
        shadow="sm"
        top={READOUT_TOP}
      >
        <Box
          as="dl"
          columnGap="2"
          css={LIST_CSS}
          display="grid"
          fontSize="xs"
          gridTemplateColumns="auto auto"
          lineHeight="1.4"
          px="2"
          py="1"
          rowGap="0.5"
        >
          <Box as="dt" color="fg.muted">
            {t('widgets.canvas.frameReadout.frame')}
          </Box>
          <Box as="dd" fontVariantNumeric="tabular-nums" textAlign="end">
            {t('widgets.canvas.frameReadout.size', { height: frameHeight, width: frameWidth })}
          </Box>
          {isResized ? (
            <>
              <Box as="dt" color="fg.muted">
                {t('widgets.canvas.frameReadout.render')}
              </Box>
              <Box as="dd" fontVariantNumeric="tabular-nums" textAlign="end">
                {t('widgets.canvas.frameReadout.size', { height: render.height, width: render.width })}
              </Box>
            </>
          ) : null}
        </Box>
      </Box>
    </Box>
  );
};
