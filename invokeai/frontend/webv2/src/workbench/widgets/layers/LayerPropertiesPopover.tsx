import type { CanvasEngine } from '@workbench/canvas-engine/engine';
import type { CanvasLayerContract } from '@workbench/types';
import type { WorkbenchAction } from '@workbench/workbenchState';
import type { Dispatch } from 'react';

import { Popover, Portal, Stack, Switch, Text } from '@chakra-ui/react';
import { IconButton } from '@workbench/components/ui';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { SlidersHorizontalIcon } from 'lucide-react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { AdjustmentsPopover } from './AdjustmentsPopover';
import { ControlLayerSettings } from './ControlLayerSettings';
import { InpaintMaskSettings } from './InpaintMaskSettings';
import { applyStructural } from './layerOps';
import { clearLayerPropertiesRequest, useLayerPropertiesRequest } from './layerPropertiesRequestStore';
import { RasterLayerFilterSection } from './RasterLayerFilterSection';
import { RegionalGuidanceSettings } from './RegionalGuidanceSettings';

const POPOVER_POSITIONING = { placement: 'left-start' } as const;

const stopPropagation = (event: { stopPropagation: () => void }): void => event.stopPropagation();

interface LayerPropertiesPopoverProps {
  dispatch: Dispatch<WorkbenchAction>;
  engine: CanvasEngine | null;
  layer: CanvasLayerContract;
}

/**
 * The per-layer "properties/configure" affordance (round-3 restructure): a sliders
 * IconButton on each row that opens a popover holding that layer's type-specific
 * settings — control adapter/filter, regional prompts/ref-images, inpaint mask
 * fill/noise, or raster adjustments. Previously these were
 * stacked in the panel header for the *selected* layer; moving them into a per-row
 * popover keeps the header slim and each layer's config next to the layer.
 *
 * Content is mounted ONLY while open (`lazyMount` + `unmountOnExit`, the codebase
 * convention) — unmount-on-close is what fires `ControlLayerSettings`' filter-preview
 * ref-callback teardown, so a previewed filter can never outlive a closed popover
 * (Task 38 lesson). The type-specific settings are additionally `key`ed on
 * `layer.id` (Task 39 lesson: the cleanup relies on a fresh instance per layer).
 */
export const LayerPropertiesPopover = ({ dispatch, engine, layer }: LayerPropertiesPopoverProps) => {
  const { t } = useTranslation();
  const [triggerOpen, setTriggerOpen] = useState(false);
  const request = useLayerPropertiesRequest(layer.id);
  const documentRevision = useActiveProjectSelector((project) => project.canvas.documentRevision);
  const isOpen = triggerOpen || request !== null;

  const handleOpenChange = useCallback(
    (details: { open: boolean }) => {
      setTriggerOpen(details.open);
      if (!details.open && request) {
        clearLayerPropertiesRequest(request.token);
      }
    },
    [request]
  );

  return (
    <Popover.Root
      lazyMount
      open={isOpen}
      positioning={POPOVER_POSITIONING}
      unmountOnExit
      onOpenChange={handleOpenChange}
    >
      <Popover.Trigger asChild>
        <IconButton
          aria-label={t('widgets.layers.properties')}
          color="fg.subtle"
          size="2xs"
          variant="ghost"
          onClick={stopPropagation}
          onPointerDown={stopPropagation}
        >
          <SlidersHorizontalIcon />
        </IconButton>
      </Popover.Trigger>
      <Portal>
        <Popover.Positioner>
          <Popover.Content bg="bg.muted" borderColor="border.emphasized" borderWidth="1px" w="20rem">
            <Popover.Body p="2.5">
              <Stack gap="2">
                <LayerTypeSettings
                  dispatch={dispatch}
                  documentRevision={documentRevision}
                  engine={engine}
                  filterRequestToken={request?.section === 'filter' ? request.token : null}
                  layer={layer}
                />
              </Stack>
            </Popover.Body>
          </Popover.Content>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
  );
};

/** Dispatches to the correct per-type settings block for the layer. */
const LayerTypeSettings = ({
  dispatch,
  documentRevision,
  engine,
  filterRequestToken,
  layer,
}: {
  dispatch: Dispatch<WorkbenchAction>;
  documentRevision: number;
  engine: CanvasEngine | null;
  filterRequestToken: number | null;
  layer: CanvasLayerContract;
}) => {
  switch (layer.type) {
    case 'inpaint_mask':
      return <InpaintMaskSettings key={layer.id} engine={engine} layer={layer} />;
    case 'regional_guidance':
      return <RegionalGuidanceSettings key={layer.id} engine={engine} layer={layer} />;
    case 'control':
      return (
        <ControlLayerSettings key={layer.id} engine={engine} filterRequestToken={filterRequestToken} layer={layer} />
      );
    case 'raster':
      return (
        <RasterLayerSettings
          key={`${engine?.projectId ?? 'none'}-${layer.id}-${documentRevision}`}
          dispatch={dispatch}
          engine={engine}
          filterRequestToken={filterRequestToken}
          layer={layer}
        />
      );
  }
};

/** Raster-layer properties: transparency lock + non-destructive adjustments. */
const RasterLayerSettings = ({
  dispatch,
  engine,
  filterRequestToken,
  layer,
}: {
  dispatch: Dispatch<WorkbenchAction>;
  engine: CanvasEngine | null;
  filterRequestToken: number | null;
  layer: Extract<CanvasLayerContract, { type: 'raster' }>;
}) => {
  const { t } = useTranslation();
  const isLocked = layer.isTransparencyLocked === true;

  const handleTransparencyLock = useCallback(
    (details: { checked: boolean }) => {
      applyStructural(
        engine,
        dispatch,
        t('widgets.layers.adjustments.transparencyLock'),
        {
          config: { isTransparencyLocked: details.checked, layerType: 'raster' },
          id: layer.id,
          type: 'updateCanvasLayerConfig',
        },
        {
          config: { isTransparencyLocked: isLocked, layerType: 'raster' },
          id: layer.id,
          type: 'updateCanvasLayerConfig',
        }
      );
    },
    [dispatch, engine, isLocked, layer.id, t]
  );

  return (
    <Stack borderColor="border.subtle" borderWidth="1px" gap="2" p="2" rounded="md">
      <Switch.Root checked={isLocked} size="sm" onCheckedChange={handleTransparencyLock}>
        <Switch.HiddenInput />
        <Switch.Control>
          <Switch.Thumb />
        </Switch.Control>
        <Switch.Label>
          <Text fontSize="xs">{t('widgets.layers.adjustments.transparencyLock')}</Text>
        </Switch.Label>
      </Switch.Root>
      <Text color="fg.subtle" fontSize="2xs" fontWeight="700" textTransform="uppercase">
        {t('widgets.layers.adjustments.title')}
      </Text>
      <AdjustmentsPopover engine={engine} layer={layer} />
      <RasterLayerFilterSection
        key={`${layer.id}-${filterRequestToken ?? 'default'}`}
        engine={engine}
        focusFilter={filterRequestToken !== null}
        layer={layer}
      />
    </Stack>
  );
};
