import type { CanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import type { CanvasLayerContract } from '@workbench/types';
import type { WorkbenchAction } from '@workbench/workbenchState';
import type { Dispatch } from 'react';

import { Box, Popover, Portal, Stack, Switch, Text } from '@chakra-ui/react';
import { IconButton } from '@workbench/components/ui';
import { useCanvasDocumentEditingLocked } from '@workbench/widgets/canvas/engineStoreHooks';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { SlidersHorizontalIcon } from 'lucide-react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { LayerPropertiesRequest } from './layerPropertiesRequestStore';

import { AdjustmentsPopover } from './AdjustmentsPopover';
import { ControlLayerSettings } from './ControlLayerSettings';
import { InpaintMaskSettings } from './InpaintMaskSettings';
import { applyStructural } from './layerOps';
import {
  closeLayerPropertiesForOperation,
  getLayerPropertiesOwnershipKey,
  isLayerPropertiesOpen,
} from './layerPropertiesOperation';
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
 * Content is mounted only while open (`lazyMount` + `unmountOnExit`). Starting a
 * canvas operation explicitly clears both trigger and request ownership before
 * the operation panel takes over. Type-specific settings are keyed by layer so
 * switching targets always mounts a fresh settings instance.
 */
export const LayerPropertiesPopover = (props: LayerPropertiesPopoverProps) => {
  const editingLocked = useCanvasDocumentEditingLocked(props.engine);
  const request = useLayerPropertiesRequest(props.layer.id);
  return (
    <LayerPropertiesPopoverOwnership
      key={`${props.layer.id}:${getLayerPropertiesOwnershipKey(editingLocked)}`}
      {...props}
      editingLocked={editingLocked}
      request={request}
    />
  );
};

const LayerPropertiesPopoverOwnership = ({
  dispatch,
  editingLocked,
  engine,
  layer,
  request,
}: LayerPropertiesPopoverProps & { editingLocked: boolean; request: LayerPropertiesRequest | null }) => {
  const { t } = useTranslation();
  const [triggerOpen, setTriggerOpen] = useState(false);
  const documentRevision = useActiveProjectSelector((project) => project.canvas.documentRevision);
  const isOpen = !editingLocked && isLayerPropertiesOpen({ requestToken: request?.token ?? null, triggerOpen });
  const consumeLockedRequestRef = useCallback(
    (node: HTMLDivElement | null) => {
      if (node && editingLocked && request) {
        clearLayerPropertiesRequest(request.token);
      }
    },
    [editingLocked, request]
  );

  const handleOpenChange = useCallback(
    (details: { open: boolean }) => {
      setTriggerOpen(details.open);
      if (!details.open && request) {
        clearLayerPropertiesRequest(request.token);
      }
    },
    [request]
  );
  const handleOperationStarted = useCallback(() => {
    const closed = closeLayerPropertiesForOperation({ requestToken: request?.token ?? null, triggerOpen });
    setTriggerOpen(closed.triggerOpen);
    if (closed.requestTokenToClear !== null) {
      clearLayerPropertiesRequest(closed.requestTokenToClear);
    }
  }, [request?.token, triggerOpen]);

  return (
    <>
      {editingLocked && request ? <Box ref={consumeLockedRequestRef} display="none" /> : null}
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
            disabled={editingLocked}
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
                <Stack gap="2" inert={editingLocked}>
                  <LayerTypeSettings
                    dispatch={dispatch}
                    documentRevision={documentRevision}
                    engine={engine}
                    layer={layer}
                    onOperationStarted={handleOperationStarted}
                  />
                </Stack>
              </Popover.Body>
            </Popover.Content>
          </Popover.Positioner>
        </Portal>
      </Popover.Root>
    </>
  );
};

/** Dispatches to the correct per-type settings block for the layer. */
const LayerTypeSettings = ({
  dispatch,
  documentRevision,
  engine,
  layer,
  onOperationStarted,
}: {
  dispatch: Dispatch<WorkbenchAction>;
  documentRevision: number;
  engine: CanvasEngine | null;
  layer: CanvasLayerContract;
  onOperationStarted(): void;
}) => {
  switch (layer.type) {
    case 'inpaint_mask':
      return <InpaintMaskSettings key={layer.id} engine={engine} layer={layer} />;
    case 'regional_guidance':
      return <RegionalGuidanceSettings key={layer.id} engine={engine} layer={layer} />;
    case 'control':
      return (
        <ControlLayerSettings key={layer.id} engine={engine} layer={layer} onOperationStarted={onOperationStarted} />
      );
    case 'raster':
      return (
        <RasterLayerSettings
          key={`${engine?.projectId ?? 'none'}-${layer.id}-${documentRevision}`}
          dispatch={dispatch}
          engine={engine}
          layer={layer}
          onOperationStarted={onOperationStarted}
        />
      );
  }
};

/** Raster-layer properties: transparency lock + non-destructive adjustments. */
const RasterLayerSettings = ({
  dispatch,
  engine,
  layer,
  onOperationStarted,
}: {
  dispatch: Dispatch<WorkbenchAction>;
  engine: CanvasEngine | null;
  layer: Extract<CanvasLayerContract, { type: 'raster' }>;
  onOperationStarted(): void;
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
    <Stack gap="2">
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
      <RasterLayerFilterSection engine={engine} layer={layer} onOperationStarted={onOperationStarted} />
    </Stack>
  );
};
