import { Button, ButtonGroup, Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { FilterSettings } from 'features/controlLayers/components/Filters/FilterSettings';
import { FilterTypeSelect } from 'features/controlLayers/components/Filters/FilterTypeSelect';
import { $canvasManager } from 'features/controlLayers/konva/CanvasManager';
import { $filteringEntity } from 'features/controlLayers/store/canvasV2Slice';
import { memo, useCallback } from 'react';

export const Filter = memo(() => {
  const filteringEntity = useStore($filteringEntity);

  const preview = useCallback(() => {
    if (!filteringEntity) {
      return;
    }
    const canvasManager = $canvasManager.get();
    if (!canvasManager) {
      return;
    }
    const entity = canvasManager.stateApi.getEntity(filteringEntity);
    if (!entity || (entity.type !== 'raster_layer' && entity.type !== 'control_layer')) {
      return;
    }
    entity.adapter.filter.previewFilter();
  }, [filteringEntity]);

  const apply = useCallback(() => {
    if (!filteringEntity) {
      return;
    }
    const canvasManager = $canvasManager.get();
    if (!canvasManager) {
      return;
    }
    const entity = canvasManager.stateApi.getEntity(filteringEntity);
    if (!entity || (entity.type !== 'raster_layer' && entity.type !== 'control_layer')) {
      return;
    }
    entity.adapter.filter.applyFilter();
  }, [filteringEntity]);

  const cancel = useCallback(() => {
    if (!filteringEntity) {
      return;
    }
    const canvasManager = $canvasManager.get();
    if (!canvasManager) {
      return;
    }
    const entity = canvasManager.stateApi.getEntity(filteringEntity);
    if (!entity || (entity.type !== 'raster_layer' && entity.type !== 'control_layer')) {
      return;
    }
    entity.adapter.filter.cancelFilter();
  }, [filteringEntity]);

  return (
    <Flex flexDir="column" gap={3} w="full" h="full">
      <FilterTypeSelect />
      <ButtonGroup isAttached={false}>
        <Button onClick={preview} isDisabled={!filteringEntity}>
          Preview
        </Button>
        <Button onClick={apply} isDisabled={!filteringEntity}>
          Apply
        </Button>
        <Button onClick={cancel} isDisabled={!filteringEntity}>
          Cancel
        </Button>
      </ButtonGroup>
      <FilterSettings />
    </Flex>
  );
});

Filter.displayName = 'Filter';
