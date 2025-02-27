import { Flex, IconButton } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppStore } from 'app/store/nanostores/store';
import { useAppSelector } from 'app/store/storeHooks';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import { BeginEndStepPct } from 'features/controlLayers/components/common/BeginEndStepPct';
import { Weight } from 'features/controlLayers/components/common/Weight';
import { ControlLayerControlAdapterControlMode } from 'features/controlLayers/components/ControlLayer/ControlLayerControlAdapterControlMode';
import { ControlLayerControlAdapterModel } from 'features/controlLayers/components/ControlLayer/ControlLayerControlAdapterModel';
import { useEntityAdapterContext } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { usePullBboxIntoLayer } from 'features/controlLayers/hooks/saveCanvasHooks';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useEntityFilter } from 'features/controlLayers/hooks/useEntityFilter';
import {
  controlLayerBeginEndStepPctChanged,
  controlLayerControlModeChanged,
  controlLayerModelChanged,
  controlLayerWeightChanged,
} from 'features/controlLayers/store/canvasSlice';
import { getFilterForModel } from 'features/controlLayers/store/filters';
import { selectIsFLUX } from 'features/controlLayers/store/paramsSlice';
import { selectCanvasSlice, selectEntityOrThrow } from 'features/controlLayers/store/selectors';
import type { CanvasEntityIdentifier, ControlModeV2 } from 'features/controlLayers/store/types';
import { replaceCanvasEntityObjectsWithImage } from 'features/imageActions/actions';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiBoundingBoxBold, PiShootingStarFill, PiUploadBold } from 'react-icons/pi';
import type {
  ControlLoRAModelConfig,
  ControlNetModelConfig,
  ImageDTO,
  T2IAdapterModelConfig,
} from 'services/api/types';

const buildSelectControlAdapter = (entityIdentifier: CanvasEntityIdentifier<'control_layer'>) =>
  createSelector(selectCanvasSlice, (canvas) => {
    const layer = selectEntityOrThrow(canvas, entityIdentifier, 'ControlLayerControlAdapter');
    return layer.controlAdapter;
  });

export const ControlLayerControlAdapter = memo(() => {
  const { t } = useTranslation();
  const { dispatch, getState } = useAppStore();
  const entityIdentifier = useEntityIdentifierContext('control_layer');
  const selectControlAdapter = useMemo(() => buildSelectControlAdapter(entityIdentifier), [entityIdentifier]);
  const controlAdapter = useAppSelector(selectControlAdapter);
  const filter = useEntityFilter(entityIdentifier);
  const isFLUX = useAppSelector(selectIsFLUX);
  const adapter = useEntityAdapterContext('control_layer');

  const onChangeBeginEndStepPct = useCallback(
    (beginEndStepPct: [number, number]) => {
      dispatch(controlLayerBeginEndStepPctChanged({ entityIdentifier, beginEndStepPct }));
    },
    [dispatch, entityIdentifier]
  );

  const onChangeControlMode = useCallback(
    (controlMode: ControlModeV2) => {
      dispatch(controlLayerControlModeChanged({ entityIdentifier, controlMode }));
    },
    [dispatch, entityIdentifier]
  );

  const onChangeWeight = useCallback(
    (weight: number) => {
      dispatch(controlLayerWeightChanged({ entityIdentifier, weight }));
    },
    [dispatch, entityIdentifier]
  );

  const onChangeModel = useCallback(
    (modelConfig: ControlNetModelConfig | T2IAdapterModelConfig | ControlLoRAModelConfig) => {
      dispatch(controlLayerModelChanged({ entityIdentifier, modelConfig }));
      // When we change the model, we need may need to start filtering w/ the simplified filter mode, and/or change the
      // filter config.
      const isFiltering = adapter.filterer.$isFiltering.get();
      const isSimple = adapter.filterer.$simple.get();
      // If we are filtering and _not_ in simple mode, that means the user has clicked Advanced. They want to be in control
      // of the settings. Bail early without doing anything else.
      if (isFiltering && !isSimple) {
        return;
      }

      // Else, we are in simple mode and will take care of some things for the user.

      // First, check if the newly-selected model has a default filter. It may not - for example, Tile controlnet models
      // don't have a default filter.
      const defaultFilterForNewModel = getFilterForModel(modelConfig);

      if (!defaultFilterForNewModel) {
        // The user has chosen a model that doesn't have a default filter - cancel any in-progress filtering and bail.
        if (isFiltering) {
          adapter.filterer.cancel();
        }
        return;
      }

      // At this point, we know the user has selected a model that has a default filter. We need to either start filtering
      // with that default filter, or update the existing filter config to match the new model's default filter.
      const filterConfig = defaultFilterForNewModel.buildDefaults();
      if (isFiltering) {
        adapter.filterer.$filterConfig.set(filterConfig);
        // The user may have disabled auto-processing, so we should process the filter manually. This is essentially a
        // no-op if auto-processing is already enabled, because the process method is debounced.
        adapter.filterer.process();
      } else {
        adapter.filterer.start(filterConfig);
      }
    },
    [adapter.filterer, dispatch, entityIdentifier]
  );

  const pullBboxIntoLayer = usePullBboxIntoLayer(entityIdentifier);
  const isBusy = useCanvasIsBusy();
  const uploadOptions = useMemo(
    () =>
      ({
        onUpload: (imageDTO: ImageDTO) => {
          replaceCanvasEntityObjectsWithImage({ entityIdentifier, imageDTO, dispatch, getState });
        },
        allowMultiple: false,
      }) as const,
    [dispatch, entityIdentifier, getState]
  );
  const uploadApi = useImageUploadButton(uploadOptions);

  return (
    <Flex flexDir="column" gap={3} position="relative" w="full">
      <Flex w="full" gap={2}>
        <ControlLayerControlAdapterModel modelKey={controlAdapter.model?.key ?? null} onChange={onChangeModel} />
        <IconButton
          onClick={filter.start}
          isDisabled={filter.isDisabled}
          size="sm"
          alignSelf="stretch"
          variant="link"
          aria-label={t('controlLayers.filter.filter')}
          tooltip={t('controlLayers.filter.filter')}
          icon={<PiShootingStarFill />}
        />
        <IconButton
          onClick={pullBboxIntoLayer}
          isDisabled={isBusy}
          size="sm"
          alignSelf="stretch"
          variant="link"
          aria-label={t('controlLayers.pullBboxIntoLayer')}
          tooltip={t('controlLayers.pullBboxIntoLayer')}
          icon={<PiBoundingBoxBold />}
        />
        <IconButton
          isDisabled={isBusy}
          size="sm"
          alignSelf="stretch"
          variant="link"
          aria-label={t('accessibility.uploadImage')}
          tooltip={t('accessibility.uploadImage')}
          icon={<PiUploadBold />}
          {...uploadApi.getUploadButtonProps()}
        />
        <input {...uploadApi.getUploadInputProps()} />
      </Flex>
      <Weight weight={controlAdapter.weight} onChange={onChangeWeight} />
      {controlAdapter.type !== 'control_lora' && (
        <BeginEndStepPct beginEndStepPct={controlAdapter.beginEndStepPct} onChange={onChangeBeginEndStepPct} />
      )}
      {controlAdapter.type === 'controlnet' && !isFLUX && (
        <ControlLayerControlAdapterControlMode
          controlMode={controlAdapter.controlMode}
          onChange={onChangeControlMode}
        />
      )}
    </Flex>
  );
});

ControlLayerControlAdapter.displayName = 'ControlLayerControlAdapter';
