import { Flex, IconButton } from '@invoke-ai/ui-library';
import { createMemoizedAppSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useImageUploadButton } from 'common/hooks/useImageUploadButton';
import { BeginEndStepPct } from 'features/controlLayers/components/common/BeginEndStepPct';
import { Weight } from 'features/controlLayers/components/common/Weight';
import { ControlLayerControlAdapterControlMode } from 'features/controlLayers/components/ControlLayer/ControlLayerControlAdapterControlMode';
import { ControlLayerControlAdapterModel } from 'features/controlLayers/components/ControlLayer/ControlLayerControlAdapterModel';
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
import { selectIsFLUX } from 'features/controlLayers/store/paramsSlice';
import { selectCanvasSlice, selectEntityOrThrow } from 'features/controlLayers/store/selectors';
import type { CanvasEntityIdentifier, ControlModeV2 } from 'features/controlLayers/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiBoundingBoxBold, PiShootingStarBold, PiUploadBold } from 'react-icons/pi';
import type { ControlNetModelConfig, PostUploadAction, T2IAdapterModelConfig } from 'services/api/types';

const useControlLayerControlAdapter = (entityIdentifier: CanvasEntityIdentifier<'control_layer'>) => {
  const selectControlAdapter = useMemo(
    () =>
      createMemoizedAppSelector(selectCanvasSlice, (canvas) => {
        const layer = selectEntityOrThrow(canvas, entityIdentifier);
        return layer.controlAdapter;
      }),
    [entityIdentifier]
  );
  const controlAdapter = useAppSelector(selectControlAdapter);
  return controlAdapter;
};

export const ControlLayerControlAdapter = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext('control_layer');
  const controlAdapter = useControlLayerControlAdapter(entityIdentifier);
  const filter = useEntityFilter(entityIdentifier);
  const isFLUX = useAppSelector(selectIsFLUX);

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
    (modelConfig: ControlNetModelConfig | T2IAdapterModelConfig) => {
      dispatch(controlLayerModelChanged({ entityIdentifier, modelConfig }));
    },
    [dispatch, entityIdentifier]
  );

  const pullBboxIntoLayer = usePullBboxIntoLayer(entityIdentifier);
  const isBusy = useCanvasIsBusy();
  const postUploadAction = useMemo<PostUploadAction>(
    () => ({ type: 'REPLACE_LAYER_WITH_IMAGE', entityIdentifier }),
    [entityIdentifier]
  );
  const uploadApi = useImageUploadButton({ postUploadAction });

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
          icon={<PiShootingStarBold />}
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
      <BeginEndStepPct beginEndStepPct={controlAdapter.beginEndStepPct} onChange={onChangeBeginEndStepPct} />
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
