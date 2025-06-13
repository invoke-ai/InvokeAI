import { Flex } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { BeginEndStepPct } from 'features/controlLayers/components/common/BeginEndStepPct';
import { FLUXReduxImageInfluence } from 'features/controlLayers/components/common/FLUXReduxImageInfluence';
import { IPAdapterCLIPVisionModel } from 'features/controlLayers/components/common/IPAdapterCLIPVisionModel';
import { Weight } from 'features/controlLayers/components/common/Weight';
import { IPAdapterMethod } from 'features/controlLayers/components/RefImage/IPAdapterMethod';
import { RefImageModel } from 'features/controlLayers/components/RefImage/RefImageModel';
import { RefImageNoImageState } from 'features/controlLayers/components/RefImage/RefImageNoImageState';
import { useRefImageIdContext } from 'features/controlLayers/contexts/RefImageIdContext';
import { selectIsFLUX } from 'features/controlLayers/store/paramsSlice';
import {
  refImageFLUXReduxImageInfluenceChanged,
  refImageImageChanged,
  refImageIPAdapterBeginEndStepPctChanged,
  refImageIPAdapterCLIPVisionModelChanged,
  refImageIPAdapterMethodChanged,
  refImageIPAdapterWeightChanged,
  refImageModelChanged,
  selectRefImageEntity,
  selectRefImageEntityOrThrow,
  selectRefImagesSlice,
} from 'features/controlLayers/store/refImagesSlice';
import {
  type CLIPVisionModelV2,
  type FLUXReduxImageInfluence as FLUXReduxImageInfluenceType,
  type IPMethodV2,
  isFLUXReduxConfig,
  isIPAdapterConfig,
} from 'features/controlLayers/store/types';
import type { SetGlobalReferenceImageDndTargetData } from 'features/dnd/dnd';
import { setGlobalReferenceImageDndTarget } from 'features/dnd/dnd';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import type { ApiModelConfig, FLUXReduxModelConfig, ImageDTO, IPAdapterModelConfig } from 'services/api/types';

import { RefImageImage } from './RefImageImage';

const buildSelectConfig = (id: string) =>
  createSelector(
    selectRefImagesSlice,
    (refImages) => selectRefImageEntityOrThrow(refImages, id, 'IPAdapterSettings').config
  );

const IPAdapterSettingsContent = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const id = useRefImageIdContext();
  const selectConfig = useMemo(() => buildSelectConfig(id), [id]);
  const config = useAppSelector(selectConfig);

  const onChangeBeginEndStepPct = useCallback(
    (beginEndStepPct: [number, number]) => {
      dispatch(refImageIPAdapterBeginEndStepPctChanged({ id, beginEndStepPct }));
    },
    [dispatch, id]
  );

  const onChangeWeight = useCallback(
    (weight: number) => {
      dispatch(refImageIPAdapterWeightChanged({ id, weight }));
    },
    [dispatch, id]
  );

  const onChangeIPMethod = useCallback(
    (method: IPMethodV2) => {
      dispatch(refImageIPAdapterMethodChanged({ id, method }));
    },
    [dispatch, id]
  );

  const onChangeFLUXReduxImageInfluence = useCallback(
    (imageInfluence: FLUXReduxImageInfluenceType) => {
      dispatch(refImageFLUXReduxImageInfluenceChanged({ id, imageInfluence }));
    },
    [dispatch, id]
  );

  const onChangeModel = useCallback(
    (modelConfig: IPAdapterModelConfig | FLUXReduxModelConfig | ApiModelConfig) => {
      dispatch(refImageModelChanged({ id, modelConfig }));
    },
    [dispatch, id]
  );

  const onChangeCLIPVisionModel = useCallback(
    (clipVisionModel: CLIPVisionModelV2) => {
      dispatch(refImageIPAdapterCLIPVisionModelChanged({ id, clipVisionModel }));
    },
    [dispatch, id]
  );

  const onChangeImage = useCallback(
    (imageDTO: ImageDTO | null) => {
      dispatch(refImageImageChanged({ id, imageDTO }));
    },
    [dispatch, id]
  );

  const dndTargetData = useMemo<SetGlobalReferenceImageDndTargetData>(
    () => setGlobalReferenceImageDndTarget.getData({ id }, config.image?.image_name),
    [id, config.image?.image_name]
  );
  // const pullBboxIntoIPAdapter = usePullBboxIntoGlobalReferenceImage(id);
  // const isBusy = useCanvasIsBusy();

  const isFLUX = useAppSelector(selectIsFLUX);

  return (
    <Flex flexDir="column" gap={2} position="relative" w="full">
      <Flex gap={2} alignItems="center" w="full">
        <RefImageModel modelKey={config.model?.key ?? null} onChangeModel={onChangeModel} />
        {isIPAdapterConfig(config) && (
          <IPAdapterCLIPVisionModel model={config.clipVisionModel} onChange={onChangeCLIPVisionModel} />
        )}
        {/* <IconButton
            onClick={pullBboxIntoIPAdapter}
            isDisabled={isBusy}
            variant="ghost"
            aria-label={t('controlLayers.pullBboxIntoReferenceImage')}
            tooltip={t('controlLayers.pullBboxIntoReferenceImage')}
            icon={<PiBoundingBoxBold />}
          /> */}
      </Flex>
      <Flex gap={2} w="full">
        {isIPAdapterConfig(config) && (
          <Flex flexDir="column" gap={2} w="full">
            {!isFLUX && <IPAdapterMethod method={config.method} onChange={onChangeIPMethod} />}
            <Weight weight={config.weight} onChange={onChangeWeight} />
            <BeginEndStepPct beginEndStepPct={config.beginEndStepPct} onChange={onChangeBeginEndStepPct} />
          </Flex>
        )}
        {isFLUXReduxConfig(config) && (
          <Flex flexDir="column" gap={2} w="full" alignItems="flex-start">
            <FLUXReduxImageInfluence
              imageInfluence={config.imageInfluence ?? 'lowest'}
              onChange={onChangeFLUXReduxImageInfluence}
            />
          </Flex>
        )}
        <Flex alignItems="center" justifyContent="center" h={32} w={32} aspectRatio="1/1" flexGrow={1}>
          <RefImageImage
            image={config.image}
            onChangeImage={onChangeImage}
            dndTarget={setGlobalReferenceImageDndTarget}
            dndTargetData={dndTargetData}
          />
        </Flex>
      </Flex>
    </Flex>
  );
});

IPAdapterSettingsContent.displayName = 'IPAdapterSettingsContent';

const buildSelectIPAdapterHasImage = (id: string) =>
  createSelector(selectRefImagesSlice, (refImages) => {
    const referenceImage = selectRefImageEntity(refImages, id);
    return !!referenceImage && referenceImage.config.image !== null;
  });

export const IPAdapterSettings = memo(() => {
  const id = useRefImageIdContext();

  const selectIPAdapterHasImage = useMemo(() => buildSelectIPAdapterHasImage(id), [id]);
  const hasImage = useAppSelector(selectIPAdapterHasImage);

  if (!hasImage) {
    return <RefImageNoImageState />;
  }

  return <IPAdapterSettingsContent />;
});

IPAdapterSettings.displayName = 'IPAdapterSettings';
