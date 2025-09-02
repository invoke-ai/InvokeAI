import { Flex, IconButton, Spacer, Text } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { BeginEndStepPct } from 'features/controlLayers/components/common/BeginEndStepPct';
import { FLUXReduxImageInfluence } from 'features/controlLayers/components/common/FLUXReduxImageInfluence';
import { IPAdapterCLIPVisionModel } from 'features/controlLayers/components/common/IPAdapterCLIPVisionModel';
import { Weight } from 'features/controlLayers/components/common/Weight';
import { IPAdapterMethod } from 'features/controlLayers/components/RefImage/IPAdapterMethod';
import { RefImageImage } from 'features/controlLayers/components/RefImage/RefImageImage';
import { RegionalGuidanceIPAdapterSettingsEmptyState } from 'features/controlLayers/components/RegionalGuidance/RegionalGuidanceIPAdapterSettingsEmptyState';
import { RegionalReferenceImageModel } from 'features/controlLayers/components/RegionalGuidance/RegionalReferenceImageModel';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { usePullBboxIntoRegionalGuidanceReferenceImage } from 'features/controlLayers/hooks/saveCanvasHooks';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import {
  rgRefImageDeleted,
  rgRefImageFLUXReduxImageInfluenceChanged,
  rgRefImageImageChanged,
  rgRefImageIPAdapterBeginEndStepPctChanged,
  rgRefImageIPAdapterCLIPVisionModelChanged,
  rgRefImageIPAdapterMethodChanged,
  rgRefImageIPAdapterWeightChanged,
  rgRefImageModelChanged,
} from 'features/controlLayers/store/canvasInstanceSlice';
import { selectCanvasSlice, selectRegionalGuidanceReferenceImage } from 'features/controlLayers/store/selectors';
import type {
  CanvasEntityIdentifier,
  CLIPVisionModelV2,
  FLUXReduxImageInfluence as FLUXReduxImageInfluenceType,
  IPMethodV2,
} from 'features/controlLayers/store/types';
import type { SetRegionalGuidanceReferenceImageDndTargetData } from 'features/dnd/dnd';
import { setRegionalGuidanceReferenceImageDndTarget } from 'features/dnd/dnd';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiBoundingBoxBold, PiXBold } from 'react-icons/pi';
import type { FLUXReduxModelConfig, ImageDTO, IPAdapterModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

type Props = {
  referenceImageId: string;
};

const RegionalGuidanceIPAdapterSettingsContent = memo(({ referenceImageId }: Props) => {
  const entityIdentifier = useEntityIdentifierContext('regional_guidance');
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const onDeleteIPAdapter = useCallback(() => {
    dispatch(rgRefImageDeleted({ entityIdentifier, referenceImageId }));
  }, [dispatch, entityIdentifier, referenceImageId]);
  const selectConfig = useMemo(
    () =>
      createSelector(selectCanvasSlice, (canvas) => {
        if (!canvas) {
return null;
}
        const referenceImage = selectRegionalGuidanceReferenceImage(canvas, entityIdentifier, referenceImageId);
        assert(referenceImage, `Regional Guidance IP Adapter with id ${referenceImageId} not found`);
        return referenceImage.config;
      }),
    [entityIdentifier, referenceImageId]
  );
  const config = useAppSelector(selectConfig);

  const onChangeBeginEndStepPct = useCallback(
    (beginEndStepPct: [number, number]) => {
      dispatch(rgRefImageIPAdapterBeginEndStepPctChanged({ entityIdentifier, referenceImageId, beginEndStepPct }));
    },
    [dispatch, entityIdentifier, referenceImageId]
  );

  const onChangeWeight = useCallback(
    (weight: number) => {
      dispatch(rgRefImageIPAdapterWeightChanged({ entityIdentifier, referenceImageId, weight }));
    },
    [dispatch, entityIdentifier, referenceImageId]
  );

  const onChangeIPMethod = useCallback(
    (method: IPMethodV2) => {
      dispatch(rgRefImageIPAdapterMethodChanged({ entityIdentifier, referenceImageId, method }));
    },
    [dispatch, entityIdentifier, referenceImageId]
  );

  const onChangeFLUXReduxImageInfluence = useCallback(
    (imageInfluence: FLUXReduxImageInfluenceType) => {
      dispatch(rgRefImageFLUXReduxImageInfluenceChanged({ entityIdentifier, referenceImageId, imageInfluence }));
    },
    [dispatch, entityIdentifier, referenceImageId]
  );

  const onChangeModel = useCallback(
    (modelConfig: IPAdapterModelConfig | FLUXReduxModelConfig) => {
      dispatch(rgRefImageModelChanged({ entityIdentifier, referenceImageId, modelConfig }));
    },
    [dispatch, entityIdentifier, referenceImageId]
  );

  const onChangeCLIPVisionModel = useCallback(
    (clipVisionModel: CLIPVisionModelV2) => {
      dispatch(rgRefImageIPAdapterCLIPVisionModelChanged({ entityIdentifier, referenceImageId, clipVisionModel }));
    },
    [dispatch, entityIdentifier, referenceImageId]
  );

  const onChangeImage = useCallback(
    (imageDTO: ImageDTO | null) => {
      dispatch(rgRefImageImageChanged({ entityIdentifier, referenceImageId, imageDTO }));
    },
    [dispatch, entityIdentifier, referenceImageId]
  );

  const dndTargetData = useMemo<SetRegionalGuidanceReferenceImageDndTargetData>(
    () =>
      setRegionalGuidanceReferenceImageDndTarget.getData(
        { entityIdentifier, referenceImageId },
        config?.image?.image_name
      ),
    [entityIdentifier, config?.image?.image_name, referenceImageId]
  );

  const pullBboxIntoIPAdapter = usePullBboxIntoRegionalGuidanceReferenceImage(entityIdentifier, referenceImageId);
  const isBusy = useCanvasIsBusy();

  if (!config) {
    return null;
  }

  return (
    <Flex flexDir="column" gap={2}>
      <Flex alignItems="center" gap={2}>
        <Text fontWeight="semibold" color="base.400">
          {t('controlLayers.referenceImage')}
        </Text>
        <Spacer />
        <IconButton
          size="sm"
          variant="link"
          alignSelf="stretch"
          icon={<PiXBold />}
          tooltip={t('controlLayers.deleteReferenceImage')}
          aria-label={t('controlLayers.deleteReferenceImage')}
          onClick={onDeleteIPAdapter}
          colorScheme="error"
        />
      </Flex>
      <Flex flexDir="column" gap={2} position="relative" w="full">
        <Flex gap={2} alignItems="center" w="full">
          <RegionalReferenceImageModel modelKey={config.model?.key ?? null} onChangeModel={onChangeModel} />
          {config.type === 'ip_adapter' && (
            <IPAdapterCLIPVisionModel model={config.clipVisionModel} onChange={onChangeCLIPVisionModel} />
          )}
          <IconButton
            onClick={pullBboxIntoIPAdapter}
            isDisabled={isBusy}
            variant="ghost"
            aria-label={t('controlLayers.pullBboxIntoReferenceImage')}
            tooltip={t('controlLayers.pullBboxIntoReferenceImage')}
            icon={<PiBoundingBoxBold />}
          />
        </Flex>
        <Flex gap={2} w="full">
          {config.type === 'ip_adapter' && (
            <Flex flexDir="column" gap={2} w="full">
              <IPAdapterMethod method={config.method} onChange={onChangeIPMethod} />
              <Weight weight={config.weight} onChange={onChangeWeight} />
              <BeginEndStepPct beginEndStepPct={config.beginEndStepPct} onChange={onChangeBeginEndStepPct} />
            </Flex>
          )}
          {config.type === 'flux_redux' && (
            <Flex flexDir="column" gap={2} w="full">
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
              dndTarget={setRegionalGuidanceReferenceImageDndTarget}
              dndTargetData={dndTargetData}
            />
          </Flex>
        </Flex>
      </Flex>
    </Flex>
  );
});

RegionalGuidanceIPAdapterSettingsContent.displayName = 'RegionalGuidanceIPAdapterSettingsContent';

const buildSelectIPAdapterHasImage = (
  entityIdentifier: CanvasEntityIdentifier<'regional_guidance'>,
  referenceImageId: string
) =>
  createSelector(selectCanvasSlice, (canvas) => {
    if (!canvas) {
return false;
}
    const referenceImage = selectRegionalGuidanceReferenceImage(canvas, entityIdentifier, referenceImageId);
    return !!referenceImage && referenceImage.config.image !== null;
  });

export const RegionalGuidanceIPAdapterSettings = memo(({ referenceImageId }: Props) => {
  const entityIdentifier = useEntityIdentifierContext('regional_guidance');
  const selectHasImage = useMemo(
    () => buildSelectIPAdapterHasImage(entityIdentifier, referenceImageId),
    [entityIdentifier, referenceImageId]
  );
  const hasImage = useAppSelector(selectHasImage);

  if (!hasImage) {
    return <RegionalGuidanceIPAdapterSettingsEmptyState referenceImageId={referenceImageId} />;
  }

  return <RegionalGuidanceIPAdapterSettingsContent referenceImageId={referenceImageId} />;
});

RegionalGuidanceIPAdapterSettings.displayName = 'RegionalGuidanceIPAdapterSettings';
