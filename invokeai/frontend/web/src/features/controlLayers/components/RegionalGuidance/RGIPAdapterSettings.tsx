import { Box, Flex, IconButton, Spacer, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { BeginEndStepPct } from 'features/controlLayers/components/common/BeginEndStepPct';
import { Weight } from 'features/controlLayers/components/common/Weight';
import { IPAImagePreview } from 'features/controlLayers/components/IPAdapter/IPAImagePreview';
import { IPAMethod } from 'features/controlLayers/components/IPAdapter/IPAMethod';
import { IPAModelCombobox } from 'features/controlLayers/components/IPAdapter/IPAModelCombobox';
import {
  rgIPAdapterBeginEndStepPctChanged,
  rgIPAdapterCLIPVisionModelChanged,
  rgIPAdapterDeleted,
  rgIPAdapterImageChanged,
  rgIPAdapterMethodChanged,
  rgIPAdapterModelChanged,
  rgIPAdapterWeightChanged,
  selectRGOrThrow,
} from 'features/controlLayers/store/regionalGuidanceSlice';
import type { CLIPVisionModelV2, IPMethodV2 } from 'features/controlLayers/store/types';
import type { RGIPAdapterImageDropData } from 'features/dnd/types';
import { memo, useCallback, useMemo } from 'react';
import { PiTrashSimpleBold } from 'react-icons/pi';
import type { ImageDTO, IPAdapterModelConfig, RGIPAdapterImagePostUploadAction } from 'services/api/types';
import { assert } from 'tsafe';

type Props = {
  id: string;
  ipAdapterId: string;
  ipAdapterNumber: number;
};

export const RGIPAdapterSettings = memo(({ id, ipAdapterId, ipAdapterNumber }: Props) => {
  const dispatch = useAppDispatch();
  const onDeleteIPAdapter = useCallback(() => {
    dispatch(rgIPAdapterDeleted({ id, ipAdapterId }));
  }, [dispatch, ipAdapterId, id]);
  const ipAdapter = useAppSelector((s) => {
    const ipa = selectRGOrThrow(s.regionalGuidance, id).ipAdapters.find((ipa) => ipa.id === ipAdapterId);
    assert(ipa, `Regional GuidanceIP Adapter with id ${ipAdapterId} not found`);
    return ipa;
  });

  const onChangeBeginEndStepPct = useCallback(
    (beginEndStepPct: [number, number]) => {
      dispatch(rgIPAdapterBeginEndStepPctChanged({ id, ipAdapterId, beginEndStepPct }));
    },
    [dispatch, ipAdapterId, id]
  );

  const onChangeWeight = useCallback(
    (weight: number) => {
      dispatch(rgIPAdapterWeightChanged({ id, ipAdapterId, weight }));
    },
    [dispatch, ipAdapterId, id]
  );

  const onChangeIPMethod = useCallback(
    (method: IPMethodV2) => {
      dispatch(rgIPAdapterMethodChanged({ id, ipAdapterId, method }));
    },
    [dispatch, ipAdapterId, id]
  );

  const onChangeModel = useCallback(
    (modelConfig: IPAdapterModelConfig) => {
      dispatch(rgIPAdapterModelChanged({ id, ipAdapterId, modelConfig }));
    },
    [dispatch, ipAdapterId, id]
  );

  const onChangeCLIPVisionModel = useCallback(
    (clipVisionModel: CLIPVisionModelV2) => {
      dispatch(rgIPAdapterCLIPVisionModelChanged({ id, ipAdapterId, clipVisionModel }));
    },
    [dispatch, ipAdapterId, id]
  );

  const onChangeImage = useCallback(
    (imageDTO: ImageDTO | null) => {
      dispatch(rgIPAdapterImageChanged({ id, ipAdapterId, imageDTO }));
    },
    [dispatch, ipAdapterId, id]
  );

  const droppableData = useMemo<RGIPAdapterImageDropData>(
    () => ({ actionType: 'SET_RG_IP_ADAPTER_IMAGE', context: { id, ipAdapterId }, id }),
    [ipAdapterId, id]
  );
  const postUploadAction = useMemo<RGIPAdapterImagePostUploadAction>(
    () => ({ type: 'SET_RG_IP_ADAPTER_IMAGE', id, ipAdapterId }),
    [ipAdapterId, id]
  );

  return (
    <Flex flexDir="column" gap={3}>
      <Flex alignItems="center" gap={3}>
        <Text fontWeight="semibold" color="base.400">{`IP Adapter ${ipAdapterNumber}`}</Text>
        <Spacer />
        <IconButton
          size="sm"
          icon={<PiTrashSimpleBold />}
          aria-label="Delete IP Adapter"
          onClick={onDeleteIPAdapter}
          variant="ghost"
          colorScheme="error"
        />
      </Flex>
      <Flex flexDir="column" gap={4} position="relative" w="full">
        <Flex gap={3} alignItems="center" w="full">
          <Box minW={0} w="full" transitionProperty="common" transitionDuration="0.1s">
            <IPAModelCombobox
              modelKey={ipAdapter.model?.key ?? null}
              onChangeModel={onChangeModel}
              clipVisionModel={ipAdapter.clipVisionModel}
              onChangeCLIPVisionModel={onChangeCLIPVisionModel}
            />
          </Box>
        </Flex>
        <Flex gap={4} w="full" alignItems="center">
          <Flex flexDir="column" gap={3} w="full">
            <IPAMethod method={ipAdapter.method} onChange={onChangeIPMethod} />
            <Weight weight={ipAdapter.weight} onChange={onChangeWeight} />
            <BeginEndStepPct beginEndStepPct={ipAdapter.beginEndStepPct} onChange={onChangeBeginEndStepPct} />
          </Flex>
          <Flex alignItems="center" justifyContent="center" h={36} w={36} aspectRatio="1/1">
            <IPAImagePreview
              image={ipAdapter.image}
              onChangeImage={onChangeImage}
              ipAdapterId={ipAdapter.id}
              droppableData={droppableData}
              postUploadAction={postUploadAction}
            />
          </Flex>
        </Flex>
      </Flex>
    </Flex>
  );
});

RGIPAdapterSettings.displayName = 'RGIPAdapterSettings';
