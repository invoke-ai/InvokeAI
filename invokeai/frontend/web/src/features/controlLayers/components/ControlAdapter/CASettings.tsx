import { Box, Divider, Flex, Icon, IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { BeginEndStepPct } from 'features/controlLayers/components/common/BeginEndStepPct';
import { Weight } from 'features/controlLayers/components/common/Weight';
import { CAControlModeSelect } from 'features/controlLayers/components/ControlAdapter/CAControlModeSelect';
import { CAImagePreview } from 'features/controlLayers/components/ControlAdapter/CAImagePreview';
import { CAModelCombobox } from 'features/controlLayers/components/ControlAdapter/CAModelCombobox';
import { CAProcessorConfig } from 'features/controlLayers/components/ControlAdapter/CAProcessorConfig';
import { CAProcessorTypeSelect } from 'features/controlLayers/components/ControlAdapter/CAProcessorTypeSelect';
import {
  caBeginEndStepPctChanged,
  caControlModeChanged,
  caImageChanged,
  caModelChanged,
  caProcessedImageChanged,
  caProcessorConfigChanged,
  caWeightChanged,
  selectCAOrThrow,
} from 'features/controlLayers/store/controlAdaptersSlice';
import type { ControlModeV2, ProcessorConfig } from 'features/controlLayers/store/types';
import type { CAImageDropData } from 'features/dnd/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretUpBold } from 'react-icons/pi';
import { useToggle } from 'react-use';
import type {
  CAImagePostUploadAction,
  ControlNetModelConfig,
  ImageDTO,
  T2IAdapterModelConfig,
} from 'services/api/types';

type Props = {
  id: string;
};

export const CASettings = memo(({ id }: Props) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const [isExpanded, toggleIsExpanded] = useToggle(false);

  const controlAdapter = useAppSelector((s) => selectCAOrThrow(s.controlAdaptersV2, id));

  const onChangeBeginEndStepPct = useCallback(
    (beginEndStepPct: [number, number]) => {
      dispatch(caBeginEndStepPctChanged({ id, beginEndStepPct }));
    },
    [dispatch, id]
  );

  const onChangeControlMode = useCallback(
    (controlMode: ControlModeV2) => {
      dispatch(caControlModeChanged({ id, controlMode }));
    },
    [dispatch, id]
  );

  const onChangeWeight = useCallback(
    (weight: number) => {
      dispatch(caWeightChanged({ id, weight }));
    },
    [dispatch, id]
  );

  const onChangeProcessorConfig = useCallback(
    (processorConfig: ProcessorConfig | null) => {
      dispatch(caProcessorConfigChanged({ id, processorConfig }));
    },
    [dispatch, id]
  );

  const onChangeModel = useCallback(
    (modelConfig: ControlNetModelConfig | T2IAdapterModelConfig) => {
      dispatch(caModelChanged({ id, modelConfig }));
    },
    [dispatch, id]
  );

  const onChangeImage = useCallback(
    (imageDTO: ImageDTO | null) => {
      dispatch(caImageChanged({ id, imageDTO }));
    },
    [dispatch, id]
  );

  const onErrorLoadingImage = useCallback(() => {
    dispatch(caImageChanged({ id, imageDTO: null }));
  }, [dispatch, id]);

  const onErrorLoadingProcessedImage = useCallback(() => {
    dispatch(caProcessedImageChanged({ id, imageDTO: null }));
  }, [dispatch, id]);

  const droppableData = useMemo<CAImageDropData>(() => ({ actionType: 'SET_CA_IMAGE', context: { id }, id }), [id]);
  const postUploadAction = useMemo<CAImagePostUploadAction>(() => ({ id, type: 'SET_CA_IMAGE' }), [id]);

  return (
    <Flex flexDir="column" gap={3} position="relative" w="full">
      <Flex gap={3} alignItems="center" w="full">
        <Box minW={0} w="full" transitionProperty="common" transitionDuration="0.1s">
          <CAModelCombobox modelKey={controlAdapter.model?.key ?? null} onChange={onChangeModel} />
        </Box>

        <IconButton
          size="sm"
          tooltip={isExpanded ? t('controlnet.hideAdvanced') : t('controlnet.showAdvanced')}
          aria-label={isExpanded ? t('controlnet.hideAdvanced') : t('controlnet.showAdvanced')}
          onClick={toggleIsExpanded}
          variant="ghost"
          icon={
            <Icon
              boxSize={4}
              as={PiCaretUpBold}
              transform={isExpanded ? 'rotate(0deg)' : 'rotate(180deg)'}
              transitionProperty="common"
              transitionDuration="normal"
            />
          }
        />
      </Flex>
      <Flex gap={3} w="full">
        <Flex flexDir="column" gap={3} w="full" h="full">
          {controlAdapter.controlMode && (
            <CAControlModeSelect controlMode={controlAdapter.controlMode} onChange={onChangeControlMode} />
          )}
          <Weight weight={controlAdapter.weight} onChange={onChangeWeight} />
          <BeginEndStepPct beginEndStepPct={controlAdapter.beginEndStepPct} onChange={onChangeBeginEndStepPct} />
        </Flex>
        <Flex alignItems="center" justifyContent="center" h={36} w={36} aspectRatio="1/1">
          <CAImagePreview
            controlAdapter={controlAdapter}
            onChangeImage={onChangeImage}
            droppableData={droppableData}
            postUploadAction={postUploadAction}
            onErrorLoadingImage={onErrorLoadingImage}
            onErrorLoadingProcessedImage={onErrorLoadingProcessedImage}
          />
        </Flex>
      </Flex>
      {isExpanded && (
        <>
          <Divider />
          <Flex flexDir="column" gap={3} w="full">
            <CAProcessorTypeSelect config={controlAdapter.processorConfig} onChange={onChangeProcessorConfig} />
            <CAProcessorConfig config={controlAdapter.processorConfig} onChange={onChangeProcessorConfig} />
          </Flex>
        </>
      )}
    </Flex>
  );
});

CASettings.displayName = 'CASettings';
