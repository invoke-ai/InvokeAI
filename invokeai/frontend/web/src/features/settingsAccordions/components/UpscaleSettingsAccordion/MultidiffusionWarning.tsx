import { Flex, Link, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from '../../../../app/store/storeHooks';
import { useControlNetModels } from '../../../../services/api/hooks/modelsByType';
import { useCallback, useEffect, useMemo } from 'react';
import { tileControlnetModelChanged } from '../../../parameters/store/upscaleSlice';
import { MODEL_TYPE_SHORT_MAP } from '../../../parameters/types/constants';
import { setActiveTab } from '../../../ui/store/uiSlice';

export const MultidiffusionWarning = () => {
  const model = useAppSelector((s) => s.generation.model);
  const { tileControlnetModel, upscaleModel } = useAppSelector((s) => s.upscale);
  const dispatch = useAppDispatch();
  const [modelConfigs, { isLoading }] = useControlNetModels();
  const disabledTabs = useAppSelector((s) => s.config.disabledTabs);
  const shouldShowButton = useMemo(() => !disabledTabs.includes('models'), [disabledTabs]);

  useEffect(() => {
    const validModel = modelConfigs.find((cnetModel) => {
      return cnetModel.base === model?.base && cnetModel.name.toLowerCase().includes('tile');
    });
    dispatch(tileControlnetModelChanged(validModel || null));
  }, [model?.base, modelConfigs, dispatch]);

  const warningText = useMemo(() => {
    if (!model) {
      return `a model`;
    }

    if (!upscaleModel && !tileControlnetModel) {
      return `an upscaler model and ${MODEL_TYPE_SHORT_MAP[model.base]} tile controlnet`;
    }
    if (!upscaleModel) {
      return 'an upscaler model';
    }
    if (!tileControlnetModel) {
      return `a ${MODEL_TYPE_SHORT_MAP[model.base]} tile controlnet`;
    }
  }, [model?.base, upscaleModel, tileControlnetModel]);

  const handleGoToModelManager = useCallback(() => {
    dispatch(setActiveTab('models'));
  }, [dispatch]);

  if (!warningText || isLoading || !shouldShowButton) {
    return <></>;
  }

  return (
    <Flex bg="error.500" borderRadius={'base'} padding="2" direction="column">
      <Text fontSize="xs" textAlign="center" display={'inline-block'}>
        Visit{' '}
        <Link fontWeight="bold" onClick={handleGoToModelManager}>
          Model Manager
        </Link>{' '}
        to install {warningText} required by this feature
      </Text>
    </Flex>
  );
};
