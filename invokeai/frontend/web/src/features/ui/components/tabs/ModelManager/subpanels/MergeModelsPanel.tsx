import { Flex, Radio, RadioGroup, Text, Tooltip } from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAIInput from 'common/components/IAIInput';
import IAISelect from 'common/components/IAISelect';
import IAISimpleCheckbox from 'common/components/IAISimpleCheckbox';
import IAISlider from 'common/components/IAISlider';
import { pickBy } from 'lodash-es';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetMainModelsQuery } from 'services/api/endpoints/models';

export default function MergeModelsPanel() {
  const { t } = useTranslation();

  const dispatch = useAppDispatch();

  const { data } = useGetMainModelsQuery();

  const diffusersModels = pickBy(
    data?.entities,
    (value, _) => value?.model_format === 'diffusers'
  );

  const [modelOne, setModelOne] = useState<string>(
    Object.keys(diffusersModels)[0]
  );
  const [modelTwo, setModelTwo] = useState<string>(
    Object.keys(diffusersModels)[1]
  );
  const [modelThree, setModelThree] = useState<string>('none');

  const [mergedModelName, setMergedModelName] = useState<string>('');
  const [modelMergeAlpha, setModelMergeAlpha] = useState<number>(0.5);

  const [modelMergeInterp, setModelMergeInterp] = useState<
    'weighted_sum' | 'sigmoid' | 'inv_sigmoid' | 'add_difference'
  >('weighted_sum');

  const [modelMergeSaveLocType, setModelMergeSaveLocType] = useState<
    'root' | 'custom'
  >('root');

  const [modelMergeCustomSaveLoc, setModelMergeCustomSaveLoc] =
    useState<string>('');

  const [modelMergeForce, setModelMergeForce] = useState<boolean>(false);

  const modelOneList = Object.keys(diffusersModels).filter(
    (model) => model !== modelTwo && model !== modelThree
  );

  const modelTwoList = Object.keys(diffusersModels).filter(
    (model) => model !== modelOne && model !== modelThree
  );

  const modelThreeList = [
    { key: t('modelManager.none'), value: 'none' },
    ...Object.keys(diffusersModels)
      .filter((model) => model !== modelOne && model !== modelTwo)
      .map((model) => ({ key: model, value: model })),
  ];

  const isProcessing = useAppSelector(
    (state: RootState) => state.system.isProcessing
  );

  const mergeModelsHandler = () => {
    let modelsToMerge: string[] = [modelOne, modelTwo, modelThree];
    modelsToMerge = modelsToMerge.filter((model) => model !== 'none');

    const mergeModelsInfo: InvokeAI.InvokeModelMergingProps = {
      models_to_merge: modelsToMerge,
      merged_model_name:
        mergedModelName !== '' ? mergedModelName : modelsToMerge.join('-'),
      alpha: modelMergeAlpha,
      interp: modelMergeInterp,
      model_merge_save_path:
        modelMergeSaveLocType === 'root' ? null : modelMergeCustomSaveLoc,
      force: modelMergeForce,
    };

    dispatch(mergeDiffusersModels(mergeModelsInfo));
  };

  return (
    <Flex flexDirection="column" rowGap={4}>
      <Flex
        sx={{
          flexDirection: 'column',
          rowGap: 1,
          bg: 'base.900',
        }}
      >
        <Text>{t('modelManager.modelMergeHeaderHelp1')}</Text>
        <Text fontSize="sm" variant="subtext">
          {t('modelManager.modelMergeHeaderHelp2')}
        </Text>
      </Flex>
      <Flex columnGap={4}>
        <IAISelect
          label={t('modelManager.modelOne')}
          validValues={modelOneList}
          onChange={(e) => setModelOne(e.target.value)}
        />
        <IAISelect
          label={t('modelManager.modelTwo')}
          validValues={modelTwoList}
          onChange={(e) => setModelTwo(e.target.value)}
        />
        <IAISelect
          label={t('modelManager.modelThree')}
          validValues={modelThreeList}
          onChange={(e) => {
            if (e.target.value !== 'none') {
              setModelThree(e.target.value);
              setModelMergeInterp('add_difference');
            } else {
              setModelThree('none');
              setModelMergeInterp('weighted_sum');
            }
          }}
        />
      </Flex>

      <IAIInput
        label={t('modelManager.mergedModelName')}
        value={mergedModelName}
        onChange={(e) => setMergedModelName(e.target.value)}
      />

      <Flex
        sx={{
          flexDirection: 'column',
          padding: 4,
          borderRadius: 'base',
          gap: 4,
          bg: 'base.900',
        }}
      >
        <IAISlider
          label={t('modelManager.alpha')}
          min={0.01}
          max={0.99}
          step={0.01}
          value={modelMergeAlpha}
          onChange={(v) => setModelMergeAlpha(v)}
          withInput
          withReset
          handleReset={() => setModelMergeAlpha(0.5)}
          withSliderMarks
        />
        <Text variant="subtext" fontSize="sm">
          {t('modelManager.modelMergeAlphaHelp')}
        </Text>
      </Flex>

      <Flex
        sx={{
          padding: 4,
          borderRadius: 'base',
          gap: 4,
          bg: 'base.900',
        }}
      >
        <Text fontWeight={500} fontSize="sm" variant="subtext">
          {t('modelManager.interpolationType')}
        </Text>
        <RadioGroup
          value={modelMergeInterp}
          onChange={(
            v: 'weighted_sum' | 'sigmoid' | 'inv_sigmoid' | 'add_difference'
          ) => setModelMergeInterp(v)}
        >
          <Flex columnGap={4}>
            {modelThree === 'none' ? (
              <>
                <Radio value="weighted_sum">
                  <Text fontSize="sm">{t('modelManager.weightedSum')}</Text>
                </Radio>
                <Radio value="sigmoid">
                  <Text fontSize="sm">{t('modelManager.sigmoid')}</Text>
                </Radio>
                <Radio value="inv_sigmoid">
                  <Text fontSize="sm">{t('modelManager.inverseSigmoid')}</Text>
                </Radio>
              </>
            ) : (
              <Radio value="add_difference">
                <Tooltip
                  label={t('modelManager.modelMergeInterpAddDifferenceHelp')}
                >
                  <Text fontSize="sm">{t('modelManager.addDifference')}</Text>
                </Tooltip>
              </Radio>
            )}
          </Flex>
        </RadioGroup>
      </Flex>

      <Flex
        sx={{
          flexDirection: 'column',
          padding: 4,
          borderRadius: 'base',
          gap: 4,
          bg: 'base.900',
        }}
      >
        <Flex columnGap={4}>
          <Text fontWeight="500" fontSize="sm" variant="subtext">
            {t('modelManager.mergedModelSaveLocation')}
          </Text>
          <RadioGroup
            value={modelMergeSaveLocType}
            onChange={(v: 'root' | 'custom') => setModelMergeSaveLocType(v)}
          >
            <Flex columnGap={4}>
              <Radio value="root">
                <Text fontSize="sm">{t('modelManager.invokeAIFolder')}</Text>
              </Radio>

              <Radio value="custom">
                <Text fontSize="sm">{t('modelManager.custom')}</Text>
              </Radio>
            </Flex>
          </RadioGroup>
        </Flex>

        {modelMergeSaveLocType === 'custom' && (
          <IAIInput
            label={t('modelManager.mergedModelCustomSaveLocation')}
            value={modelMergeCustomSaveLoc}
            onChange={(e) => setModelMergeCustomSaveLoc(e.target.value)}
          />
        )}
      </Flex>

      <IAISimpleCheckbox
        label={t('modelManager.ignoreMismatch')}
        isChecked={modelMergeForce}
        onChange={(e) => setModelMergeForce(e.target.checked)}
        fontWeight="500"
      />

      <IAIButton
        onClick={mergeModelsHandler}
        isLoading={isProcessing}
        isDisabled={
          modelMergeSaveLocType === 'custom' && modelMergeCustomSaveLoc === ''
        }
      >
        {t('modelManager.merge')}
      </IAIButton>
    </Flex>
  );
}
