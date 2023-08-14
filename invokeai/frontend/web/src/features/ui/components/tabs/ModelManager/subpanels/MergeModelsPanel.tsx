import { Flex, Radio, RadioGroup, Text, Tooltip } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAIInput from 'common/components/IAIInput';
import IAIMantineSearchableSelect from 'common/components/IAIMantineSearchableSelect';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import IAISimpleCheckbox from 'common/components/IAISimpleCheckbox';
import IAISlider from 'common/components/IAISlider';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { pickBy } from 'lodash-es';
import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ALL_BASE_MODELS } from 'services/api/constants';
import {
  useGetMainModelsQuery,
  useMergeMainModelsMutation,
} from 'services/api/endpoints/models';
import { BaseModelType, MergeModelConfig } from 'services/api/types';

const baseModelTypeSelectData = [
  { label: 'Stable Diffusion 1', value: 'sd-1' },
  { label: 'Stable Diffusion 2', value: 'sd-2' },
];

type MergeInterpolationMethods =
  | 'weighted_sum'
  | 'sigmoid'
  | 'inv_sigmoid'
  | 'add_difference';

export default function MergeModelsPanel() {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const { data } = useGetMainModelsQuery(ALL_BASE_MODELS);

  const [mergeModels, { isLoading }] = useMergeMainModelsMutation();

  const [baseModel, setBaseModel] = useState<BaseModelType>('sd-1');

  const sd1DiffusersModels = pickBy(
    data?.entities,
    (value, _) =>
      value?.model_format === 'diffusers' && value?.base_model === 'sd-1'
  );

  const sd2DiffusersModels = pickBy(
    data?.entities,
    (value, _) =>
      value?.model_format === 'diffusers' && value?.base_model === 'sd-2'
  );

  const modelsMap = useMemo(() => {
    return {
      'sd-1': sd1DiffusersModels,
      'sd-2': sd2DiffusersModels,
    };
  }, [sd1DiffusersModels, sd2DiffusersModels]);

  const [modelOne, setModelOne] = useState<string | null>(
    Object.keys(modelsMap[baseModel as keyof typeof modelsMap])?.[0] ?? null
  );
  const [modelTwo, setModelTwo] = useState<string | null>(
    Object.keys(modelsMap[baseModel as keyof typeof modelsMap])?.[1] ?? null
  );

  const [modelThree, setModelThree] = useState<string | null>(null);

  const [mergedModelName, setMergedModelName] = useState<string>('');
  const [modelMergeAlpha, setModelMergeAlpha] = useState<number>(0.5);

  const [modelMergeInterp, setModelMergeInterp] =
    useState<MergeInterpolationMethods>('weighted_sum');

  const [modelMergeSaveLocType, setModelMergeSaveLocType] = useState<
    'root' | 'custom'
  >('root');

  const [modelMergeCustomSaveLoc, setModelMergeCustomSaveLoc] =
    useState<string>('');

  const [modelMergeForce, setModelMergeForce] = useState<boolean>(false);

  const modelOneList = Object.keys(
    modelsMap[baseModel as keyof typeof modelsMap]
  ).filter((model) => model !== modelTwo && model !== modelThree);

  const modelTwoList = Object.keys(
    modelsMap[baseModel as keyof typeof modelsMap]
  ).filter((model) => model !== modelOne && model !== modelThree);

  const modelThreeList = Object.keys(
    modelsMap[baseModel as keyof typeof modelsMap]
  ).filter((model) => model !== modelOne && model !== modelTwo);

  const handleBaseModelChange = (v: string) => {
    setBaseModel(v as BaseModelType);
    setModelOne(null);
    setModelTwo(null);
  };

  const mergeModelsHandler = () => {
    const models_names: string[] = [];

    let modelsToMerge: (string | null)[] = [modelOne, modelTwo, modelThree];
    modelsToMerge = modelsToMerge.filter((model) => model !== null);
    modelsToMerge.forEach((model) => {
      const n = model?.split('/')?.[2];
      if (n) {
        models_names.push(n);
      }
    });

    const mergeModelsInfo: MergeModelConfig = {
      model_names: models_names,
      merged_model_name:
        mergedModelName !== '' ? mergedModelName : models_names.join('-'),
      alpha: modelMergeAlpha,
      interp: modelMergeInterp,
      force: modelMergeForce,
      merge_dest_directory:
        modelMergeSaveLocType === 'root' ? undefined : modelMergeCustomSaveLoc,
    };

    mergeModels({
      base_model: baseModel,
      body: mergeModelsInfo,
    })
      .unwrap()
      .then((_) => {
        dispatch(
          addToast(
            makeToast({
              title: t('modelManager.modelsMerged'),
              status: 'success',
            })
          )
        );
      })
      .catch((error) => {
        if (error) {
          dispatch(
            addToast(
              makeToast({
                title: t('modelManager.modelsMergeFailed'),
                status: 'error',
              })
            )
          );
        }
      });
  };

  return (
    <Flex flexDirection="column" rowGap={4}>
      <Flex
        sx={{
          flexDirection: 'column',
          rowGap: 1,
        }}
      >
        <Text>{t('modelManager.modelMergeHeaderHelp1')}</Text>
        <Text fontSize="sm" variant="subtext">
          {t('modelManager.modelMergeHeaderHelp2')}
        </Text>
      </Flex>

      <Flex columnGap={4}>
        <IAIMantineSelect
          label="Model Type"
          w="100%"
          data={baseModelTypeSelectData}
          value={baseModel}
          onChange={handleBaseModelChange}
        />
        <IAIMantineSearchableSelect
          label={t('modelManager.modelOne')}
          w="100%"
          value={modelOne}
          placeholder={t('modelManager.selectModel')}
          data={modelOneList}
          onChange={(v) => setModelOne(v)}
        />
        <IAIMantineSearchableSelect
          label={t('modelManager.modelTwo')}
          w="100%"
          placeholder={t('modelManager.selectModel')}
          value={modelTwo}
          data={modelTwoList}
          onChange={(v) => setModelTwo(v)}
        />
        <IAIMantineSearchableSelect
          label={t('modelManager.modelThree')}
          data={modelThreeList}
          w="100%"
          placeholder={t('modelManager.selectModel')}
          clearable
          onChange={(v) => {
            if (!v) {
              setModelThree(null);
              setModelMergeInterp('add_difference');
            } else {
              setModelThree(v);
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
          bg: 'base.200',
          _dark: {
            bg: 'base.800',
          },
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
          bg: 'base.200',
          _dark: {
            bg: 'base.800',
          },
        }}
      >
        <Text fontWeight={500} fontSize="sm" variant="subtext">
          {t('modelManager.interpolationType')}
        </Text>
        <RadioGroup
          value={modelMergeInterp}
          onChange={(v: MergeInterpolationMethods) => setModelMergeInterp(v)}
        >
          <Flex columnGap={4}>
            {modelThree === null ? (
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
          bg: 'base.200',
          _dark: {
            bg: 'base.900',
          },
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
        isLoading={isLoading}
        isDisabled={modelOne === null || modelTwo === null}
      >
        {t('modelManager.merge')}
      </IAIButton>
    </Flex>
  );
}
