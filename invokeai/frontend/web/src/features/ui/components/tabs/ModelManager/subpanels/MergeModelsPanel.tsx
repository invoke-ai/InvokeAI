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
import { ChangeEvent, useCallback, useMemo, useState } from 'react';
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

  const handleBaseModelChange = useCallback((v: string) => {
    setBaseModel(v as BaseModelType);
    setModelOne(null);
    setModelTwo(null);
  }, []);

  const handleChangeModelOne = useCallback((v: string) => {
    setModelOne(v);
  }, []);
  const handleChangeModelTwo = useCallback((v: string) => {
    setModelTwo(v);
  }, []);
  const handleChangeModelThree = useCallback((v: string) => {
    if (!v) {
      setModelThree(null);
      setModelMergeInterp('add_difference');
    } else {
      setModelThree(v);
      setModelMergeInterp('weighted_sum');
    }
  }, []);
  const handleChangeMergedModelName = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => setMergedModelName(e.target.value),
    []
  );
  const handleChangeModelMergeAlpha = useCallback(
    (v: number) => setModelMergeAlpha(v),
    []
  );
  const handleResetModelMergeAlpha = useCallback(
    () => setModelMergeAlpha(0.5),
    []
  );
  const handleChangeMergeInterp = useCallback(
    (v: MergeInterpolationMethods) => setModelMergeInterp(v),
    []
  );
  const handleChangeMergeSaveLocType = useCallback(
    (v: 'root' | 'custom') => setModelMergeSaveLocType(v),
    []
  );
  const handleChangeMergeCustomSaveLoc = useCallback(
    (e: ChangeEvent<HTMLInputElement>) =>
      setModelMergeCustomSaveLoc(e.target.value),
    []
  );
  const handleChangeModelMergeForce = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => setModelMergeForce(e.target.checked),
    []
  );

  const mergeModelsHandler = useCallback(() => {
    const models_names: string[] = [];

    let modelsToMerge: (string | null)[] = [modelOne, modelTwo, modelThree];
    modelsToMerge = modelsToMerge.filter((model) => model !== null);
    modelsToMerge.forEach((model) => {
      const n = model?.split('/')?.[2];
      if (n) {
        models_names.push(n);
      }
    });

    const mergeModelsInfo: MergeModelConfig['body'] = {
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
      body: { body: mergeModelsInfo },
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
  }, [
    baseModel,
    dispatch,
    mergeModels,
    mergedModelName,
    modelMergeAlpha,
    modelMergeCustomSaveLoc,
    modelMergeForce,
    modelMergeInterp,
    modelMergeSaveLocType,
    modelOne,
    modelThree,
    modelTwo,
    t,
  ]);

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
          onChange={handleChangeModelOne}
        />
        <IAIMantineSearchableSelect
          label={t('modelManager.modelTwo')}
          w="100%"
          placeholder={t('modelManager.selectModel')}
          value={modelTwo}
          data={modelTwoList}
          onChange={handleChangeModelTwo}
        />
        <IAIMantineSearchableSelect
          label={t('modelManager.modelThree')}
          data={modelThreeList}
          w="100%"
          placeholder={t('modelManager.selectModel')}
          clearable
          onChange={handleChangeModelThree}
        />
      </Flex>

      <IAIInput
        label={t('modelManager.mergedModelName')}
        value={mergedModelName}
        onChange={handleChangeMergedModelName}
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
          onChange={handleChangeModelMergeAlpha}
          withInput
          withReset
          handleReset={handleResetModelMergeAlpha}
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
        <RadioGroup value={modelMergeInterp} onChange={handleChangeMergeInterp}>
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
            onChange={handleChangeMergeSaveLocType}
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
            onChange={handleChangeMergeCustomSaveLoc}
          />
        )}
      </Flex>

      <IAISimpleCheckbox
        label={t('modelManager.ignoreMismatch')}
        isChecked={modelMergeForce}
        onChange={handleChangeModelMergeForce}
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
