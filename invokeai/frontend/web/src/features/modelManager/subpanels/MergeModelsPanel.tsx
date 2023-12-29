import { Flex, Radio, RadioGroup } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { InvButton } from 'common/components/InvButton/InvButton';
import { InvCheckbox } from 'common/components/InvCheckbox/wrapper';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvInput } from 'common/components/InvInput/InvInput';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import type {
  InvSelectOnChange,
  InvSelectOption,
} from 'common/components/InvSelect/types';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { InvText } from 'common/components/InvText/wrapper';
import { InvTooltip } from 'common/components/InvTooltip/InvTooltip';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { pickBy } from 'lodash-es';
import type { ChangeEvent } from 'react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ALL_BASE_MODELS } from 'services/api/constants';
import {
  useGetMainModelsQuery,
  useMergeMainModelsMutation,
} from 'services/api/endpoints/models';
import type { BaseModelType, MergeModelConfig } from 'services/api/types';

const baseModelTypeSelectOptions: InvSelectOption[] = [
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
  const valueBaseModel = useMemo(
    () => baseModelTypeSelectOptions.find((o) => o.value === baseModel),
    [baseModel]
  );
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

  const optionsModelOne = useMemo(
    () =>
      Object.keys(modelsMap[baseModel as keyof typeof modelsMap])
        .filter((model) => model !== modelTwo && model !== modelThree)
        .map((model) => ({ label: model, value: model })),
    [modelsMap, baseModel, modelTwo, modelThree]
  );

  const optionsModelTwo = useMemo(
    () =>
      Object.keys(modelsMap[baseModel as keyof typeof modelsMap])
        .filter((model) => model !== modelOne && model !== modelThree)
        .map((model) => ({ label: model, value: model })),
    [modelsMap, baseModel, modelOne, modelThree]
  );

  const optionsModelThree = useMemo(
    () =>
      Object.keys(modelsMap[baseModel as keyof typeof modelsMap])
        .filter((model) => model !== modelOne && model !== modelTwo)
        .map((model) => ({ label: model, value: model })),
    [modelsMap, baseModel, modelOne, modelTwo]
  );

  const onChangeBaseModel = useCallback<InvSelectOnChange>((v) => {
    if (!v) {
      return;
    }
    if (!(v.value === 'sd-1' || v.value === 'sd-2')) {
      return;
    }
    setBaseModel(v.value);
    setModelOne(null);
    setModelTwo(null);
  }, []);

  const onChangeModelOne = useCallback<InvSelectOnChange>((v) => {
    if (!v) {
      return;
    }
    setModelOne(v.value);
  }, []);
  const onChangeModelTwo = useCallback<InvSelectOnChange>((v) => {
    if (!v) {
      return;
    }
    setModelTwo(v.value);
  }, []);
  const onChangeModelThree = useCallback<InvSelectOnChange>((v) => {
    if (!v) {
      setModelThree(null);
      setModelMergeInterp('add_difference');
    } else {
      setModelThree(v.value);
      setModelMergeInterp('weighted_sum');
    }
  }, []);

  const valueModelOne = useMemo(
    () => optionsModelOne.find((o) => o.value === modelOne),
    [modelOne, optionsModelOne]
  );
  const valueModelTwo = useMemo(
    () => optionsModelTwo.find((o) => o.value === modelTwo),
    [modelTwo, optionsModelTwo]
  );
  const valueModelThree = useMemo(
    () => optionsModelThree.find((o) => o.value === modelThree),
    [modelThree, optionsModelThree]
  );

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
    <Flex flexDir="column" gap={4}>
      <Flex flexDir="column" gap={1}>
        <InvText>{t('modelManager.modelMergeHeaderHelp1')}</InvText>
        <InvText fontSize="sm" variant="subtext">
          {t('modelManager.modelMergeHeaderHelp2')}
        </InvText>
      </Flex>

      <Flex columnGap={4}>
        <InvControl label={t('modelManager.modelType')} w="full">
          <InvSelect
            options={baseModelTypeSelectOptions}
            value={valueBaseModel}
            onChange={onChangeBaseModel}
          />
        </InvControl>
        <InvControl label={t('modelManager.modelOne')} w="full">
          <InvSelect
            options={optionsModelOne}
            value={valueModelOne}
            onChange={onChangeModelOne}
          />
        </InvControl>
        <InvControl label={t('modelManager.modelTwo')} w="full">
          <InvSelect
            options={optionsModelTwo}
            value={valueModelTwo}
            onChange={onChangeModelTwo}
          />
        </InvControl>
        <InvControl label={t('modelManager.modelThree')} w="full">
          <InvSelect
            options={optionsModelThree}
            value={valueModelThree}
            onChange={onChangeModelThree}
            isClearable
          />
        </InvControl>
      </Flex>

      <InvControl label={t('modelManager.mergedModelName')}>
        <InvInput
          value={mergedModelName}
          onChange={handleChangeMergedModelName}
        />
      </InvControl>

      <Flex
        flexDirection="column"
        padding={4}
        borderRadius="base"
        gap={4}
        bg="base.800"
      >
        <InvControl
          label={t('modelManager.alpha')}
          helperText={t('modelManager.modelMergeAlphaHelp')}
        >
          <InvSlider
            min={0.01}
            max={0.99}
            step={0.01}
            value={modelMergeAlpha}
            onChange={handleChangeModelMergeAlpha}
            onReset={handleResetModelMergeAlpha}
            withNumberInput
            marks
          />
        </InvControl>
      </Flex>

      <Flex padding={4} gap={4} borderRadius="base" bg="base.800">
        <InvText fontSize="sm" variant="subtext">
          {t('modelManager.interpolationType')}
        </InvText>
        <RadioGroup value={modelMergeInterp} onChange={handleChangeMergeInterp}>
          <Flex columnGap={4}>
            {modelThree === null ? (
              <>
                <Radio value="weighted_sum">
                  <InvText fontSize="sm">
                    {t('modelManager.weightedSum')}
                  </InvText>
                </Radio>
                <Radio value="sigmoid">
                  <InvText fontSize="sm">{t('modelManager.sigmoid')}</InvText>
                </Radio>
                <Radio value="inv_sigmoid">
                  <InvText fontSize="sm">
                    {t('modelManager.inverseSigmoid')}
                  </InvText>
                </Radio>
              </>
            ) : (
              <Radio value="add_difference">
                <InvTooltip
                  label={t('modelManager.modelMergeInterpAddDifferenceHelp')}
                >
                  <InvText fontSize="sm">
                    {t('modelManager.addDifference')}
                  </InvText>
                </InvTooltip>
              </Radio>
            )}
          </Flex>
        </RadioGroup>
      </Flex>

      <Flex
        flexDirection="column"
        padding={4}
        borderRadius="base"
        gap={4}
        bg="base.900"
      >
        <Flex columnGap={4}>
          <InvText fontSize="sm" variant="subtext">
            {t('modelManager.mergedModelSaveLocation')}
          </InvText>
          <RadioGroup
            value={modelMergeSaveLocType}
            onChange={handleChangeMergeSaveLocType}
          >
            <Flex columnGap={4}>
              <Radio value="root">
                <InvText fontSize="sm">
                  {t('modelManager.invokeAIFolder')}
                </InvText>
              </Radio>

              <Radio value="custom">
                <InvText fontSize="sm">{t('modelManager.custom')}</InvText>
              </Radio>
            </Flex>
          </RadioGroup>
        </Flex>

        {modelMergeSaveLocType === 'custom' && (
          <InvControl label={t('modelManager.mergedModelCustomSaveLocation')}>
            <InvInput
              value={modelMergeCustomSaveLoc}
              onChange={handleChangeMergeCustomSaveLoc}
            />
          </InvControl>
        )}
      </Flex>

      <InvControl label={t('modelManager.ignoreMismatch')}>
        <InvCheckbox
          isChecked={modelMergeForce}
          onChange={handleChangeModelMergeForce}
        />
      </InvControl>

      <InvButton
        onClick={mergeModelsHandler}
        isLoading={isLoading}
        isDisabled={modelOne === null || modelTwo === null}
      >
        {t('modelManager.merge')}
      </InvButton>
    </Flex>
  );
}
