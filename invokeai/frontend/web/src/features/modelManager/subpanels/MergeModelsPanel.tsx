import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import {
  Button,
  Checkbox,
  Combobox,
  CompositeNumberInput,
  CompositeSlider,
  Flex,
  FormControl,
  FormHelperText,
  FormLabel,
  Input,
  Radio,
  RadioGroup,
  Text,
  Tooltip,
} from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { pickBy } from 'lodash-es';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ALL_BASE_MODELS } from 'services/api/constants';
import { useGetMainModelsQuery, useMergeMainModelsMutation } from 'services/api/endpoints/models';
import type { BaseModelType, MergeModelConfig } from 'services/api/types';

const baseModelTypeSelectOptions: ComboboxOption[] = [
  { label: 'Stable Diffusion 1', value: 'sd-1' },
  { label: 'Stable Diffusion 2', value: 'sd-2' },
];

type MergeInterpolationMethods = 'weighted_sum' | 'sigmoid' | 'inv_sigmoid' | 'add_difference';

const MergeModelsPanel = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const { data } = useGetMainModelsQuery(ALL_BASE_MODELS);

  const [mergeModels, { isLoading }] = useMergeMainModelsMutation();

  const [baseModel, setBaseModel] = useState<BaseModelType>('sd-1');
  const valueBaseModel = useMemo(() => baseModelTypeSelectOptions.find((o) => o.value === baseModel), [baseModel]);
  const sd1DiffusersModels = pickBy(
    data?.entities,
    (value, _) => value?.model_format === 'diffusers' && value?.base_model === 'sd-1'
  );

  const sd2DiffusersModels = pickBy(
    data?.entities,
    (value, _) => value?.model_format === 'diffusers' && value?.base_model === 'sd-2'
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

  const [modelMergeInterp, setModelMergeInterp] = useState<MergeInterpolationMethods>('weighted_sum');

  const [modelMergeSaveLocType, setModelMergeSaveLocType] = useState<'root' | 'custom'>('root');

  const [modelMergeCustomSaveLoc, setModelMergeCustomSaveLoc] = useState<string>('');

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

  const onChangeBaseModel = useCallback<ComboboxOnChange>((v) => {
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

  const onChangeModelOne = useCallback<ComboboxOnChange>((v) => {
    if (!v) {
      return;
    }
    setModelOne(v.value);
  }, []);
  const onChangeModelTwo = useCallback<ComboboxOnChange>((v) => {
    if (!v) {
      return;
    }
    setModelTwo(v.value);
  }, []);
  const onChangeModelThree = useCallback<ComboboxOnChange>((v) => {
    if (!v) {
      setModelThree(null);
      setModelMergeInterp('add_difference');
    } else {
      setModelThree(v.value);
      setModelMergeInterp('weighted_sum');
    }
  }, []);

  const valueModelOne = useMemo(() => optionsModelOne.find((o) => o.value === modelOne), [modelOne, optionsModelOne]);
  const valueModelTwo = useMemo(() => optionsModelTwo.find((o) => o.value === modelTwo), [modelTwo, optionsModelTwo]);
  const valueModelThree = useMemo(
    () => optionsModelThree.find((o) => o.value === modelThree),
    [modelThree, optionsModelThree]
  );

  const handleChangeMergedModelName = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => setMergedModelName(e.target.value),
    []
  );
  const handleChangeModelMergeAlpha = useCallback((v: number) => setModelMergeAlpha(v), []);
  const handleResetModelMergeAlpha = useCallback(() => setModelMergeAlpha(0.5), []);
  const handleChangeMergeInterp = useCallback((v: MergeInterpolationMethods) => setModelMergeInterp(v), []);
  const handleChangeMergeSaveLocType = useCallback((v: 'root' | 'custom') => setModelMergeSaveLocType(v), []);
  const handleChangeMergeCustomSaveLoc = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => setModelMergeCustomSaveLoc(e.target.value),
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
      merged_model_name: mergedModelName !== '' ? mergedModelName : models_names.join('-'),
      alpha: modelMergeAlpha,
      interp: modelMergeInterp,
      force: modelMergeForce,
      merge_dest_directory: modelMergeSaveLocType === 'root' ? undefined : modelMergeCustomSaveLoc,
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
        <Text>{t('modelManager.modelMergeHeaderHelp1')}</Text>
        <Text fontSize="sm" variant="subtext">
          {t('modelManager.modelMergeHeaderHelp2')}
        </Text>
      </Flex>

      <Flex columnGap={4}>
        <FormControl w="full">
          <FormLabel>{t('modelManager.modelType')}</FormLabel>
          <Combobox options={baseModelTypeSelectOptions} value={valueBaseModel} onChange={onChangeBaseModel} />
        </FormControl>
        <FormControl w="full">
          <FormLabel>{t('modelManager.modelOne')}</FormLabel>
          <Combobox options={optionsModelOne} value={valueModelOne} onChange={onChangeModelOne} />
        </FormControl>
        <FormControl w="full">
          <FormLabel>{t('modelManager.modelTwo')}</FormLabel>
          <Combobox options={optionsModelTwo} value={valueModelTwo} onChange={onChangeModelTwo} />
        </FormControl>
        <FormControl w="full">
          <FormLabel>{t('modelManager.modelThree')}</FormLabel>
          <Combobox options={optionsModelThree} value={valueModelThree} onChange={onChangeModelThree} isClearable />
        </FormControl>
      </Flex>

      <FormControl>
        <FormLabel>{t('modelManager.mergedModelName')}</FormLabel>
        <Input value={mergedModelName} onChange={handleChangeMergedModelName} />
      </FormControl>

      <Flex flexDirection="column" padding={4} borderRadius="base" gap={4} bg="base.800">
        <FormControl>
          <FormLabel>{t('modelManager.alpha')}</FormLabel>
          <CompositeSlider
            min={0.01}
            max={0.99}
            step={0.01}
            value={modelMergeAlpha}
            onChange={handleChangeModelMergeAlpha}
            onReset={handleResetModelMergeAlpha}
            marks
          />
          <CompositeNumberInput
            min={0.01}
            max={0.99}
            step={0.01}
            value={modelMergeAlpha}
            onChange={handleChangeModelMergeAlpha}
            onReset={handleResetModelMergeAlpha}
          />
          <FormHelperText>{t('modelManager.modelMergeAlphaHelp')}</FormHelperText>
        </FormControl>
      </Flex>

      <Flex padding={4} gap={4} borderRadius="base" bg="base.800">
        <Text fontSize="sm" variant="subtext">
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
                <Tooltip label={t('modelManager.modelMergeInterpAddDifferenceHelp')}>
                  <Text fontSize="sm">{t('modelManager.addDifference')}</Text>
                </Tooltip>
              </Radio>
            )}
          </Flex>
        </RadioGroup>
      </Flex>

      <Flex flexDirection="column" padding={4} borderRadius="base" gap={4} bg="base.900">
        <Flex columnGap={4}>
          <Text fontSize="sm" variant="subtext">
            {t('modelManager.mergedModelSaveLocation')}
          </Text>
          <RadioGroup value={modelMergeSaveLocType} onChange={handleChangeMergeSaveLocType}>
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
          <FormControl>
            <FormLabel>{t('modelManager.mergedModelCustomSaveLocation')}</FormLabel>
            <Input value={modelMergeCustomSaveLoc} onChange={handleChangeMergeCustomSaveLoc} />
          </FormControl>
        )}
      </Flex>

      <FormControl>
        <FormLabel>{t('modelManager.ignoreMismatch')}</FormLabel>
        <Checkbox isChecked={modelMergeForce} onChange={handleChangeModelMergeForce} />
      </FormControl>

      <Button onClick={mergeModelsHandler} isLoading={isLoading} isDisabled={modelOne === null || modelTwo === null}>
        {t('modelManager.merge')}
      </Button>
    </Flex>
  );
};

export default memo(MergeModelsPanel);
