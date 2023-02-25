import {
  Flex,
  Modal,
  ModalCloseButton,
  ModalContent,
  ModalHeader,
  ModalOverlay,
  Radio,
  RadioGroup,
  Text,
  Tooltip,
  useDisclosure,
} from '@chakra-ui/react';
import { mergeDiffusersModels } from 'app/socketio/actions';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAIInput from 'common/components/IAIInput';
import IAISelect from 'common/components/IAISelect';
import { diffusersModelsSelector } from 'features/system/store/systemSelectors';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import * as InvokeAI from 'app/invokeai';
import IAISlider from 'common/components/IAISlider';
import IAICheckbox from 'common/components/IAICheckbox';

export default function MergeModels() {
  const dispatch = useAppDispatch();

  const { isOpen, onOpen, onClose } = useDisclosure();

  const diffusersModels = useAppSelector(diffusersModelsSelector);

  const { t } = useTranslation();

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
    <>
      <IAIButton onClick={onOpen} className="modal-close-btn" size="sm">
        <Flex columnGap="0.5rem" alignItems="center">
          {t('modelManager.mergeModels')}
        </Flex>
      </IAIButton>

      <Modal
        isOpen={isOpen}
        onClose={onClose}
        size="4xl"
        closeOnOverlayClick={false}
      >
        <ModalOverlay />
        <ModalContent className="modal" fontFamily="Inter" margin="auto">
          <ModalHeader>{t('modelManager.mergeModels')}</ModalHeader>
          <ModalCloseButton />
          <Flex flexDirection="column" padding="1rem" rowGap={4}>
            <Flex
              flexDirection="column"
              marginBottom="1rem"
              padding="1rem"
              borderRadius="0.3rem"
              backgroundColor="var(--background-color)"
              rowGap={1}
            >
              <Text>{t('modelManager.modelMergeHeaderHelp1')}</Text>
              <Text fontSize="0.9rem" color="var(--text-color-secondary)">
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
              flexDir="column"
              backgroundColor="var(--background-color)"
              padding="1rem 1rem"
              borderRadius="0.2rem"
              rowGap={2}
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
                sliderMarkRightOffset={-7}
              />
              <Text fontSize="0.9rem" color="var(--text-color-secondary)">
                {t('modelManager.modelMergeAlphaHelp')}
              </Text>
            </Flex>

            <Flex
              columnGap={4}
              backgroundColor="var(--background-color)"
              padding="1rem 1rem"
              borderRadius="0.2rem"
            >
              <Text
                fontWeight="bold"
                fontSize="0.9rem"
                color="var(--text-color-secondary)"
              >
                {t('modelManager.interpolationType')}
              </Text>
              <RadioGroup
                value={modelMergeInterp}
                onChange={(
                  v:
                    | 'weighted_sum'
                    | 'sigmoid'
                    | 'inv_sigmoid'
                    | 'add_difference'
                ) => setModelMergeInterp(v)}
              >
                <Flex columnGap={4}>
                  {modelThree === 'none' ? (
                    <>
                      <Radio value="weighted_sum">
                        {t('modelManager.weightedSum')}
                      </Radio>
                      <Radio value="sigmoid">{t('modelManager.sigmoid')}</Radio>
                      <Radio value="inv_sigmoid">
                        {t('modelManager.inverseSigmoid')}
                      </Radio>
                    </>
                  ) : (
                    <Radio value="add_difference">
                      <Tooltip
                        label={t(
                          'modelManager.modelMergeInterpAddDifferenceHelp'
                        )}
                      >
                        {t('modelManager.addDifference')}
                      </Tooltip>
                    </Radio>
                  )}
                </Flex>
              </RadioGroup>
            </Flex>

            <Flex
              gap={4}
              flexDirection="column"
              backgroundColor="var(--background-color)"
              padding="1rem 1rem"
              borderRadius="0.2rem"
            >
              <Flex columnGap={4}>
                <Text
                  fontWeight="bold"
                  fontSize="0.9rem"
                  color="var(--text-color-secondary)"
                >
                  {t('modelManager.mergedModelSaveLocation')}
                </Text>
                <RadioGroup
                  value={modelMergeSaveLocType}
                  onChange={(v: 'root' | 'custom') =>
                    setModelMergeSaveLocType(v)
                  }
                >
                  <Flex columnGap={4}>
                    <Radio value="root">
                      {t('modelManager.invokeAIFolder')}
                    </Radio>
                    <Radio value="custom">{t('modelManager.custom')}</Radio>
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

            <IAICheckbox
              label={t('modelManager.ignoreMismatch')}
              isChecked={modelMergeForce}
              onChange={(e) => setModelMergeForce(e.target.checked)}
              fontWeight="bold"
            />

            <IAIButton
              onClick={mergeModelsHandler}
              isLoading={isProcessing}
              isDisabled={
                modelMergeSaveLocType === 'custom' &&
                modelMergeCustomSaveLoc === ''
              }
              className="modal modal-close-btn"
            >
              {t('modelManager.merge')}
            </IAIButton>
          </Flex>
        </ModalContent>
      </Modal>
    </>
  );
}
