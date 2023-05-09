import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAISwitch from 'common/components/IAISwitch';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { shouldShowImageParametersChanged } from 'features/ui/store/uiSlice';
import { ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  [uiSelector, generationSelector],
  (ui, generation) => {
    const { isImageToImageEnabled } = generation;
    const { shouldShowImageParameters } = ui;
    return {
      isImageToImageEnabled,
      shouldShowImageParameters,
    };
  },
  defaultSelectorOptions
);

export default function ImageToImageToggle() {
  const { shouldShowImageParameters } = useAppSelector(selector);

  const { t } = useTranslation();

  const dispatch = useAppDispatch();

  const handleChange = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(shouldShowImageParametersChanged(e.target.checked));

  return (
    <Flex py={1.5} px={4} borderRadius={4}>
      <IAISwitch
        label={t('parameters.initialImage')}
        isChecked={shouldShowImageParameters}
        width="full"
        onChange={handleChange}
        justifyContent="space-between"
        formLabelProps={{
          fontWeight: 400,
        }}
      />
    </Flex>
  );
}
