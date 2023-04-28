import { Flex } from '@chakra-ui/react';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { isImageToImageEnabledChanged } from 'features/parameters/store/generationSlice';
import { ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';

export default function ImageToImageToggle() {
  const isImageToImageEnabled = useAppSelector(
    (state: RootState) => state.generation.isImageToImageEnabled
  );

  const { t } = useTranslation();

  const dispatch = useAppDispatch();

  const handleChange = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(isImageToImageEnabledChanged(e.target.checked));

  return (
    <Flex background="base.800" py={1.5} px={4} borderRadius={4}>
      <IAISwitch
        label={t('common.img2img')}
        isChecked={isImageToImageEnabled}
        width="full"
        onChange={handleChange}
        justifyContent="space-between"
        formLabelProps={{
          fontWeight: 'bold',
          color: 'base.200',
        }}
      />
    </Flex>
  );
}
