import { ButtonGroup, Flex, Spacer, Text } from '@chakra-ui/react';
import IAIIconButton from 'common/components/IAIIconButton';

import { useTranslation } from 'react-i18next';
import { FaUndo, FaUpload } from 'react-icons/fa';
import { useAppDispatch } from 'app/store/storeHooks';
import { useCallback } from 'react';
import { clearInitialImage } from 'features/parameters/store/generationSlice';

const ImageToImageSettingsHeader = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleResetInitialImage = useCallback(() => {
    dispatch(clearInitialImage());
  }, [dispatch]);

  return (
    <Flex w="full" alignItems="center">
      <Text size="sm" fontWeight={500} color="base.300">
        Image to Image
      </Text>
      <Spacer />
      <ButtonGroup>
        <IAIIconButton
          size="sm"
          icon={<FaUndo />}
          aria-label={t('accessibility.reset')}
          onClick={handleResetInitialImage}
        />
        <IAIIconButton
          size="sm"
          icon={<FaUpload />}
          aria-label={t('common.upload')}
        />
      </ButtonGroup>
    </Flex>
  );
};

export default ImageToImageSettingsHeader;
