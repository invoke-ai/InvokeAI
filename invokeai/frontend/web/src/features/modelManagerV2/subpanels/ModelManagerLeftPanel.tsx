import { Button, Flex, Heading } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { setSelectedModelKey } from 'features/modelManagerV2/store/modelManagerV2Slice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';

import ModelList from './ModelManagerPanel/ModelList';
import { ModelListNavigation } from './ModelManagerPanel/ModelListNavigation';

export const ModelManagerLeftPanel = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const handleClickAddModel = useCallback(() => {
    dispatch(setSelectedModelKey(null));
  }, [dispatch]);

  return (
    <Flex flexDir="column" gap={2} w="full" h="full">
      <Flex w="full" gap={4} justifyContent="space-between" alignItems="center">
        <Heading fontSize="xl">{t('common.modelManager')}</Heading>
        <Button size="sm" colorScheme="invokeYellow" leftIcon={<PiPlusBold />} onClick={handleClickAddModel}>
          {t('modelManager.addModels')}
        </Button>
      </Flex>
      <ModelListNavigation />
      <ModelList />
    </Flex>
  );
});

ModelManagerLeftPanel.displayName = 'ModelManagerLeftPanel';
