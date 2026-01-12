import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Button, Flex, Heading } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectCurrentUser } from 'features/auth/store/authSlice';
import { selectSelectedModelKey, setSelectedModelKey } from 'features/modelManagerV2/store/modelManagerV2Slice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';

import ModelList from './ModelManagerPanel/ModelList';
import { ModelListNavigation } from './ModelManagerPanel/ModelListNavigation';

const modelManagerSx: SystemStyleObject = {
  flexDir: 'column',
  p: 4,
  gap: 4,
  borderRadius: 'base',
  w: '50%',
  minWidth: '360px',
  h: 'full',
};

export const ModelManager = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const user = useAppSelector(selectCurrentUser);
  const handleClickAddModel = useCallback(() => {
    dispatch(setSelectedModelKey(null));
  }, [dispatch]);
  const selectedModelKey = useAppSelector(selectSelectedModelKey);

  // Hide "Add Models" button for non-admin users
  const canAddModels = user?.is_admin ?? false;

  return (
    <Flex sx={modelManagerSx}>
      <Flex w="full" gap={4} justifyContent="space-between" alignItems="center">
        <Heading fontSize="xl" py={1}>
          {t('common.modelManager')}
        </Heading>
        {!!selectedModelKey && canAddModels && (
          <Button size="sm" colorScheme="invokeYellow" leftIcon={<PiPlusBold />} onClick={handleClickAddModel}>
            {t('modelManager.addModels')}
          </Button>
        )}
      </Flex>
      <Flex flexDir="column" gap={4} w="full" h="full">
        <ModelListNavigation />
        <ModelList />
      </Flex>
    </Flex>
  );
});

ModelManager.displayName = 'ModelManager';
