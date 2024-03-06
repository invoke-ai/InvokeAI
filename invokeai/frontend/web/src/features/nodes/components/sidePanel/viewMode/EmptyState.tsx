import { Button, Flex, Text, Image } from '@invoke-ai/ui-library';
import { useTranslation } from 'react-i18next';
import { useAppDispatch } from '../../../../../app/store/storeHooks';
import { useCallback } from 'react';
import { workflowModeChanged } from '../../../store/workflowSlice';
import InvokeLogoSVG from 'public/assets/images/invoke-symbol-wht-lrg.svg';

export const EmptyState = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const onClick = useCallback(() => {
    dispatch(workflowModeChanged('edit'));
  }, [dispatch]);

  return (
    <Flex
      sx={{
        w: 'full',
        h: 'full',
        userSelect: 'none',
      }}
    >
      <Flex
        sx={{
          alignItems: 'center',
          justifyContent: 'center',
          borderRadius: 'base',
          flexDir: 'column',
          gap: 5,
          maxW: '230px',
          margin: '0 auto',
        }}
      >
        <Image
          src={InvokeLogoSVG}
          alt="invoke-ai-logo"
          opacity={0.2}
          mixBlendMode="overlay"
          w={16}
          h={16}
          minW={16}
          minH={16}
          userSelect="none"
        />
        <Text textAlign="center" fontSize="md">
          {t('nodes.noFieldsViewMode')}
        </Text>
        <Button colorScheme="invokeBlue" onClick={onClick}>
          {t('nodes.edit')}
        </Button>
      </Flex>
    </Flex>
  );
};
