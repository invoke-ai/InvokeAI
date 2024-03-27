import { Button, Flex, Image, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { workflowModeChanged } from 'features/nodes/store/workflowSlice';
/** @knipignore */
import InvokeLogoSVG from 'public/assets/images/invoke-symbol-wht-lrg.svg';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

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
