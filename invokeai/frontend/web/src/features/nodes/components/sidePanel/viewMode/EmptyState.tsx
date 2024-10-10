import { Button, Flex, Image, Link, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useWorkflowListMenu } from 'features/nodes/store/workflowListMenu';
import { selectCleanEditor, workflowModeChanged } from 'features/nodes/store/workflowSlice';
import InvokeLogoSVG from 'public/assets/images/invoke-symbol-wht-lrg.svg';
import { useCallback } from 'react';
import { Trans, useTranslation } from 'react-i18next';

export const EmptyState = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const workflowListMenu = useWorkflowListMenu();
  const isCleanEditor = useAppSelector(selectCleanEditor);

  const onClick = useCallback(() => {
    dispatch(workflowModeChanged('edit'));
  }, [dispatch]);

  const onClickNewWorkflow = useCallback(() => {
    dispatch(workflowModeChanged('edit'));
  }, [dispatch]);

  return (
    <Flex w="full" userSelect="none" justifyContent="center">
      <Flex
        alignItems="center"
        justifyContent="center"
        borderRadius="base"
        flexDir="column"
        gap={5}
        maxW="230px"
        pt={24}
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
        {isCleanEditor ? (
          <>
            <Flex gap={2}>
              <Button size="sm" onClick={onClickNewWorkflow}>
                {t('nodes.newWorkflow')}
              </Button>
              <Button size="sm" colorScheme="invokeBlue" onClick={workflowListMenu.open}>
                {t('nodes.loadWorkflow')}
              </Button>
            </Flex>
            <Text textAlign="center" fontSize="md">
              <Trans i18nKey="nodes.workflowHelpText" size="sm" components={workflowHelpTextComponents} />
            </Text>
          </>
        ) : (
          <>
            <Text textAlign="center" fontSize="md">
              {t('nodes.noFieldsViewMode')}
            </Text>
            <Button size="sm" colorScheme="invokeBlue" onClick={onClick}>
              {t('nodes.edit')}
            </Button>
          </>
        )}
      </Flex>
    </Flex>
  );
};

const workflowHelpTextComponents = {
  LinkComponent: (
    <Link
      fontSize="md"
      fontWeight="semibold"
      href="https://support.invoke.ai/support/solutions/articles/151000159663-example-workflows"
      target="_blank"
    />
  ),
};
