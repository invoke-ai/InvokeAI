import { Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { $templates } from 'features/nodes/store/nodesSlice';
import { selectInvocationNode, selectNodesSlice } from 'features/nodes/store/selectors';
import type { PropsWithChildren } from 'react';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

type Props = PropsWithChildren<{
  nodeId: string;
  fieldName: string;
}>;

export const InvocationInputFieldCheck = memo(({ nodeId, fieldName, children }: Props) => {
  const { t } = useTranslation();
  const templates = useStore($templates);
  const selector = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodesSlice) => {
        const node = selectInvocationNode(nodesSlice, nodeId);
        const instance = node.data.inputs[fieldName];
        const template = templates[node.data.type];
        const fieldTemplate = template?.inputs[fieldName];
        return {
          name: instance?.label || fieldTemplate?.title || fieldName,
          hasInstance: Boolean(instance),
          hasTemplate: Boolean(fieldTemplate),
        };
      }),
    [fieldName, nodeId, templates]
  );
  const { hasInstance, hasTemplate, name } = useAppSelector(selector);

  if (!hasTemplate || !hasInstance) {
    return (
      <Flex position="relative" minH={8} py={0.5} alignItems="center" w="full" h="full">
        <FormControl
          isInvalid={true}
          alignItems="stretch"
          justifyContent="center"
          flexDir="column"
          gap={2}
          h="full"
          w="full"
        >
          <FormLabel display="flex" mb={0} px={1} py={2} gap={2}>
            {t('nodes.unknownInput', { name })}
          </FormLabel>
        </FormControl>
      </Flex>
    );
  }

  return children;
});

InvocationInputFieldCheck.displayName = 'InvocationInputFieldCheck';
