/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type {
  FieldInputTemplate,
  FieldOutputTemplate,
  FieldType,
  InvocationTemplate,
} from '@features/workflow/contracts';

import { Box, Flex, Stack, Text } from '@chakra-ui/react';
import {
  getWorkflowNodeBodyProps,
  getWorkflowNodeHeaderProps,
  getWorkflowNodeShellProps,
  WORKFLOW_NODE_DENSITY,
  WORKFLOW_NODE_LABEL_TEXT_CSS,
  WORKFLOW_NODE_SPLIT_LABEL_TEXT_PROPS,
  WorkflowNodeHandleDot,
  WorkflowNodeInfoButton,
} from '@features/workflow/preview';
import { getFieldTypeLabel } from '@features/workflow/utility';
import { Tooltip } from '@platform/ui';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { useTranslation } from 'react-i18next';

/**
 * Render template-only node previews without ReactFlow or workbench dependencies; handles and fields are
 * noninteractive.
 */

const sortByUiOrder = <T extends { uiOrder?: number | null }>(templates: T[]): T[] =>
  [...templates].sort((a, b) => (a.uiOrder ?? Number.MAX_SAFE_INTEGER) - (b.uiOrder ?? Number.MAX_SAFE_INTEGER));

const FieldTooltip = ({
  description,
  direction,
  template,
}: {
  description: string;
  direction: 'input' | 'output';
  template: { name: string; title: string; type: FieldType };
}) => {
  const { t } = useTranslation();

  return (
    <Stack gap="0.5" maxW="18rem">
      <Text fontWeight="700">{template.title}</Text>
      <Text color="fg.muted">{t('nodes.fieldName', { name: template.name })}</Text>
      <Text color="fg.muted">{t('nodes.fieldType', { type: getFieldTypeLabel(template.type) })}</Text>
      <Text color="fg.muted">{t(direction === 'input' ? 'nodes.input' : 'nodes.output')}</Text>
      {description ? <Text>{description}</Text> : null}
    </Stack>
  );
};

const OutputRow = ({ template }: { template: FieldOutputTemplate }) => (
  <Box position="relative" px={WORKFLOW_NODE_DENSITY.rowPaddingX} py={WORKFLOW_NODE_DENSITY.rowPaddingY}>
    <Tooltip
      content={<FieldTooltip description={template.description} direction="output" template={template} />}
      positioning={{ placement: 'top-end' }}
    >
      <Flex align="center" h={WORKFLOW_NODE_DENSITY.labelLine} justify="flex-end">
        <MiddleTruncate
          color="fg.muted"
          {...WORKFLOW_NODE_SPLIT_LABEL_TEXT_PROPS}
          fontSize="xs"
          justifyContent="flex-end"
          lineHeight={WORKFLOW_NODE_DENSITY.labelLine}
          maxW="full"
          text={template.title}
        />
      </Flex>
    </Tooltip>
    <WorkflowNodeHandleDot side="right" type={template.type} />
  </Box>
);

const InputRow = ({ template }: { template: FieldInputTemplate }) => (
  <Box position="relative" px={WORKFLOW_NODE_DENSITY.rowPaddingX} py={WORKFLOW_NODE_DENSITY.rowPaddingY}>
    <Tooltip
      content={<FieldTooltip description={template.description} direction="input" template={template} />}
      positioning={{ placement: 'top-start' }}
    >
      <Flex align="center" h={WORKFLOW_NODE_DENSITY.labelLine}>
        <Text
          color="fg"
          css={WORKFLOW_NODE_LABEL_TEXT_CSS}
          fontSize="xs"
          lineHeight={WORKFLOW_NODE_DENSITY.labelLine}
          minW="0"
        >
          {template.title}
          {template.required ? (
            <Text as="span" color="fg.error">
              {' *'}
            </Text>
          ) : null}
        </Text>
      </Flex>
    </Tooltip>
    {template.input !== 'direct' ? <WorkflowNodeHandleDot side="left" type={template.type} /> : null}
  </Box>
);

const NodeInfoTooltip = ({ template }: { template: InvocationTemplate }) => {
  const { t } = useTranslation();

  return (
    <Stack gap="1" maxW="20rem">
      <Text fontWeight="700">{template.title}</Text>
      <Text color="fg.muted">{t('nodes.nodeType', { type: template.type })}</Text>
      <Text color="fg.muted">{t('nodes.nodeCategory', { category: template.category })}</Text>
      <Text color="fg.muted">{t('nodes.nodeVersion', { version: template.version })}</Text>
      {template.description ? <Text fontStyle="italic">{template.description}</Text> : null}
    </Stack>
  );
};

export const NodePreviewCard = ({ template }: { template: InvocationTemplate }) => {
  const { t } = useTranslation();
  const inputTemplates = sortByUiOrder(Object.values(template.inputs).filter((input) => !input.uiHidden));
  const outputTemplates = Object.values(template.outputs);
  const hasFields = inputTemplates.length > 0 || outputTemplates.length > 0;

  return (
    <Box w="full" {...getWorkflowNodeShellProps({ selected: false })}>
      <Flex {...getWorkflowNodeHeaderProps({ endsWithButton: true })}>
        <MiddleTruncate {...WORKFLOW_NODE_SPLIT_LABEL_TEXT_PROPS} fontWeight="600" minW="0" text={template.title} />
        <Box flex="1" />
        <WorkflowNodeInfoButton
          content={<NodeInfoTooltip template={template} />}
          label={t('nodes.nodeDetailsAria', { title: template.title })}
        />
      </Flex>
      <Stack gap="0" {...getWorkflowNodeBodyProps()}>
        {hasFields ? (
          <>
            {outputTemplates.map((output) => (
              <OutputRow key={output.name} template={output} />
            ))}
            {inputTemplates.map((input) => (
              <InputRow key={input.name} template={input} />
            ))}
          </>
        ) : (
          <Text color="fg.muted" fontSize="xs" px="3" py="1">
            {t('nodes.noExposedFields')}
          </Text>
        )}
      </Stack>
    </Box>
  );
};
