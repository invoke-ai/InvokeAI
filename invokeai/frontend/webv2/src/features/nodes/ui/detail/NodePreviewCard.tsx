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
  WorkflowNodeHandleDot,
  WorkflowNodeInfoIcon,
} from '@features/workflow/preview';
import { getFieldTypeLabel } from '@features/workflow/utility';
import { Tooltip } from '@platform/ui';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { useTranslation } from 'react-i18next';

/**
 * A read-only, static rendering of a single node — the same visual language as
 * the editor's `InvocationFlowNode` (titled header, output rows above input
 * rows, colored field handles), shared via the workflow feature's node chrome
 * helpers, but with none of its editor coupling. The real node depends on
 * ReactFlow context (`Handle`, `useStore`) and the workbench dispatch; the
 * Launchpad mounts neither, so this previews the node's shape from just its
 * `InvocationTemplate`. Handles are drawn as plain dots, fields are labels
 * only (no value controls), and nothing is interactive.
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
      <Flex justify="flex-end">
        <MiddleTruncate
          color="fg.muted"
          fontSize="2xs"
          justifyContent="flex-end"
          lineHeight="shorter"
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
      <Text color="fg" fontSize="2xs" lineHeight="shorter" minW="0" truncate>
        {template.title}
        {template.required ? (
          <Text as="span" color="fg.error">
            {' *'}
          </Text>
        ) : null}
      </Text>
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
      <Flex {...getWorkflowNodeHeaderProps()}>
        <MiddleTruncate fontWeight="700" minW="0" text={template.title} />
        <Box flex="1" />
        <WorkflowNodeInfoIcon
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
          <Text color="fg.muted" fontSize="2xs" px="3" py="1">
            {t('nodes.noExposedFields')}
          </Text>
        )}
      </Stack>
    </Box>
  );
};
