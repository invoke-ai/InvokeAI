import type {
  FieldInputTemplate,
  FieldOutputTemplate,
  FieldType,
  InvocationTemplate,
} from '@workbench/workflows/types';

import { Box, Flex, Icon, Stack, Text } from '@chakra-ui/react';
import { Tooltip } from '@workbench/components/ui';
import { getWorkflowNodeChromeProps } from '@workbench/widgets/workflow/editor/nodeChrome';
import { getFieldTypeColor, getFieldTypeLabel, isModelFieldType } from '@workbench/workflows/fields';
import { InfoIcon } from 'lucide-react';

/**
 * A read-only, static rendering of a single node — the same visual language as
 * the editor's `InvocationFlowNode` (titled header, output rows above input
 * rows, colored field handles) but with none of its editor coupling. The real
 * node depends on ReactFlow context (`Handle`, `useStore`) and the workbench
 * dispatch; the Launchpad mounts neither, so this previews the node's shape
 * from just its `InvocationTemplate`. Handles are drawn as plain dots, fields
 * are labels only (no value controls), and nothing is interactive.
 */

const sortByUiOrder = <T extends { uiOrder?: number | null }>(templates: T[]): T[] =>
  [...templates].sort((a, b) => (a.uiOrder ?? Number.MAX_SAFE_INTEGER) - (b.uiOrder ?? Number.MAX_SAFE_INTEGER));

/** A static stand-in for a ReactFlow connection handle, tinted and shaped by field type. */
const HandleDot = ({ side, type }: { side: 'left' | 'right'; type: FieldType }) => {
  const color = getFieldTypeColor(type);
  const isFilled = type.cardinality === 'SINGLE';
  const isAngular = isModelFieldType(type) || type.batch;

  return (
    <Box
      bg={isFilled ? color : 'bg.muted'}
      borderColor={color}
      borderRadius={isAngular ? 'xs' : 'full'}
      borderWidth={isFilled ? '0' : '2px'}
      boxShadow="0 0 0 1.5px {colors.bg.muted}"
      boxSize="3"
      position="absolute"
      top="50%"
      transform={`translate(${side === 'left' ? '-50%' : '50%'}, -50%)${type.batch ? ' rotate(45deg)' : ''}`}
      {...(side === 'left' ? { left: '0' } : { right: '0' })}
    />
  );
};

const FieldTooltip = ({
  description,
  direction,
  template,
}: {
  description: string;
  direction: 'Input' | 'Output';
  template: { name: string; title: string; type: FieldType };
}) => (
  <Stack gap="0.5" maxW="18rem">
    <Text fontWeight="700">{template.title}</Text>
    <Text color="fg.subtle">Field: {template.name}</Text>
    <Text color="fg.subtle">Type: {getFieldTypeLabel(template.type)}</Text>
    <Text color="fg.subtle">{direction}</Text>
    {description ? <Text>{description}</Text> : null}
  </Stack>
);

const OutputRow = ({ template }: { template: FieldOutputTemplate }) => (
  <Box position="relative" px="3" py="1">
    <Tooltip
      content={<FieldTooltip description={template.description} direction="Output" template={template} />}
      positioning={{ placement: 'top-end' }}
    >
      <Flex justify="flex-end">
        <Text color="fg.muted" fontSize="2xs" lineHeight="1" maxW="full" textAlign="end" truncate>
          {template.title}
        </Text>
      </Flex>
    </Tooltip>
    <HandleDot side="right" type={template.type} />
  </Box>
);

const InputRow = ({ template }: { template: FieldInputTemplate }) => (
  <Box position="relative" px="3" py="1">
    <Tooltip
      content={<FieldTooltip description={template.description} direction="Input" template={template} />}
      positioning={{ placement: 'top-start' }}
    >
      <Text color="fg" fontSize="2xs" lineHeight="1" minW="0" truncate>
        {template.title}
        {template.required ? (
          <Text as="span" color="fg.error">
            {' *'}
          </Text>
        ) : null}
      </Text>
    </Tooltip>
    {template.input !== 'direct' ? <HandleDot side="left" type={template.type} /> : null}
  </Box>
);

const NodeInfoTooltip = ({ template }: { template: InvocationTemplate }) => (
  <Stack gap="1" maxW="20rem">
    <Text fontWeight="700">{template.title}</Text>
    <Text color="fg.subtle">Type: {template.type}</Text>
    <Text color="fg.subtle">Category: {template.category}</Text>
    <Text color="fg.subtle">Version: {template.version}</Text>
    {template.description ? <Text fontStyle="italic">{template.description}</Text> : null}
  </Stack>
);

export const NodePreviewCard = ({ template }: { template: InvocationTemplate }) => {
  const inputTemplates = sortByUiOrder(Object.values(template.inputs).filter((input) => !input.uiHidden));
  const outputTemplates = Object.values(template.outputs);
  const hasFields = inputTemplates.length > 0 || outputTemplates.length > 0;

  return (
    <Box bg="bg" fontSize="xs" rounded="lg" w="full" {...getWorkflowNodeChromeProps({ selected: false })}>
      <Flex
        align="center"
        bg="bg.subtle"
        borderBottomWidth="1px"
        borderColor="border.subtle"
        borderTopRadius="lg"
        gap="2"
        pe="2"
        ps="2.5"
        py="1.5"
      >
        <Text fontWeight="700" minW="0" title={template.title} truncate>
          {template.title}
        </Text>
        <Box flex="1" />
        <Tooltip content={<NodeInfoTooltip template={template} />} positioning={{ placement: 'top-end' }} showArrow>
          <Icon aria-label={`Details for ${template.title}`} as={InfoIcon} boxSize="3.5" color="fg.subtle" />
        </Tooltip>
      </Flex>
      <Stack bg="bg.muted" borderBottomRadius="lg" gap="0" py="1">
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
          <Text color="fg.subtle" fontSize="2xs" px="3" py="1">
            No exposed fields.
          </Text>
        )}
      </Stack>
    </Box>
  );
};
