import type {
  WorkflowPreviewFormItem,
  WorkflowPreviewInput,
  WorkflowPreviewModel,
} from '@features/workflow/core/workflowPreview';
import type { ModelRequirementDeps } from '@features/workflow/ui/modelRequirementStatus';

import { Box, Center, HStack, Icon, Spinner, Stack, type SystemStyleObject, Text } from '@chakra-ui/react';
import { resolveWorkflowModelRequirements } from '@features/workflow/core/modelRequirements';
import {
  REQUIREMENT_STATUS_PRESENTATION,
  useLoadedModelRequirementDeps,
} from '@features/workflow/ui/modelRequirementStatus';
import { Button } from '@platform/ui/Button';
import { Scrollable } from '@platform/ui/Scrollable';
import { useCallback, useId, useLayoutEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { formatWorkflowPreviewValue } from './nodeSummaries';

/**
 * The preview's Inputs view of a workflow document: its form as its author arranged it, read-only, with the saved
 * values. It answers "what do I fill in" without reading the graph; each input names the node it sets, which opens that
 * node in the graph.
 */

const VALUE_CSS: SystemStyleObject = {
  bg: 'bg.subtle',
  borderColor: 'border.subtle',
  borderWidth: '1px',
  fontSize: 'md',
  lineHeight: '1.45',
  minH: '7.5',
  overflowWrap: 'anywhere',
  px: '2.5',
  py: '1.5',
  rounded: 'md',
  whiteSpace: 'pre-wrap',
};
const EMPTY_VALUE_CSS: SystemStyleObject = { ...VALUE_CSS, borderStyle: 'dashed', color: 'fg.muted' };
const NEEDS_VALUE_CSS: SystemStyleObject = { ...VALUE_CSS, borderColor: 'border.warning' };
/** About four lines of value; a longer text scrolls inside its box rather than being cut. */
const TEXT_VALUE_CSS: SystemStyleObject = { ...VALUE_CSS, maxH: '6.5rem', overflowY: 'auto' };

const InputValue = ({ deps, input }: { deps: ModelRequirementDeps; input: WorkflowPreviewInput }) => {
  const { t } = useTranslation();
  // A model the field names, or the kind it asks for while empty, that this server does not have.
  const missing = resolveWorkflowModelRequirements(input.modelRequirements, deps).find(
    (resolved) => resolved.status !== 'installed'
  );

  if (missing) {
    const presentation = REQUIREMENT_STATUS_PRESENTATION[missing.status];

    return (
      <HStack color={presentation.color} css={NEEDS_VALUE_CSS} data-input-value="missing" gap="2">
        {presentation.icon ? (
          <Icon aria-hidden as={presentation.icon} boxSize="3" flexShrink={0} />
        ) : (
          <Spinner aria-hidden borderWidth="1.5px" boxSize="3" flexShrink={0} />
        )}
        <Text minW="0">
          {input.value ? formatWorkflowPreviewValue(input.value, t) : missing.requirement.label} ·{' '}
          {t(presentation.labelKey)}
        </Text>
      </HStack>
    );
  }

  if (!input.value) {
    return (
      <Text css={EMPTY_VALUE_CSS} data-input-value="empty">
        {input.isModelField ? t('workflowLibrary.preview.pickAfterOpening') : t('workflowLibrary.preview.emptyValue')}
      </Text>
    );
  }

  const text = formatWorkflowPreviewValue(input.value, t);

  return input.value.kind === 'text' ? (
    <TextValue text={text} />
  ) : (
    <Text css={VALUE_CSS} data-input-value="">
      {text}
    </Text>
  );
};

/** A text value: a long one scrolls inside its box, which then takes focus so the keyboard can scroll it too. */
const TextValue = ({ text }: { text: string }) => {
  const ref = useRef<HTMLParagraphElement>(null);
  const [overflows, setOverflows] = useState(false);

  // Whether the box overflows decides whether it is a tab stop; measured before paint, once per text.
  useLayoutEffect(() => {
    const element = ref.current;

    if (element) {
      setOverflows(element.scrollHeight > element.clientHeight + 1);
    }
  }, [text]);

  return (
    <Text ref={ref} css={TEXT_VALUE_CSS} data-input-value="" tabIndex={overflows ? 0 : undefined}>
      {text}
    </Text>
  );
};

const InputField = ({
  deps,
  input,
  onShowNode,
}: {
  deps: ModelRequirementDeps;
  input: WorkflowPreviewInput;
  onShowNode: (nodeId: string) => void;
}) => {
  const { t } = useTranslation();
  const labelId = useId();
  const showNode = useCallback(() => onShowNode(input.nodeId), [input.nodeId, onShowNode]);

  return (
    <Stack aria-labelledby={labelId} data-preview-input={input.id} flex="1" gap="1" minW="0" role="group">
      <HStack align="baseline" gap="2" justify="space-between" minW="0">
        <Text fontSize="md" fontWeight="500" id={labelId} minW="0" truncate>
          {input.label}
        </Text>
        <Button
          aria-label={t('workflowLibrary.preview.showNode', { node: input.nodeTitle })}
          color="accent.fg"
          flexShrink={1}
          fontSize="sm"
          fontWeight="400"
          h="auto"
          minW="0"
          px="0"
          size="xs"
          variant="plain"
          onClick={showNode}
        >
          <Text as="span" truncate>
            {input.nodeTitle}
          </Text>
        </Button>
      </HStack>
      <InputValue deps={deps} input={input} />
    </Stack>
  );
};

const FormItems = ({
  deps,
  items,
  onShowNode,
}: {
  deps: ModelRequirementDeps;
  items: readonly WorkflowPreviewFormItem[];
  onShowNode: (nodeId: string) => void;
}) =>
  items.map((item) => {
    switch (item.kind) {
      case 'heading':
        return (
          <Text
            key={item.id}
            as="h3"
            borderBottomWidth="1px"
            borderColor="border.subtle"
            fontSize="lg"
            fontWeight="600"
            pb="2"
            pt="1.5"
          >
            {item.text}
          </Text>
        );
      case 'text':
        return (
          <Text key={item.id} color="fg.muted" fontSize="md" whiteSpace="pre-wrap">
            {item.text}
          </Text>
        );
      case 'divider':
        return <Box key={item.id} aria-hidden bg="border.subtle" flexShrink={0} h="1px" />;
      case 'container':
        return item.layout === 'row' ? (
          <HStack key={item.id} align="flex-start" flexWrap="wrap" gap="3" minW="0">
            <FormItems deps={deps} items={item.children} onShowNode={onShowNode} />
          </HStack>
        ) : (
          <Stack key={item.id} gap="3.5" minW="0">
            <FormItems deps={deps} items={item.children} onShowNode={onShowNode} />
          </Stack>
        );
      case 'input':
        return <InputField key={item.id} deps={deps} input={item.input} onShowNode={onShowNode} />;
    }
  });

export const WorkflowPreviewInputs = ({
  model,
  onShowNode,
}: {
  model: WorkflowPreviewModel;
  /** Shows the node an input sets, selected in the graph. */
  onShowNode: (nodeId: string) => void;
}) => {
  const { t } = useTranslation();
  const deps = useLoadedModelRequirementDeps();

  if (model.inputs.length === 0) {
    return (
      <Center h="full" p="6">
        <Text color="fg.muted" fontSize="lg" maxW="28rem" textAlign="center">
          {t('workflowLibrary.preview.noInputs')}
        </Text>
      </Center>
    );
  }

  return (
    <Scrollable h="full" label={t('workflowLibrary.preview.inputs')}>
      <Stack gap="3.5" maxW="40rem" mx="auto" px="6" py="5">
        <Text color="fg.muted" fontSize="md">
          {t('workflowLibrary.preview.inputsIntro')}
        </Text>
        <FormItems deps={deps} items={model.form} onShowNode={onShowNode} />
      </Stack>
    </Scrollable>
  );
};
