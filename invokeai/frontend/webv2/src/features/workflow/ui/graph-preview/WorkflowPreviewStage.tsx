import type { WorkflowPreviewModel } from '@features/workflow/core/workflowPreview';
import type { WorkflowDocumentGraphHandle } from '@features/workflow/ui/graph-preview/WorkflowDocumentGraph';
import type { TFunction } from 'i18next';

import { Center, HStack, Icon, Input, InputGroup, Spinner, type SystemStyleObject, Text } from '@chakra-ui/react';
import { SearchIcon } from 'lucide-react';
import {
  type ChangeEvent,
  lazy,
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactNode,
  type RefObject,
  useCallback,
  useState,
} from 'react';
import { useTranslation } from 'react-i18next';

/**
 * The parts every workflow preview stage shares (the library's preview mode and the graph preview dialog): the bar
 * over the stage, its find field, its Nodes / Inputs / Graph views and the messages the stage shows in their place.
 * The bar keeps its tabs at the start and its menu at the end in every view; only the find field, shown with the
 * nodes, comes and goes in the space between, so nothing a reader aims at moves.
 */

/**
 * What the stage shows: the nodes (the diagram), the form's inputs, or the graph: the compiled JSON the server is sent,
 * which is what "graph" means here.
 */
export type WorkflowPreviewView = 'nodes' | 'inputs' | 'graph';

/**
 * The size of the dialogs a workflow preview shows in (the library, and the graph preview), so both previews read the
 * same: most of the window, wide and landscape.
 */
export const WORKFLOW_PREVIEW_DIALOG_SIZE = {
  h: '80vh',
  maxH: '80vh',
  maxW: 'min(80rem, calc(100vw - 4rem))',
} as const;

/** The Graph view and the platform's JSON preview load when that view first shows, outside every startup chunk. */
export const WorkflowPreviewJson = lazy(() => import('./WorkflowPreviewJson'));

const FIND_START_ELEMENT = <Icon aria-hidden as={SearchIcon} boxSize="3.5" color="fg.subtle" />;

export const WorkflowPreviewBar = ({ children }: { children: ReactNode }) => (
  <HStack
    borderBottomWidth="1px"
    borderColor="border.subtle"
    data-preview-stage-bar=""
    flexShrink={0}
    flexWrap="wrap"
    gap="2.5"
    minH="11"
    px="3"
    py="1.5"
  >
    {children}
  </HStack>
);

/**
 * The view tabs: Nodes; Inputs, counting the form's inputs once the document is read, where there is a form; Graph,
 * where there is a compiled graph (or a document to compile) to show.
 */
export const getWorkflowPreviewViewTabs = (
  model: WorkflowPreviewModel | null,
  { hasGraph, hasInputs }: { hasGraph: boolean; hasInputs: boolean },
  t: TFunction
) => [
  { id: 'nodes' as const, label: t('workflowLibrary.preview.nodes') },
  ...(hasInputs
    ? [
        {
          id: 'inputs' as const,
          label: model ? (
            <>
              {t('workflowLibrary.preview.inputs')}{' '}
              <Text as="span" opacity={0.75}>
                {model.inputs.length}
              </Text>
            </>
          ) : (
            t('workflowLibrary.preview.inputs')
          ),
        },
      ]
    : []),
  ...(hasGraph ? [{ id: 'graph' as const, label: t('workflowLibrary.preview.graph') }] : []),
];

export interface WorkflowPreviewFind {
  query: string;
  matches: number;
  onChange: (event: ChangeEvent<HTMLInputElement>) => void;
  onKeyDown: (event: ReactKeyboardEvent<HTMLInputElement>) => void;
  /** The graph's `onMatchCountChange`: only it knows whether every node is drawn. */
  onMatchCountChange: (count: number) => void;
  /** Empties the field, for a graph drawn afresh. */
  reset: () => void;
}

/**
 * The find field's state. It belongs to the drawing it searched (`drawing` names it): another document, or this one
 * replaced, starts unfiltered.
 */
export const useWorkflowPreviewFind = (
  graphRef: RefObject<WorkflowDocumentGraphHandle | null>,
  drawing: number
): WorkflowPreviewFind => {
  const [find, setFind] = useState({ drawing: -1, matches: 0, query: '' });
  const onChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const query = event.currentTarget.value;

      setFind({ drawing, matches: 0, query });
      graphRef.current?.find(query);
    },
    [drawing, graphRef]
  );
  const onKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLInputElement>) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        graphRef.current?.revealNextMatch();
      }
    },
    [graphRef]
  );
  const onMatchCountChange = useCallback((matches: number) => setFind((current) => ({ ...current, matches })), []);
  const reset = useCallback(() => setFind({ drawing, matches: 0, query: '' }), [drawing]);

  return {
    matches: find.matches,
    onChange,
    onKeyDown,
    onMatchCountChange,
    query: find.drawing === drawing ? find.query : '',
    reset,
  };
};

export const WorkflowPreviewFindField = ({ css, find }: { css?: SystemStyleObject; find: WorkflowPreviewFind }) => {
  const { t } = useTranslation();

  return (
    <HStack css={css} gap="2" minW="0">
      {find.query ? (
        <Text aria-live="polite" color="fg.muted" fontSize="md" whiteSpace="nowrap">
          {find.matches === 0
            ? t('workflowLibrary.preview.findNone')
            : t('workflowLibrary.preview.findMatches', { count: find.matches })}
        </Text>
      ) : null}
      <InputGroup startElement={FIND_START_ELEMENT} w="11rem">
        <Input
          aria-label={t('workflowLibrary.preview.find')}
          data-preview-find=""
          placeholder={t('workflowLibrary.preview.find')}
          type="search"
          value={find.query}
          onChange={find.onChange}
          onKeyDown={find.onKeyDown}
        />
      </InputGroup>
    </HStack>
  );
};

/** A message in the graph's place: why there is nothing to draw. `alert` for a failure the reader should hear. */
export const WorkflowPreviewStageMessage = ({
  children,
  isAlert = false,
}: {
  children: ReactNode;
  isAlert?: boolean;
}) => (
  <Center h="full" p="6">
    <Text color="fg.muted" fontSize="lg" maxW="32rem" role={isAlert ? 'alert' : undefined} textAlign="center">
      {children}
    </Text>
  </Center>
);

export const WorkflowPreviewStageLoading = () => {
  const { t } = useTranslation();

  return (
    <Center aria-live="polite" flexDirection="column" gap="2" h="full" role="status">
      <Spinner color="fg.subtle" size="lg" />
      <Text color="fg.muted" fontSize="md">
        {t('workflowLibrary.preview.loading')}
      </Text>
    </Center>
  );
};
