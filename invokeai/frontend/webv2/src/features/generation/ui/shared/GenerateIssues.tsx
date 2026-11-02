import type { GenerateComponentValueKey } from '@features/generation/core/baseGenerationPolicies';
import type {
  GenerateSectionId,
  GenerationIssue,
  GenerationIssueField,
  GenerationIssueFix,
} from '@features/generation/core/generationIssues';
import type { ReactNode } from 'react';

import { Badge, HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { describeGenerationIssue, describeGenerationIssueFix } from '@features/generation/core/generationIssues';
import { Button } from '@platform/ui/Button';
import { CircleAlertIcon } from 'lucide-react';
import { createContext, use, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

interface GenerateIssuesValue {
  issues: readonly GenerationIssue[];
  applyFix: (fix: GenerationIssueFix) => void;
}

const GenerateIssuesContext = createContext<GenerateIssuesValue | null>(null);

/** The form's Invoke-blocking issues, for its sections' badges and the notes under their fields. */
export const GenerateIssuesProvider = ({
  applyFix,
  children,
  issues,
}: GenerateIssuesValue & { children: ReactNode }) => {
  const value = useMemo(() => ({ applyFix, issues }), [applyFix, issues]);

  return <GenerateIssuesContext value={value}>{children}</GenerateIssuesContext>;
};

const NO_ISSUES: readonly GenerationIssue[] = [];

/** Issues with the same ids read the same, so a re-validation that finds nothing new keeps the list identity. */
export const areGenerationIssuesEqual = (
  left: readonly GenerationIssue[],
  right: readonly GenerationIssue[]
): boolean =>
  left.length === right.length &&
  left.every(
    (issue, index) =>
      issue.id === right[index]?.id &&
      issue.message === right[index]?.message &&
      JSON.stringify(issue.fix) === JSON.stringify(right[index]?.fix)
  );

/** A collapsed section names how many problems it holds, so a closed section never hides why Invoke is blocked. */
export const GenerateSectionIssueBadge = ({ section }: { section: string }) => {
  const { t } = useTranslation();
  const issues = use(GenerateIssuesContext)?.issues ?? NO_ISSUES;
  const count = issues.filter((issue) => issue.target.section === section).length;

  return count > 0 ? (
    <Badge colorPalette="red" variant="surface">
      <Icon as={CircleAlertIcon} boxSize="3" />
      {t('widgets.generate.issueCount', { count })}
    </Badge>
  ) : null;
};

/**
 * Where each issue field shows its issue. Notes render under the control; the rest show the message themselves. A
 * component slot shows its own through the slot picker. A Record, so a new field cannot go without a renderer.
 */
export const ISSUE_FIELD_RENDERERS: Record<
  Exclude<GenerationIssueField, GenerateComponentValueKey>,
  'notes' | 'own message'
> = {
  cfgScale: 'notes',
  // Under the row of the concept it names, beside the row's "Not installed" badge.
  concepts: 'notes',
  krea2RebalanceWeights: 'own message',
  mossImageUncondSkipLayers: 'own message',
  // On the card of the reference image it names.
  referenceImage: 'notes',
  referenceImages: 'notes',
};

const FIX_BUTTON_SELECTOR = '[data-issue-fix]';

/**
 * After a fix resolves its issue the button unmounts; focus moves to the next fix in the same notes, else to the
 * section's header, so it never falls to the page.
 */
const moveFocusAfterFix = (button: HTMLElement) => {
  const notes = button.closest('[data-issue-notes]');
  const section = button.closest('.generate-section');
  const following = notes ? [...notes.querySelectorAll<HTMLElement>(FIX_BUTTON_SELECTOR)] : [];
  const nextIndex = following.indexOf(button) + 1;

  requestAnimationFrame(() => {
    if (button.isConnected) {
      return;
    }

    const next = [...following.slice(nextIndex), ...following.slice(0, Math.max(0, nextIndex - 1))].find(
      (candidate) => candidate.isConnected
    );

    (next ?? section?.querySelector<HTMLElement>('[data-scope="collapsible"][data-part="trigger"]'))?.focus();
  });
};

/**
 * The issues a section shows itself, each with its one-step fix. Without `field` it lists the section's unplaced
 * issues. `fixesOnly` is for a control that already shows the message (the guidance scrubber): only the fix is added.
 */
export const GenerateIssueNotes = ({
  conceptKey,
  field,
  fixesOnly = false,
  referenceImageId,
  section,
}: {
  /** Only the issues about this concept, for notes under its row. */
  conceptKey?: string;
  field?: GenerationIssueField;
  fixesOnly?: boolean;
  /** Only the issues about this reference image, for notes on its card. */
  referenceImageId?: string;
  section: GenerateSectionId;
}) => {
  const { t } = useTranslation();
  const context = use(GenerateIssuesContext);
  const issues = (context?.issues ?? NO_ISSUES).filter(
    (issue) =>
      issue.target.section === section &&
      issue.target.field === field &&
      (!fixesOnly || issue.fix !== undefined) &&
      (conceptKey === undefined || (issue.fix?.kind === 'removeConcept' && issue.fix.modelKey === conceptKey)) &&
      (referenceImageId === undefined || issue.target.referenceImageId === referenceImageId)
  );

  if (!context || issues.length === 0) {
    return null;
  }

  return (
    <Stack data-issue-notes="" gap="1">
      {issues.map((issue) => {
        const message = describeGenerationIssue(t, issue);

        return (
          <HStack
            key={issue.id}
            alignItems="flex-start"
            data-issue-id={issue.id}
            gap="1.5"
            justifyContent={fixesOnly ? 'flex-end' : undefined}
          >
            {fixesOnly ? null : (
              <>
                <Icon as={CircleAlertIcon} boxSize="3" color="fg.error" flexShrink={0} mt="px" />
                <Text color="fg.error" flex="1" fontSize="xs" minW="0" textWrap="pretty">
                  {message}
                </Text>
              </>
            )}
            {issue.fix ? (
              <IssueFixButton
                fix={issue.fix}
                label={describeGenerationIssueFix(t, issue.fix)}
                name={t('widgets.generate.issueFixNamed', {
                  fix: describeGenerationIssueFix(t, issue.fix),
                  issue: message,
                })}
                onFix={context.applyFix}
              />
            ) : null}
          </HStack>
        );
      })}
    </Stack>
  );
};

const IssueFixButton = ({
  fix,
  label,
  name,
  onFix,
}: {
  fix: GenerationIssueFix;
  label: string;
  /** The label followed by the issue it fixes, so the button names its target. */
  name: string;
  onFix: (fix: GenerationIssueFix) => void;
}) => (
  <Button
    aria-label={name}
    data-issue-fix=""
    flexShrink={0}
    size="xs"
    variant="outline"
    // oxlint-disable-next-line react-perf/jsx-no-new-function-as-prop -- one button per shown issue.
    onClick={(event) => {
      const button = event.currentTarget;

      onFix(fix);
      moveFocusAfterFix(button);
    }}
  >
    {label}
  </Button>
);

/** How many issues a section holds, for sections that place their own badge. */
export const useGenerateSectionIssues = (section: GenerateSectionId): readonly GenerationIssue[] => {
  const issues = use(GenerateIssuesContext)?.issues ?? NO_ISSUES;

  return useMemo(() => issues.filter((issue) => issue.target.section === section), [issues, section]);
};
