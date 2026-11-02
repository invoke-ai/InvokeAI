import type {
  ExpandPromptSuggestion,
  GenerateLora,
  MainModelConfig,
  PromptHistoryItem,
} from '@features/generation/contracts';
import type { VideoWidgetValues } from '@features/video/core/types';

import { Stack } from '@chakra-ui/react';
import { NegativePromptField, PositivePromptField } from '@features/generation/components';
import { normalizeVideoWidgetValues } from '@features/video/core/settings';
import { flushWorkbenchDrafts } from '@platform/react/draftRegistry';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { areVideoLorasEquivalent, areVideoModelsEquivalent } from './videoComparators';
import { useVideoUiActions } from './VideoUiContext';

type VideoPrompts = Pick<VideoWidgetValues, 'negativePrompt' | 'negativePromptEnabled' | 'positivePrompt'>;

const PROMPT_KEYS = ['negativePrompt', 'negativePromptEnabled', 'positivePrompt'] as const;

/** The earlier value of each key a recall changed, where it still holds what the recall wrote; edits since stay. */
const getRecallRevert = (before: VideoPrompts, written: VideoPrompts, current: VideoPrompts): Partial<VideoPrompts> =>
  Object.fromEntries(
    PROMPT_KEYS.filter((key) => before[key] !== written[key] && current[key] === written[key]).map((key) => [
      key,
      before[key],
    ])
  );

/**
 * Video prompts are independent of the shared image-generation draft. Compare content because patches recreate
 * values and remounts would lose autocomplete state.
 */
export const VideoPromptFields = memo(
  function VideoPromptFields({
    expandPromptSuggestion,
    loras,
    model,
    negativeHelpText,
    negativePrompt,
    negativePromptEnabled,
    negativePromptHeightPx,
    negativeVisible,
    onPatchValues,
    positivePrompt,
    positivePromptHeightPx,
    projectId,
    showSyntaxHighlighting,
  }: {
    expandPromptSuggestion: ExpandPromptSuggestion | null;
    loras: GenerateLora[];
    model: MainModelConfig | null;
    negativeHelpText?: string;
    negativePrompt: string;
    negativePromptEnabled: boolean;
    negativePromptHeightPx: number;
    /** From the video prompt policy: MiniMax H3 has no negative conditioning at all. */
    negativeVisible: boolean;
    onPatchValues: (patch: Partial<VideoWidgetValues>) => void;
    positivePrompt: string;
    positivePromptHeightPx: number;
    projectId: string;
    showSyntaxHighlighting: boolean;
  }) {
    const { t } = useTranslation();
    const { notifications, readValues } = useVideoUiActions();
    // The positive prompt also lands in the prompt box as an edit (Ctrl+Z); replacing the negative prompt as well
    // needs the toast's Undo.
    const handleUsePrompt = useCallback(
      (prompt: PromptHistoryItem) => {
        // A hidden negative prompt is not this model's to replace.
        if (!negativeVisible) {
          onPatchValues({ positivePrompt: prompt.positivePrompt });
          return;
        }

        // The field flushed its draft before calling, so the stored values are the ones on screen.
        const before: VideoPrompts = normalizeVideoWidgetValues(readValues()) ?? {
          negativePrompt,
          negativePromptEnabled,
          positivePrompt,
        };
        const written: VideoPrompts = {
          negativePrompt: prompt.negativePrompt ?? '',
          negativePromptEnabled: prompt.negativePrompt ? true : before.negativePromptEnabled,
          positivePrompt: prompt.positivePrompt,
        };

        onPatchValues(written);

        if (
          written.negativePrompt === before.negativePrompt &&
          written.negativePromptEnabled === before.negativePromptEnabled
        ) {
          return;
        }

        notifications.undoable(t('widgets.generate.promptHistoryUsed'), {
          onUndo: () => {
            // Typing still inside a draft's debounce counts as an edit since, so it lands before the comparison.
            flushWorkbenchDrafts();
            const current = normalizeVideoWidgetValues(readValues());
            const revert = current ? getRecallRevert(before, written, current) : {};

            if (Object.keys(revert).length > 0) {
              onPatchValues(revert);
            } else {
              notifications.info(t('widgets.generate.undoStale'));
            }
          },
          supersedes: `video-prompt:${projectId}`,
        });
      },
      [
        negativePrompt,
        negativePromptEnabled,
        negativeVisible,
        notifications,
        onPatchValues,
        positivePrompt,
        projectId,
        readValues,
        t,
      ]
    );
    const handlePositiveChange = useCallback(
      (nextPositivePrompt: string) => onPatchValues({ positivePrompt: nextPositivePrompt }),
      [onPatchValues]
    );
    const handleNegativeChange = useCallback(
      (nextNegativePrompt: string) => onPatchValues({ negativePrompt: nextNegativePrompt }),
      [onPatchValues]
    );
    const handleNegativeEnabledChange = useCallback(
      (nextNegativePromptEnabled: boolean) => onPatchValues({ negativePromptEnabled: nextNegativePromptEnabled }),
      [onPatchValues]
    );
    const handlePositiveResizeEnd = useCallback(
      (positivePromptHeight: number) => onPatchValues({ positivePromptHeightPx: positivePromptHeight }),
      [onPatchValues]
    );
    const handleNegativeResizeEnd = useCallback(
      (negativePromptHeight: number) => onPatchValues({ negativePromptHeightPx: negativePromptHeight }),
      [onPatchValues]
    );

    return (
      <Stack gap="2" p="2">
        <PositivePromptField
          expandPromptSuggestion={expandPromptSuggestion}
          heightPx={positivePromptHeightPx}
          loras={loras}
          projectId={projectId}
          selectedModel={model ?? undefined}
          showSyntaxHighlighting={showSyntaxHighlighting}
          value={positivePrompt}
          onChange={handlePositiveChange}
          onResizeEnd={handlePositiveResizeEnd}
          onUsePrompt={handleUsePrompt}
        />
        {negativeVisible ? (
          <NegativePromptField
            heightPx={negativePromptHeightPx}
            helpText={negativeHelpText}
            isEnabled={negativePromptEnabled}
            loras={loras}
            projectId={projectId}
            selectedModel={model ?? undefined}
            showSyntaxHighlighting={showSyntaxHighlighting}
            value={negativePrompt}
            onChange={handleNegativeChange}
            onEnabledChange={handleNegativeEnabledChange}
            onResizeEnd={handleNegativeResizeEnd}
          />
        ) : null}
      </Stack>
    );
  },
  (previous, next) =>
    previous.expandPromptSuggestion === next.expandPromptSuggestion &&
    previous.negativePrompt === next.negativePrompt &&
    previous.negativePromptEnabled === next.negativePromptEnabled &&
    previous.negativePromptHeightPx === next.negativePromptHeightPx &&
    previous.negativeVisible === next.negativeVisible &&
    previous.negativeHelpText === next.negativeHelpText &&
    previous.onPatchValues === next.onPatchValues &&
    previous.positivePrompt === next.positivePrompt &&
    previous.positivePromptHeightPx === next.positivePromptHeightPx &&
    previous.projectId === next.projectId &&
    previous.showSyntaxHighlighting === next.showSyntaxHighlighting &&
    areVideoModelsEquivalent(previous.model, next.model) &&
    areVideoLorasEquivalent(previous.loras, next.loras)
);
