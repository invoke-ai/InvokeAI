import { handlers, parseAndRecallAllMetadata, parseAndRecallPrompts } from 'features/metadata/util/handlers';
import { useCallback, useEffect, useState } from 'react';
import { useDebouncedMetadata } from 'services/api/hooks/useDebouncedMetadata';

export const useImageActions = (image_name?: string) => {
  const { metadata, isLoading: isLoadingMetadata } = useDebouncedMetadata(image_name);
  const [hasMetadata, setHasMetadata] = useState(false);
  const [hasSeed, setHasSeed] = useState(false);
  const [hasPrompts, setHasPrompts] = useState(false);

  useEffect(() => {
    const parseMetadata = async () => {
      if (metadata) {
        setHasMetadata(true);
        try {
          await handlers.seed.parse(metadata);
          setHasSeed(true);
        } catch {
          setHasSeed(false);
        }

        const promptParseResults = await Promise.allSettled([
          handlers.positivePrompt.parse(metadata),
          handlers.negativePrompt.parse(metadata),
          handlers.sdxlPositiveStylePrompt.parse(metadata),
          handlers.sdxlNegativeStylePrompt.parse(metadata),
        ]);
        if (promptParseResults.some((result) => result.status === 'fulfilled')) {
          setHasPrompts(true);
        } else {
          setHasPrompts(false);
        }
      } else {
        setHasMetadata(false);
        setHasSeed(false);
        setHasPrompts(false);
      }
    };
    parseMetadata();
  }, [metadata]);

  const recallAll = useCallback(() => {
    parseAndRecallAllMetadata(metadata);
  }, [metadata]);

  const remix = useCallback(() => {
    // Recalls all metadata parameters except seed
    parseAndRecallAllMetadata(metadata, ['seed']);
  }, [metadata]);

  const recallSeed = useCallback(() => {
    handlers.seed.parse(metadata).then((seed) => {
      handlers.seed.recall && handlers.seed.recall(seed);
    });
  }, [metadata]);

  const recallPrompts = useCallback(() => {
    parseAndRecallPrompts(metadata);
  }, [metadata]);

  return { recallAll, remix, recallSeed, recallPrompts , hasMetadata, hasSeed,  hasPrompts, isLoadingMetadata };
};
